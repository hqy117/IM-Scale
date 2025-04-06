import os
import os.path
import datetime
import time
import numpy as np
from scipy import*
from copy import*
import sys
import pandas as pd
import shutil
from tqdm import tqdm
import glob
import dimod
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pickle

import dwave.inspector


## DATA
class CustomDataset(Dataset):
    def __init__(self, images, labels=None):
        self.x = images
        self.y = labels

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        return (data, target)

    def __len__(self):
        return (len(self.x))


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes, args):
        self.number_classes = number_classes
        self.outputlayer = args.layersList[2]
        self.mode = args.mode

    def __call__(self, target):
        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        if self.mode == "qubo":
            target_onehot = torch.zeros((1, self.number_classes))
        elif self.mode == "ising":
            target_onehot = -1*torch.ones((1, self.number_classes))

        return target_onehot.scatter_(1, target.long(), 1).repeat_interleave(int(self.outputlayer/self.number_classes)).squeeze(0)


class DefineDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, target_transforms=None):
        self.x = images
        self.y = labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        if self.transforms:
            data = self.transforms(data)

        if self.target_transforms:
            target = self.target_transforms(target)

        if self.y is not None:
            return (data, target)
        else:
            return data

    def __len__(self):
        return (len(self.x))


def generate_digits(args):
    '''
    Generate the dataloaders for digits dataset
    '''
    digits = load_digits()

    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1, random_state=10, shuffle=True)
    normalisation = 8
    x_train, x_test = x_train / normalisation, x_test / normalisation

    train_data = DefineDataset(x_train, labels=y_train, target_transforms=ReshapeTransformTarget(10, args))
    test_data = DefineDataset(x_test, labels=y_test, target_transforms=ReshapeTransformTarget(10, args))

    ## Data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader


def generate_mnist(args):
    '''
    Generate mnist dataloaders - 1000 training images, 100 testing images
    '''
    N_class = 10
    N_data = args.N_data
    N_data_test = args.N_data_test

    # Use a consistent dataset directory if provided
    data_root = './data'
    if hasattr(args, 'dataset_dir'):
        data_root = args.dataset_dir

    with torch.no_grad():
        # Check if data_augmentation attribute exists, default to False if not
        data_augmentation = getattr(args, 'data_augmentation', False)
        
        if data_augmentation:
            transforms_train=[torchvision.transforms.ToTensor(), torchvision.transforms.RandomAffine(10, translate=(0.04, 0.04), scale=None, shear=None, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=0), ReshapeTransform((-1,))]
        else:
            transforms_train=[torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

        transforms_test=[torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

        #Training data
        mnist_train = torchvision.datasets.MNIST(root=data_root, train=True, download=True,
                                                transform=torchvision.transforms.Compose(transforms_train),
                                                target_transform=ReshapeTransformTarget(10, args))

        mnist_train_data, mnist_train_targets, comp = torch.empty(N_data,28,28,dtype=mnist_train.data.dtype), torch.empty(N_data,dtype=mnist_train.targets.dtype), torch.zeros(N_class)
        idx_0, idx_1 = 0, 0
        
        # Safety check to prevent infinite loop
        max_attempts = 100000
        attempts = 0
        
        while idx_1 < N_data and attempts < max_attempts:
            if idx_0 >= len(mnist_train.targets):
                # If we've gone through all the data, reset to beginning
                idx_0 = 0
                # If we've cycled through data but still haven't filled our requirement,
                # we might need to allow more samples per class
                if idx_1 < N_data:
                    # Increase the quota per class
                    comp = torch.zeros(N_class)
                
            class_data = mnist_train.targets[idx_0]
            per_class_limit = max(int(N_data/N_class), 1)  # Ensure at least 1 per class
            
            if comp[class_data] < per_class_limit:
                mnist_train_data[idx_1,:,:] = mnist_train.data[idx_0,:,:].clone()
                mnist_train_targets[idx_1] = class_data.clone()
                comp[class_data] += 1
                idx_1 += 1
            idx_0 += 1
            attempts += 1

        mnist_train.data, mnist_train.targets = mnist_train_data, mnist_train_targets

        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = args.batch_size, shuffle=True)

        #Testing data
        mnist_test = torchvision.datasets.MNIST(root=data_root, train=False, download=True,
                                                transform=torchvision.transforms.Compose(transforms_test),
                                                target_transform=ReshapeTransformTarget(10, args))

        mnist_test_data, mnist_test_targets, comp = torch.empty(N_data_test,28,28,dtype=mnist_test.data.dtype), torch.empty(N_data_test,dtype=mnist_test.targets.dtype), torch.zeros(N_class)
        idx_0, idx_1 = 0, 0
        
        # Safety check to prevent infinite loop
        attempts = 0
        
        while idx_1 < N_data_test and attempts < max_attempts:
            if idx_0 >= len(mnist_test.targets):
                # If we've gone through all the data, reset to beginning
                idx_0 = 0
                # If we've cycled through data but still haven't filled our requirement,
                # we might need to allow more samples per class
                if idx_1 < N_data_test:
                    # Increase the quota per class
                    comp = torch.zeros(N_class)
                
            class_data = mnist_test.targets[idx_0]
            per_class_limit = max(int(N_data_test/N_class), 1)  # Ensure at least 1 per class
            
            if comp[class_data] < per_class_limit:
                mnist_test_data[idx_1,:,:] = mnist_test.data[idx_0,:,:].clone()
                mnist_test_targets[idx_1] = class_data.clone()
                comp[class_data] += 1
                idx_1 += 1
            idx_0 += 1
            attempts += 1

        mnist_test.data, mnist_test.targets = mnist_test_data, mnist_test_targets

        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 1, shuffle=False)

        return train_loader, test_loader, mnist_train

def createBQM(net, args, input, beta = 0, target = None, mode = None):

    with torch.no_grad():
        bias_input = np.matmul(input, net.weights_0)
        bias_lim = args.bias_lim
        h = {idx_loc: (bias + bias_input[idx_loc]).clip(-bias_lim,bias_lim).item() for idx_loc, bias in enumerate(net.bias_0)}

        if target is not None:
            if args.mode == "qubo":
                bias_nudge = 0.5*beta*(2*target-1)
            elif args.mode == "ising":
                bias_nudge = -beta*target
            h.update({idx_loc + args.layersList[1]: (bias + bias_nudge[idx_loc]).clip(-bias_lim,bias_lim).item() for idx_loc, bias in enumerate(net.bias_1)})
        else:
            h.update({idx_loc + args.layersList[1]: bias.clip(-bias_lim,bias_lim).item() for idx_loc, bias in enumerate(net.bias_1)})

        J = {}
        for k in range(args.layersList[1]):
            for j in range(args.layersList[2]):
                J.update({(k,j+args.layersList[1]): net.weights_1[k][j].clip(-1,1)})

        if args.mode == "qubo":
            model = dimod.BinaryQuadraticModel(h, J, 0, dimod.BINARY)
        elif args.mode == "ising":
            model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)

        if mode == "DEBUG":
            print(dimod.ExactSolver().sample_ising(h, J))

        return model


def train(net, args, train_loader, simu_sampler, exact_sampler, qpu_sampler):
    '''
    function to train the network for 1 epoch
    '''
    exact_pred, exact_loss = 0, 0
    qpu_pred, qpu_loss = 0, 0

    with torch.no_grad():
        for idx, (DATA, TARGET) in enumerate(tqdm(train_loader, position=0, leave=True)):
            store_seq = None
            store_s = None
            #if args.random_beta == 1:
            # net.sign_beta = torch.sign(torch.randn(1)).item()

            for k in range(DATA.size()[0]):
                data, target = DATA[k].numpy(), TARGET[k].numpy()

                # if args.data_augmentation and random.randint(0, 3) == 0: #0.25 de chance d'être augmenté
                #     data = augment_data(data)

                ## Free phase

                model = createBQM(net, args, data)

                # Simulated annealing sampling
                if args.simulated == 1:
                    qpu_seq = qpu_sampler.sample(model, num_reads = args.n_iter_free, num_sweeps = 100)

                # QPU sampling
                else:
                    qpu_seq = qpu_sampler.sample(model, num_reads = args.n_iter_free, chain_strength = args.chain_strength, auto_scale = args.auto_scale)

                ## Nudge phase: same system except bias for the output layer
                model = createBQM(net, args, data, beta = net.sign_beta * args.beta, target = target)
                #QPU sampling
                reverse_schedule = [[0.0, 1.0], [10, args.frac_anneal_nudge], [20, 1]]
                reverse_anneal_params = dict(anneal_schedule=reverse_schedule,
                                    initial_state=qpu_seq.first.sample,
                                    reinitialize_state=True)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                #print(qpu_seq.record["sample"][0].reshape(1,-1).shape)
                #print(qpu_seq.record["sample"][0].reshape(1,-1)[:,args.layersList[1]:])
                #print(qpu_seq.record["sample"][0].reshape(1,-1)[:,args.layersList[1]:][0].shape)
                #print(target)
                #print(target.shape)
                #print(np.array_equal(qpu_seq.record["sample"][0].reshape(1,-1)[:,args.layersList[1]:][0], target))

                # print("++++++++++++")
                # print("target="+str(target))
                if np.array_equal(qpu_seq.record["sample"][0].reshape(1,-1)[:,args.layersList[1]:][0], target):  #we skip if free state == target (the nudge won't bring anything)
                    qpu_s = qpu_seq
                else:
                    # Simulated reverse annealing
                    if args.simulated == 1:
                        qpu_s = qpu_sampler.sample(model, num_reads = args.n_iter_nudge, num_sweeps = 100, initial_states = qpu_seq.first.sample, reverse = True, fraction_annealed = args.frac_anneal_nudge)
                    # Quantum reverse annealing
                    else:
                        qpu_s = qpu_sampler.sample(model, num_reads = args.n_iter_nudge, chain_strength = args.chain_strength, auto_scale = args.auto_scale, **reverse_anneal_params)

                # print("qpu_seq - memory = " + str(sys.getsizeof(qpu_seq.record["sample"])))
                if store_seq is None:
                    store_seq = qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0]) #qpu_seq
                    store_s = qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0]) #qpu_s
                else:
                    store_seq = np.concatenate((store_seq, qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])),0)
                    store_s = np.concatenate((store_s, qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])),0)
                #
                # print("store_seq - memory = " + str(sys.getsizeof(store_seq)))
                # print("store_s - memory = " + str(sys.getsizeof(store_s)))

                del qpu_seq, qpu_s
                del data, target

                # attribs = [attrib for attrib in qpu_sampler.__dict__.keys()]
                # for attrib in attribs:
                #     print(attrib + " - memory = " + str(sys.getsizeof(getattr(qpu_sampler,attrib))))
                #
                # attribs = [attrib for attrib in net.__dict__.keys()]
                # for attrib in attribs:
                #     print(attrib + " - memory = " + str(sys.getsizeof(getattr(net,attrib))))

            seq = [store_seq[:,:args.layersList[1]], store_seq[:,args.layersList[1]:]]
            s   = [store_s[:,:args.layersList[1]], store_s[:,args.layersList[1]:]]


            ## Compute loss and error for QPU sampling
            loss, pred = net.computeLossAcc(seq, TARGET, args, stage = 'training')

            qpu_pred += pred
            qpu_loss += loss

            # print("pred - memory = " + str(sys.getsizeof(qpu_pred)))
            # print("loss - memory = " + str(sys.getsizeof(qpu_loss)))

            #net.updateParams(data, exact_s, exact_seq, args)
            net.updateParams(DATA, s, seq, args)
            net.sign_beta = 1

            del seq, s
            # del loss, pred
            del DATA, TARGET

    # attribs = [attrib for attrib in qpu_sampler.__dict__.keys()]
    # for attrib in attribs:
    #     print(attrib + " - memory = " + str(sys.getsizeof(getattr(qpu_sampler,attrib))))
    #
    # attribs = [attrib for attrib in net.__dict__.keys()]
    # for attrib in attribs:
    #     print(attrib + " - memory = " + str(sys.getsizeof(getattr(net,attrib))))

    return exact_loss, exact_pred, qpu_loss, qpu_pred


def test(net, args, test_loader, simu_sampler, exact_sampler, qpu_sampler):
    '''
    function to test the performance of the network on testset
    '''
    exact_pred, exact_loss = 0, 0
    qpu_pred, qpu_loss = 0, 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(test_loader, position=0, leave=True)):
            data, target = data[0].numpy(), target[0].numpy()

            seq = None
            exact_seq = None

            model = createBQM(net, args, data)

            #Simulated annealing sampling
            if args.simulated == 1:
                qpu_seq = qpu_sampler.sample(model, num_reads = args.n_iter_free, num_sweeps = 100)

                # Conver QUBO/Ising solution to seq format

                s = np.array([list(qpu_seq.record["sample"][0])])
                seq = []
                seq.append(s[:, :args.layersList[1]])
                seq.append(s[:, args.layersList[1]:])

            # Use exact solver
            #exact_seq = exact_solver(model)
            #loss, pred = net.computeLossError(exact_seq, target, args)
            
            # Compute loss and error
            loss, pred = net.computeLossAcc(seq, target.reshape(1,target.shape[0]), args, stage = 'testing')

            qpu_pred += pred
            qpu_loss += loss

            del qpu_seq, seq
            del data, target

    return exact_loss, exact_pred, qpu_loss, qpu_pred


def initDataframe(path, dataframe_to_init = 'results.csv'):

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + prefix + dataframe_to_init):
        dataframe = pd.read_csv(path + prefix + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['Exact_Train_Error','Exact_Test_Error','QPU_Train_Error','QPU_Test_Error','Exact_Train_Loss','Exact_Test_Loss','QPU_Train_Loss','QPU_Test_Loss']
        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')

    return dataframe


def updateDataframe(BASE_PATH, dataframe, exact_train_error, exact_test_error, qpu_train_error, qpu_test_error, exact_train_loss, exact_test_loss, qpu_train_loss, qpu_test_loss):

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    # Handle the case where exact_train_loss and others might be empty lists
    exact_train_loss_value = exact_train_loss[-1] if exact_train_loss and len(exact_train_loss) > 0 else 0
    exact_test_loss_value = exact_test_loss[-1] if exact_test_loss and len(exact_test_loss) > 0 else 0
    qpu_train_loss_value = qpu_train_loss[-1] if qpu_train_loss and len(qpu_train_loss) > 0 else 0
    qpu_test_loss_value = qpu_test_loss[-1] if qpu_test_loss and len(qpu_test_loss) > 0 else 0

    data = [exact_train_error, exact_test_error, qpu_train_error, qpu_test_error, 
            exact_train_loss_value, exact_test_loss_value, qpu_train_loss_value, qpu_test_loss_value]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix + 'results.csv')

    return dataframe


def create_binary_targets(targets, digit, args):
    """
    Convert multi-class targets to binary targets for a specific digit
    
    Args:
        targets: Original targets (multi-class)
        digit: The digit to create binary targets for (0-9)
        args: Arguments containing mode information
        
    Returns:
        Binary targets (1 for the specified digit, -1 or 0 for other digits)
    """
    # Debug the target structure
    print(f"Creating binary targets for digit {digit}")
    print(f"Target shape: {targets.shape}")
    
    # Get the expanded target size
    expand_output = args.layersList[2]  # Use all output neurons as binary output
    
    # Create binary targets
    binary_targets = torch.zeros((targets.size(0), expand_output))
    
    # Find instances of the target digit
    if args.mode == "ising":
        # For ising mode, use -1 and 1
        binary_targets.fill_(-1)
        
        # Determine which samples match the target digit
        # The target representation depends on the ReshapeTransformTarget class
        # We need to check if any of the target positions has the value 1
        # as that indicates this sample belongs to our digit
        
        # For each sample, check if it's our target digit
        for i in range(targets.size(0)):
            # Check the target - if any position has 1, check which digit it corresponds to
            non_negative_positions = (targets[i] > 0).nonzero(as_tuple=True)[0]
            if len(non_negative_positions) > 0:
                # If any non-negative position corresponds to our digit class
                # Note: This is a heuristic approach, might need adjustment based on actual encoding
                target_digit = non_negative_positions[0] // (args.layersList[2] // 10)
                if target_digit == digit:
                    binary_targets[i, :] = 1
    else:  # qubo mode
        # For qubo mode, use 0 and 1
        binary_targets.fill_(0)
        
        # Similar approach for qubo mode
        for i in range(targets.size(0)):
            non_zero_positions = (targets[i] > 0.5).nonzero(as_tuple=True)[0]
            if len(non_zero_positions) > 0:
                target_digit = non_zero_positions[0] // (args.layersList[2] // 10)
                if target_digit == digit:
                    binary_targets[i, :] = 1
    
    # Verify the binary target distribution
    positive_count = (binary_targets[:, 0] > 0).sum().item()
    negative_count = (binary_targets[:, 0] < 0).sum().item()
    print(f"Binary targets: {positive_count} positive samples, {negative_count} negative samples")
    
    return binary_targets


def createPath(args, digit=None, simu='mini-batch'):
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    # Use the provided results directory if available
    if hasattr(args, 'results_dir'):
        BASE_PATH = args.results_dir
    else:
        # Legacy path creation
        if os.name != 'posix':
            BASE_PATH = prefix + prefix + "?" + prefix + os.getcwd()
        else:
            BASE_PATH = '' + os.getcwd()

        if digit is not None:
            BASE_PATH += prefix + f'DATA-Binary-Classifier-{digit}'
        else:
            BASE_PATH += prefix + 'DATA-SA-'

        BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")

    # Create the base path if it doesn't exist
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    # Try to copy plotFunction.py, but don't fail if not found
    try:
        filePath = shutil.copy('plotFunction.py', BASE_PATH)
    except FileNotFoundError:
        try:
            filePath = shutil.copy('MLP/Binary-Classifiers/plotFunction.py', BASE_PATH)
        except FileNotFoundError:
            print("Warning: plotFunction.py not found - continuing without it")

    # Determine the experiment subfolder (S-1, S-2, etc.)
    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))
    if '.DS_Store' in files:
        files.pop(files.index('.DS_Store'))

    # Find experiment subfolder
    experiment_path = None
    if not files:
        experiment_path = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        for names in files:
            if '-' in names:
                try:
                    tab.append(int(names.split('-')[1]))
                except (ValueError, IndexError):
                    continue
        if not tab:
            experiment_path = BASE_PATH + prefix + 'S-1'
        elif args.load_model == 0:
            experiment_path = BASE_PATH + prefix + 'S-' + str(max(tab)+1)
        elif args.load_model > 0:
            experiment_path = BASE_PATH + prefix + 'S-' + str(args.load_model)

    # Create experiment directories
    if args.load_model == 0:
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(experiment_path + prefix + "Visu_weights", exist_ok=True)
        os.makedirs(experiment_path + prefix + "Confusion_matrices", exist_ok=True)

    return experiment_path


def saveHyperparameters(BASE_PATH, args, digit=None, simu='mini-batch'):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    if digit is not None:
        f.write(f"Binary Classifier for Digit {digit} with Equilibrium Propagation \n")
    else:
        f.write("Equilibrium Propagation with d'wave substrat \n")
    
    f.write('   Parameters of the simulation \n ')
    if simu == 'mini-batch':
        f.write('Averaging on a mini-batch \n ')
    elif simu == 'binary-classifier':
        f.write('Binary classification task \n ')
    else:
        f.write('Averaging on a single data point \n ')
    f.write('\n')

    for key in args.__dict__.keys():
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()


def visualize_dataset(dataset, label = 'train'):
    colormap = np.array(['b', 'r', 'g', 'y'])
    xdata = dataset[0]
    categories = dataset[1]
    categories = np.argmax(categories, axis=1).astype(int)
    plt.scatter(xdata[:,0], xdata[:,1], color = colormap[categories])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(label)
    plt.show()


def save_model_numpy(path, net):
    '''
    Save the parameters of the model as a dictionnary in a pickel file
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    with open(path + prefix + 'model_parameters.pickle', 'wb') as f:
            pickle.dump(net, f)

    return 0


def load_model_numpy(path):
    '''
    Save the parameters of the model as a dictionnary in a pickel file
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    with open(path + prefix + 'model_parameters.pickle', 'rb') as f:
            net = pickle.load(f)

    return net

def plot_functions(net, BASE_PATH, prefix, epoch, args):
    '''
    Plot Weights of the network along the training
    Plot Weights histogram along the training
    '''
    path = BASE_PATH + prefix + "Confusion_matrices" + prefix

    offset = int(len(os.listdir(path))/2)
    if args.load_model > 0:
        offset += 1

    if epoch > -1:
        plt.figure()
        plt.imshow(net.confusion_matrix_train)
        for i in range(10):
            for j in range(10):
                plt.text(i, j, str(net.confusion_matrix_train[j,i]), va='center', ha='center')

        plt.xlabel("Predictions - train")
        plt.ylabel("Target")
        plt.savefig(path + 'Confusion_matrix_train_epoch' + str(epoch+offset) + '.png')
        plt.close()

        plt.figure()
        plt.imshow(net.confusion_matrix_test)
        for i in range(10):
            for j in range(10):
                plt.text(i, j, str(net.confusion_matrix_test[j,i]), va='center', ha='center')

        plt.xlabel("Predictions - test")
        plt.ylabel("Target")
        plt.savefig(path + 'Confusion_matrix_test_epoch' + str(epoch+offset) + '.png')
        plt.close()

    if args.load_model == 0:
        path = BASE_PATH + prefix + "Visu_weights" + prefix

        plt.hist(net.weights_0)# bins = np.arange(100), align = 'left',rwidth = 0.9)
        plt.savefig(path + 'histo_weights_0_epoch' + str(epoch+offset) + '.png')
        plt.close()

        plt.hist(net.weights_1)# bins = np.arange(100), align = 'left',rwidth = 0.9)
        plt.savefig(path + 'histo_weights_1_epoch' + str(epoch+offset) + '.png')
        plt.close()

def train_binary(net, args, train_loader, simu_sampler, exact_sampler, qpu_sampler):
    '''
    function to train the binary network for 1 epoch
    Simplified version of train function for binary classification
    '''
    exact_pred, exact_loss = 0, 0
    qpu_pred, qpu_loss = 0, 0

    with torch.no_grad():
        for idx, (DATA, TARGET) in enumerate(tqdm(train_loader, position=0, leave=True, desc="Training")):
            store_seq = None
            store_s = None

            for k in range(DATA.size()[0]):
                data, target = DATA[k].numpy(), TARGET[k].numpy()

                ## Free phase
                model = createBQM(net, args, data)

                # Simulated annealing sampling
                if args.simulated == 1:
                    qpu_seq = qpu_sampler.sample(model, num_reads=args.n_iter_free, num_sweeps=100)
                # QPU sampling
                else:
                    qpu_seq = qpu_sampler.sample(model, num_reads=args.n_iter_free, chain_strength=args.chain_strength, auto_scale=args.auto_scale)

                ## Nudge phase: same system except bias for the output layer
                model = createBQM(net, args, data, beta=net.sign_beta * args.beta, target=target)
                
                # Skip if free state already matches target
                if np.array_equal(qpu_seq.record["sample"][0].reshape(1,-1)[:,args.layersList[1]:][0], target):
                    qpu_s = qpu_seq
                else:
                    # Simulated reverse annealing
                    if args.simulated == 1:
                        qpu_s = qpu_sampler.sample(model, num_reads=args.n_iter_nudge, num_sweeps=100, 
                                                 initial_states=qpu_seq.first.sample, reverse=True, 
                                                 fraction_annealed=args.frac_anneal_nudge)
                    # Quantum reverse annealing
                    else:
                        reverse_schedule = [[0.0, 1.0], [10, args.frac_anneal_nudge], [20, 1]]
                        reverse_anneal_params = dict(anneal_schedule=reverse_schedule,
                                           initial_state=qpu_seq.first.sample,
                                           reinitialize_state=True)
                        qpu_s = qpu_sampler.sample(model, num_reads=args.n_iter_nudge, 
                                                 chain_strength=args.chain_strength, 
                                                 auto_scale=args.auto_scale, **reverse_anneal_params)

                if store_seq is None:
                    store_seq = qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])
                    store_s = qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])
                else:
                    store_seq = np.concatenate((store_seq, qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])),0)
                    store_s = np.concatenate((store_s, qpu_s.record["sample"][0].reshape(1, qpu_s.record["sample"][0].shape[0])),0)

                del qpu_seq, qpu_s
                del data, target

            seq = [store_seq[:,:args.layersList[1]], store_seq[:,args.layersList[1]:]]
            s   = [store_s[:,:args.layersList[1]], store_s[:,args.layersList[1]:]]

            ## Compute loss and error for binary classification
            loss, pred = net.computeLossAcc(seq, TARGET, args, stage='training')

            qpu_pred += pred
            qpu_loss += loss

            # Update network parameters
            net.updateParams(DATA, s, seq, args)
            net.sign_beta = 1

            del seq, s
            del DATA, TARGET

    return exact_loss, exact_pred, qpu_loss, qpu_pred


def test_binary(net, args, test_loader, simu_sampler, exact_sampler, qpu_sampler):
    '''
    function to test the binary network
    Simplified version of test function for binary classification
    '''
    exact_pred, exact_loss = 0, 0
    qpu_pred, qpu_loss = 0, 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(test_loader, position=0, leave=True, desc="Testing")):
            data, target = data[0].numpy(), target[0].numpy()

            seq = None
            model = createBQM(net, args, data)

            # Sampling
            if args.simulated == 1:
                qpu_seq = qpu_sampler.sample(model, num_reads=args.n_iter_free, num_sweeps=100)

                # Convert to seq format
                s = np.array([list(qpu_seq.record["sample"][0])])
                seq = []
                seq.append(s[:, :args.layersList[1]])
                seq.append(s[:, args.layersList[1]:])
            else:
                qpu_seq = qpu_sampler.sample(model, num_reads=args.n_iter_free, 
                                          chain_strength=args.chain_strength, 
                                          auto_scale=args.auto_scale)
                qpu_seq = qpu_seq.record["sample"][0].reshape(1, qpu_seq.record["sample"][0].shape[0])
                seq = [qpu_seq[:, :args.layersList[1]], qpu_seq[:, args.layersList[1]:]]

            # Compute loss and error
            loss, pred = net.computeLossAcc(seq, target.reshape(1,target.shape[0]), args, stage='testing')

            qpu_pred += pred
            qpu_loss += loss

            del qpu_seq, seq
            del data, target

    return exact_loss, exact_pred, qpu_loss, qpu_pred