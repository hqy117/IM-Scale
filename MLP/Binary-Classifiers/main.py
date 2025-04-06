import dimod
import neal
import matplotlib.pyplot as plt
import numpy as np
import argparse
from math import*
from Tools import*
from Network import*
from random import*
from tqdm import tqdm
import os
import copy
import shutil

import torch
import torchvision

import dwave.inspector
from dwave.system import EmbeddingComposite, DWaveSampler, DWaveCliqueSampler, LazyFixedEmbeddingComposite
from simulated_sampler import SimulatedAnnealingSampler

# Set consistent paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'binary_classifiers')

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

parser = argparse.ArgumentParser(description='Binary Classifiers with Equilibrium Propagation using Simulated Annealing')
#Architecture settings
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='Dataset we use for training (default=mnist)')
parser.add_argument(
    '--simulated',
    type=int,
    default=1,
    help='specify if we use simulated annealing (=1) or quantum annealing (=0) (default=1)')
parser.add_argument(
    '--mode',
    type=str,
    default='ising',
    help='Which problem we submit to the annealer (default=ising (-1/+1 variables))')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 120, 4],
    help='List of layer sizes (default: [784, 120, 4])')
parser.add_argument(
    '--n_iter_free',
    type=int,
    default=10,
    help='Times to iterate for the annealer on a single data point for free phase (default=10)')
parser.add_argument(
    '--n_iter_nudge',
    type=int,
    default=10,
    help='Times to iterate for the annealer on a single data point for nudge phase (default=10)')
parser.add_argument(
    '--frac_anneal_nudge',
    type=float,
    default=0.25,
    help='fraction of system non-annealed (default=0.25)')
parser.add_argument(
    '--N_data',
    type=int,
    default=1000,
    help='Number of data points for training (default=1000)')
parser.add_argument(
    '--N_data_test',
    type=int,
    default=100,
    help='Number of data points for testing (default=100)')
parser.add_argument(
    '--beta',
    type=float,
    default=5,
    help='Beta - hyperparameter of EP (default=5)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Size of mini-batches we use (for training only)')
parser.add_argument(
    '--lrW0',
    type=float,
    default=0.01,
    help='Learning rate for weights - input-hidden  (default=0.01)')
parser.add_argument(
    '--lrW1',
    type=float,
    default=0.01,
    help='Learning rate for weights - hidden-output (default=0.01)')
parser.add_argument(
    '--lrB0',
    type=float,
    default=0.001,
    help='Learning rate for biases - hidden (default=0.001)')
parser.add_argument(
    '--lrB1',
    type=float,
    default=0.001,
    help='Learning rate for biases - output (default=0.001)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    help='Number of epochs (default=10)')
parser.add_argument(
    '--load_model',
    type=int,
    default=0,
    help='If we load the parameters from a previously trained model (default=0, else = 1)')
parser.add_argument(
    '--digit',
    type=int,
    default=-1,
    help='If >= 0, only train the classifier for this specific digit (default=-1 for all digits)')
parser.add_argument(
    '--gain_weight0',
    type=float,
    default=0.5,
    help='Gain for initialization of the weights - input-hidden (default=0.5)')
parser.add_argument(
    '--gain_weight1',
    type=float,
    default=0.25,
    help='Gain for initialization of the weights  - hidden-output (default=0.25)')
parser.add_argument(
    '--bias_lim',
    type=float,
    default=4.0,
    help='Max limit for the amplitude of the local biases (default=4)')
parser.add_argument(
    '--chain_strength',
    type=float,
    default=1.0,
    help='Value of the coupling in the chain of identical spins (default=1)')
parser.add_argument(
    '--auto_scale',
    type=int,
    default=0,
    help='Set auto_scale or not for the problems (default=0)')
args = parser.parse_args()

if args.auto_scale == 0:
    args.auto_scale = False
else:
    args.auto_scale = True

with torch.no_grad():
    ## SAMPLERs
    simu_sampler = neal.SimulatedAnnealingSampler()
    exact_sampler = dimod.ExactSolver()

    if args.simulated == 1:
        qpu_sampler = SimulatedAnnealingSampler()
    else:
        qpu_sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}, auto_scale = args.auto_scale))

    ## Generate DATA
    if args.dataset == "digits":
        # Add dataset_dir to args
        args.dataset_dir = DATASET_DIR
        train_loader, test_loader = generate_digits(args)
    elif args.dataset == "mnist":
        # Add dataset_dir to args
        args.dataset_dir = DATASET_DIR
        train_loader, test_loader, dataset = generate_mnist(args)

    # Create a custom target transformer for binary classification
    class BinaryTargetTransform:
        def __init__(self, digit, mode='ising'):
            self.digit = digit
            self.mode = mode
            
        def __call__(self, target):
            # If the target equals our digit, return 1 (or 1s), else -1 (or 0s)
            if target == self.digit:
                return torch.ones(4)
            else:
                return torch.ones(4) * (-1 if self.mode == 'ising' else 0)
    
    # Get full dataset for binary targets
    print("Loading original MNIST dataset...")
    mnist_train = torchvision.datasets.MNIST(root=DATASET_DIR, train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())
    mnist_test = torchvision.datasets.MNIST(root=DATASET_DIR, train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())
    
    # Extract the data and targets we need
    all_train_data = []
    original_train_targets = []
    for i in range(min(args.N_data, len(mnist_train))):
        all_train_data.append(mnist_train[i][0].reshape(-1))
        original_train_targets.append(mnist_train[i][1])
    
    all_test_data = []
    original_test_targets = []
    for i in range(min(args.N_data_test, len(mnist_test))):
        all_test_data.append(mnist_test[i][0].reshape(-1))
        original_test_targets.append(mnist_test[i][1])
    
    all_train_data = torch.stack(all_train_data)
    original_train_targets = torch.tensor(original_train_targets)
    
    all_test_data = torch.stack(all_test_data)
    original_test_targets = torch.tensor(original_test_targets)
    
    print(f"Train data shape: {all_train_data.shape}, Train targets shape: {original_train_targets.shape}")
    print(f"Test data shape: {all_test_data.shape}, Test targets shape: {original_test_targets.shape}")

    # Determine which digits to process
    digits_to_process = [args.digit] if args.digit >= 0 else list(range(10))
    
    for digit in digits_to_process:
        print(f"\n==== Training Binary Classifier for Digit {digit} ====")
        
        # Create a copy of args for this digit
        digit_args = copy.deepcopy(args)
        # Set output layer to 4 neurons (expanded from 1)
        digit_args.layersList = [args.layersList[0], args.layersList[1], 4]  
        
        # Create binary targets for this digit
        binary_train_targets = torch.zeros((len(original_train_targets), 4))
        if args.mode == "ising":
            binary_train_targets.fill_(-1)
        
        # Set to 1 for matching targets
        for i, target in enumerate(original_train_targets):
            if target == digit:
                binary_train_targets[i].fill_(1)
        
        binary_test_targets = torch.zeros((len(original_test_targets), 4))
        if args.mode == "ising":
            binary_test_targets.fill_(-1)
            
        for i, target in enumerate(original_test_targets):
            if target == digit:
                binary_test_targets[i].fill_(1)
        
        # Print summary of binary targets
        pos_train = (binary_train_targets[:, 0] > 0).sum().item()
        neg_train = len(binary_train_targets) - pos_train
        pos_test = (binary_test_targets[:, 0] > 0).sum().item()
        neg_test = len(binary_test_targets) - pos_test
        
        print(f"Binary train targets: {pos_train} positive, {neg_train} negative")
        print(f"Binary test targets: {pos_test} positive, {neg_test} negative")
        
        binary_train_dataset = torch.utils.data.TensorDataset(all_train_data, binary_train_targets)
        binary_test_dataset = torch.utils.data.TensorDataset(all_test_data, binary_test_targets)
        
        binary_train_loader = torch.utils.data.DataLoader(
            binary_train_dataset, batch_size=args.batch_size, shuffle=True)
        binary_test_loader = torch.utils.data.DataLoader(
            binary_test_dataset, batch_size=1, shuffle=False)
        
        # Create results directory for this digit classifier
        digit_results_dir = os.path.join(RESULTS_DIR, f'digit_{digit}')
        os.makedirs(digit_results_dir, exist_ok=True)
        
        # Save plotFunction.py if it exists
        try:
            shutil.copy('MLP/Binary-Classifiers/plotFunction.py', digit_results_dir)
        except FileNotFoundError:
            print("Warning: plotFunction.py not found")
        
        # Set custom path for this model
        digit_args.results_dir = digit_results_dir
        
        ## Files saving: create a folder for this digit
        BASE_PATH = createPath(digit_args, digit=digit, simu='')
        dataframe = initDataframe(BASE_PATH)
        print(BASE_PATH)
        
        ## Create the network for this digit
        if args.load_model == 0:
            saveHyperparameters(BASE_PATH, digit_args, digit=digit, simu='binary-classifier')
            net = Network(digit_args)
        else:
            net = load_model_numpy(BASE_PATH)
        
        ## Monitor loss and prediction error
        qpu_loss_tab, qpu_falsePred_tab = [], []
        qpu_loss_test_tab, qpu_falsePred_test_tab = [], []
        
        for epoch in tqdm(range(args.epochs), desc=f"Digit {digit} Epochs"):
            # Train the network
            _, _, qpu_loss, qpu_falsePred = train_binary(net, digit_args, binary_train_loader, 
                                                      simu_sampler, exact_sampler, qpu_sampler)
            qpu_loss_tab.append(qpu_loss)
            qpu_falsePred_tab.append(qpu_falsePred)
            
            # Test the network
            _, _, qpu_loss, qpu_falsePred = test_binary(net, digit_args, binary_test_loader, 
                                                     simu_sampler, exact_sampler, qpu_sampler)
            qpu_loss_test_tab.append(qpu_loss)
            qpu_falsePred_test_tab.append(qpu_falsePred)
            
            # Calculate error rates for this epoch
            train_error = qpu_falsePred_tab[-1]/len(binary_train_loader.dataset)*100
            test_error = qpu_falsePred_test_tab[-1]/len(binary_test_loader.dataset)*100
            
            # Print epoch stats
            print(f"Digit {digit}, Epoch {epoch+1}/{args.epochs}: " 
                  f"Training Error: {train_error:.2f}%, "
                  f"Test Error: {test_error:.2f}%, "
                  f"Train Loss: {qpu_loss_tab[-1]:.2f}, "
                  f"Test Loss: {qpu_loss_test_tab[-1]:.2f}")
            
            # Store error and loss at each epoch
            dataframe = updateDataframe(BASE_PATH, dataframe, 
                                     0, 0,  # Placeholder for exact values
                                     train_error, test_error, 
                                     [0], [0],  # Placeholder for exact values
                                     qpu_loss_tab, qpu_loss_test_tab)
            
            save_model_numpy(BASE_PATH, net)
        
        print(f"Training completed for digit {digit}.")
        print(f"Final training error: {qpu_falsePred_tab[-1]/len(binary_train_loader.dataset)*100:.2f}%")
        print(f"Final test error: {qpu_falsePred_test_tab[-1]/len(binary_test_loader.dataset)*100:.2f}%")