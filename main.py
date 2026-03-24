"""
Main execution script for building architectures, setting device configurations, 
and managing the training, testing, and transfer learning pipelines.
"""

import random
import ssl
import numpy as np
import torch

from parameters import get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from train import run_training
from test  import run_test
from utils import print_macs_flops
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision import models
import torch.nn as nn
from utils import save_model_graph
from train import run_training, run_distillation, run_modified_distillation

def set_seed(seed):
    """
    Fixes the random seed across standard library, NumPy, and PyTorch 
    to ensure reproducible deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(params):
    """
    Constructs and returns the requested neural network model. Handles modifications 
    required for Transfer Learning Option 1 (Frozen) and Option 2 (Finetuned).
    """
    model_name = params.model
    dataset    = params.dataset
    nc         = params.num_classes

    if params.transfer_opt == "1":
        # Approach 1
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, nc)
        return model

    elif params.transfer_opt == "2":
        # Approach 2
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, nc)
        return model
    
    if model_name == "mlp":
        return MLP(
            input_size   = params.input_size,
            hidden_sizes = params.hidden_sizes,
            num_classes  = nc,
            dropout      = params.dropout,
        )

    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        else:
            return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=params.vgg_depth, num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, params.resnet_layers, num_classes=nc)
        
    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10 with mobilenet.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")

def main():
    """
    Primary entry point: Initializes parameters, configures device, generates model graphs, 
    and triggers the appropriate execution mode (Train, Distill, Test).
    """
    params = get_params()

    set_seed(params.seed)
    print(f"Seed set to: {params.seed}")
    print(f"Dataset: {params.dataset}  |  Model: {params.model}")

    device = torch.device(
        params.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    input_shape = (3, 224, 224) if params.transfer_opt == "1" else (3, 32, 32)
    save_model_graph(model, input_size=input_shape)
    print_macs_flops(model, input_size=input_shape)
    print(model)

    if params.mode in ("train", "both"):
        if params.transfer_opt == "distill":
            run_distillation(model, params, device)
        elif params.transfer_opt == "mobilenet_distill":
            run_modified_distillation(model, params, device)
        else:
            run_training(model, params, device)

    if params.mode in ("test", "both"):
        run_test(model, params, device)

if __name__ == "__main__":
    main()