import argparse
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class ExperimentParams:
    """Dataclass for storing experiment hyperparameters and settings."""
    dataset: str
    data_dir: str
    num_workers: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    
    model: str
    input_size: int
    hidden_sizes: List[int]
    num_classes: int
    dropout: float
    vgg_depth: str
    resnet_layers: List[int]
    
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    
    seed: int
    device: str
    save_path: str
    log_interval: int
    mode: str
    transfer_opt: str
    label_smoothing: float

def get_params() -> ExperimentParams:
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode",         choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",      choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",        choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="mlp")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--vgg_depth",    choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer")
    
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Amount of label smoothing (e.g., 0.1)")
    parser.add_argument("--transfer_opt", choices=["none", "1", "2", "distill", "mobilenet_distill"], default="none", 
                        help="Transfer learning option (1: Resize/Freeze, 2: Modify/Finetune, distill: Knowledge Distill, mobilenet_distill: Modified Distill)")

    args = parser.parse_args()

    # Dataset
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std  = (0.1307,), (0.3081,)
    else:                         # cifar10
        input_size = 3072         # 3 × 32 × 32
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    return ExperimentParams(
        dataset=args.dataset,
        data_dir="./data",
        num_workers=2,
        mean=mean,
        std=std,
        
        model=args.model,
        input_size=input_size,
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        dropout=0.3,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,
        
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-4,
        
        seed=42,
        device=args.device,
        save_path="best_model.pth",
        log_interval=100,
        mode=args.mode,
        transfer_opt=args.transfer_opt,
        label_smoothing=args.label_smoothing
    )