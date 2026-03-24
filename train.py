"""
Training module for handling standard optimization, standard knowledge distillation,
and modified probability distillation loops.
"""

import copy
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from utils import plot_training_curves

def get_transforms(params, train=True):
    """
    Generates the data transformation pipelines for training and validation datasets.
    Includes spatial resizing for Transfer Learning Option 1.
    """
    mean, std = params.mean, params.std
    
    transform_list = []
    
    if params.transfer_opt == "1":
        transform_list.append(transforms.Resize((224, 224)))
        
    if params.dataset == "mnist":
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transform_list)
    else:  # cifar10
        if train and params.transfer_opt != "1":
            # Augmentations (skip for Option 1)
            transform_list.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()
            ])
        transform_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(transform_list)

def run_distillation(student_model, params, device):
    """
    Trains a student model using a pre-trained teacher model via standard Knowledge Distillation.
    Utilizes KL-Divergence to match the student's soft targets with the teacher's soft targets.
    """
    train_loader, val_loader = get_loaders(params)
    
    # 1. Load the Teacher Model 
    from main import build_model
    teacher_params = copy.deepcopy(params)
    teacher_params.model = "resnet" 
    teacher_model = build_model(teacher_params).to(device)
    
    try:
        teacher_model.load_state_dict(torch.load("resnet_teacher_smooth.pth", map_location=device))
        print("Successfully loaded resnet_teacher_smooth.pth")
    except Exception as e:
        print(f"Error")
        return
        
    teacher_model.eval()
    
    # 2. Setup Student Training
    optimizer = torch.optim.Adam(student_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Distillation Hyperparams
    T = 2.0     # Temperature
    alpha = 0.5 # Weight
    
    best_acc = 0.0
    best_weights = None
    
    # history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\n=== Starting Knowledge Distillation Training ===")
    
    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")
        student_model.train()
        total_loss, correct, n = 0.0, 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            student_logits = student_model(imgs)
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            
            # 1. Standard Loss (Student predictions - True hard labels)
            standard_loss = F.cross_entropy(student_logits, labels)
            
            # 2. Distillation Loss (Student soft predictions - Teacher soft predictions)
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)
            
            # 3. Combine losses
            loss = (alpha * standard_loss) + ((1. - alpha) * distillation_loss)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += student_logits.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

            if (batch_idx + 1) % params.log_interval == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}] loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

        tr_loss, tr_acc = total_loss / n, correct / n
        
        # Same - no teacher needed for validation
        val_loss, val_acc = validate(student_model, val_loader, nn.CrossEntropyLoss(), device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(student_model.state_dict())
            torch.save(best_weights, params.save_path)
            print(f"  Saved best student model (val_acc={best_acc:.4f})")

    student_model.load_state_dict(best_weights)
    print(f"\nDistillation done. Best val accuracy: {best_acc:.4f}")
    
    history_path = params.save_path.replace('.pth', '_history_distill.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    try:
        plot_path = params.save_path.replace('.pth', '_curves_distill.png')
        plot_training_curves(history['train_loss'], history['val_loss'], 
                             history['train_acc'], history['val_acc'], save_path=plot_path)
    except Exception as e:
        print(f"  Could not plot training curves: {e}")

def get_loaders(params):
    """
    Initializes and returns PyTorch DataLoaders for the specified training and validation datasets.
    """
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params.dataset == "mnist":
        train_ds = datasets.MNIST(params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params.data_dir, train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params.batch_size,
                              shuffle=True,  num_workers=params.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=params.batch_size,
                              shuffle=False, num_workers=params.num_workers)
    
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, criterion, device, log_interval):
    """
    Executes a single training epoch, performing forward passes, loss calculation, and backpropagation.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n

def validate(model, loader, criterion, device):
    """
    Evaluates the model's performance on the validation dataset without updating gradients.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n

def run_training(model, params, device):
    """
    Executes the standard training loop including model optimization, learning rate scheduling, 
    and model checkpointing based on validation accuracy.
    """
    train_loader, val_loader = get_loaders(params)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)
    
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(params_to_update,
                                 lr=params.learning_rate,
                                 weight_decay=params.weight_decay)        
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, params.log_interval)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    history_path = params.save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"  Saved training history to {history_path}")
    
    try:
        plot_path = params.save_path.replace('.pth', '_curves.png')
        plot_training_curves(history['train_loss'], history['val_loss'], 
                             history['train_acc'], history['val_acc'], save_path=plot_path)
    except Exception as e:
        print(f"  Could not plot training curves: {e}")

def run_modified_distillation(student_model, params, device):
    """
    Trains a MobileNet student utilizing a custom distillation logic where probabilities
    assigned to background classes by the teacher are dynamically flattened and distributed equally.
    """
    train_loader, val_loader = get_loaders(params)
    
    # 1. Load the Teacher Model
    from main import build_model
    teacher_params = copy.deepcopy(params)
    teacher_params.model = "resnet"
    teacher_model = build_model(teacher_params).to(device)
    
    try:
        teacher_model.load_state_dict(torch.load("resnet_teacher_smooth.pth", map_location=device))
        print("Successfully loaded resnet_teacher_smooth.pth")
    except Exception as e:
        print(f"Error loading teacher: {e}. Did you rename your 40-epoch model?")
        return
        
    teacher_model.eval()
    
    # 2. Setup Student Training (MobileNet)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    T = 2.0
    alpha = 0.5
    best_acc = 0.0
    best_weights = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\n=== Starting Modified Knowledge Distillation (MobileNet) ===")
    
    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")
        student_model.train()
        total_loss, correct, n = 0.0, 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            student_logits = student_model(imgs)
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            
            standard_loss = F.cross_entropy(student_logits, labels)
            
            # 1. Get standard probabilities from teacher
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            batch_size = labels.size(0)
            num_classes = teacher_probs.size(1)
            
            # 2. Extracting the probability of the TRUE class
            true_probs = teacher_probs[torch.arange(batch_size), labels]
            
            distributed_probs = (1.0 - true_probs) / (num_classes - 1)
            
            modified_probs = distributed_probs.unsqueeze(1).repeat(1, num_classes)
            modified_probs[torch.arange(batch_size), labels] = true_probs
            
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                modified_probs,
                reduction='batchmean'
            ) * (T * T)
            
            loss = (alpha * standard_loss) + ((1. - alpha) * distillation_loss)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += student_logits.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

            if (batch_idx + 1) % params.log_interval == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}] loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

        tr_loss, tr_acc = total_loss / n, correct / n
        val_loss, val_acc = validate(student_model, val_loader, nn.CrossEntropyLoss(), device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(student_model.state_dict())
            torch.save(best_weights, params.save_path)
            print(f"  Saved best student model (val_acc={best_acc:.4f})")

    student_model.load_state_dict(best_weights)
    print(f"\nModified Distillation done. Best val accuracy: {best_acc:.4f}")
    
    history_path = params.save_path.replace('.pth', '_history_mod_distill.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    try:
        plot_path = params.save_path.replace('.pth', '_curves_mod_distill.png')
        plot_training_curves(history['train_loss'], history['val_loss'], 
                             history['train_acc'], history['val_acc'], save_path=plot_path)
    except Exception as e:
        pass