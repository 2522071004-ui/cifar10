import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# ==========================================
# 1. Configuration
# ==========================================
BATCH_SIZE = 128
EPOCHS = 50 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"Running on Device: {DEVICE}")

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# ==========================================
# 2. Data Preprocessing
# ==========================================
# CIFAR-10 mean and std
stats = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# 3. Model Definition: ResNet18 for CIFAR
# ==========================================
def get_resnet18_cifar():
    """
    Modified ResNet18 for CIFAR-10:
    1. conv1 changed to 3x3, stride=1 (since images are 32x32)
    2. Removed maxpool
    3. fc layer output changed to 10 classes
    """
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

# ==========================================
# 4. Training Engine
# ==========================================
def train_model(opt_name, opt_conf):
    model = get_resnet18_cifar()
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Optimizer
    if opt_name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=opt_conf['lr'], momentum=0.9, weight_decay=opt_conf['wd'])
    elif opt_name == 'SGD+Nesterov':
        optimizer = optim.SGD(model.parameters(), lr=opt_conf['lr'], momentum=0.9, nesterov=True, weight_decay=opt_conf['wd'])
    elif opt_name == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt_conf['lr'], weight_decay=opt_conf['wd'])
    elif opt_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=opt_conf['lr'], alpha=0.99, weight_decay=opt_conf['wd'])
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt_conf['lr'], betas=(0.9, 0.999), weight_decay=opt_conf['wd'])
    elif opt_name == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr=opt_conf['lr'], betas=(0.9, 0.999), weight_decay=opt_conf['wd'])
    elif opt_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=opt_conf['lr'], betas=(0.9, 0.999), weight_decay=opt_conf['wd'])
    
    # Scheduler: CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_losses, test_accs = [], []
    
    print(f"\n>>> Training {opt_name} | LR: {opt_conf['lr']} | WD: {opt_conf['wd']}")
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Update Learning Rate
        scheduler.step()
        
        # Record Loss
        train_losses.append(running_loss / len(trainloader))
        
        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        test_accs.append(acc)
        
        # Print progress (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            print(f"    Epoch {epoch+1}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | Acc: {acc:.2f}%")
            
    total_time = time.time() - start_time
    print(f"<<< Finished {opt_name} | Best Acc: {max(test_accs):.2f}% | Time: {total_time:.0f}s")
    
    return train_losses, test_accs

# ==========================================
# 5. Experiment Parameter Dictionary
# ==========================================
optimizers_config = {
    'SGD+Momentum': {'lr': 0.1,   'wd': 5e-4}, 
    'SGD+Nesterov': {'lr': 0.1,   'wd': 5e-4}, 
    'AdaGrad':      {'lr': 0.01,  'wd': 0},    # Adagrad has built-in decay, WD usually set to 0
    'RMSProp':      {'lr': 0.001, 'wd': 1e-4}, 
    'Adam':         {'lr': 0.001, 'wd': 1e-4}, 
    'Nadam':        {'lr': 0.002, 'wd': 1e-4}, 
    'AdamW':        {'lr': 0.001, 'wd': 1e-2}, # Note: WD must be larger for AdamW (0.01)
}

results = {}

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == '__main__':
    for name, conf in optimizers_config.items():
        losses, accs = train_model(name, conf)
        results[name] = {'loss': losses, 'acc': accs}

    # Plotting
    plt.figure(figsize=(15, 6))

    # Loss Curve
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['loss'], label=name, alpha=0.8)
    plt.title('Training Loss (ResNet18 / CIFAR-10)')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['acc'], label=f"{name} (Best: {max(data['acc']):.1f}%)", alpha=0.8)
    plt.title('Test Accuracy (ResNet18 / CIFAR-10)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('optimizer_benchmark_result.png')
    plt.show()
    print("\nExperiment finished! Result saved to optimizer_benchmark_result.png")