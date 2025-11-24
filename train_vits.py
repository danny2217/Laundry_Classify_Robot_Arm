
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
from timm import create_model
from utils import progress_bar
import os
from torch.amp import GradScaler, autocast
from multiprocessing import freeze_support

freeze_support()

parser = argparse.ArgumentParser(description='PyTorch TextileNet ViT Training (AMP + EarlyStopping)')
parser.add_argument('--partition', default='fiber', type=str, help='type of data: fiber or fabric')
parser.add_argument('--data_parent_dir', default='../data', type=str, help='parent directory of data')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')
parser.add_argument('--patience', default=3, type=int, help='early stopping patience')
args = parser.parse_args()

DATA = args.partition
train_data = os.path.join(args.data_parent_dir, DATA, 'train')
test_data  = os.path.join(args.data_parent_dir, DATA, 'test')

# Remove empty class directories
valid_exts = ('.jpg','.jpeg','.png','.ppm','.bmp','.pgm','.tif','.tiff','.webp')
def remove_empty_dirs(root):
    for cls in os.listdir(root):
        path = os.path.join(root, cls)
        if os.path.isdir(path) and not any(f.lower().endswith(valid_exts) for f in os.listdir(path)):
            os.rmdir(path)
remove_empty_dirs(train_data)
remove_empty_dirs(test_data)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"   if torch.backends.mps.is_available() else "cpu")
use_amp = device.type == "cuda"
scaler = GradScaler(device_type="cuda") if use_amp else None

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

pin_memory = True if device.type == "cuda" else False
train_set = datasets.ImageFolder(root=train_data, transform=data_transform)
test_set  = datasets.ImageFolder(root=test_data,  transform=test_transform)
train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=pin_memory,
    prefetch_factor=2, persistent_workers=True
)
test_loader  = DataLoader(
    test_set,  batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=pin_memory,
    prefetch_factor=2, persistent_workers=True
)

model = create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

start_epoch = 0
best_acc = 0
if args.resume:
    ckpt_dir = f"./{DATA}_results/checkpoint_vit"
    checkpoint = torch.load(os.path.join(ckpt_dir, "ckpt.pth"), map_location=device)
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint.get('acc', 0)
    start_epoch = checkpoint.get('epoch', 0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

def train_one_epoch():
    model.train()
    running_loss = correct = total = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        progress_bar(i, len(train_loader), f"Train Loss: {running_loss/(i+1):.3f} | Acc: {100.*correct/total:.2f}%")

def validate():
    global best_acc
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            progress_bar(i, len(test_loader), f"Val Acc so far: {100.*correct/total:.2f}%")
    acc = 100.*correct/total
    if acc > best_acc:
        os.makedirs(f"./{DATA}_results/checkpoint_vit", exist_ok=True)
        torch.save({'model': model.state_dict(), 'acc': acc, 'epoch': epoch},
                   f"./{DATA}_results/checkpoint_vit/ckpt.pth")
        best_acc = acc
    return acc

if __name__ == "__main__":
    no_improve = 0
    best_val = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"\nEpoch {epoch+1}/{start_epoch+args.epochs}")
        train_one_epoch()
        val_acc = validate()
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    print("Training complete.")
