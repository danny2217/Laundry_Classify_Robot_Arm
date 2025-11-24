
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import ImageFile
from resnet_model.resnet import get_resnet18
from utils import progress_bar
import multiprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser(description='Optimized Fast Training: ResNet18 TextileNet')
    parser.add_argument('--partition',       default='fiber',    choices=['fiber','fabric'])
    parser.add_argument('--data_parent_dir', default='data')
    parser.add_argument('--lr',              default=5e-4,       type=float)
    parser.add_argument('--batch_size',      default=256,        type=int)
    parser.add_argument('--num_workers',     default=None,       type=int, help='If None, uses os.cpu_count()')
    parser.add_argument('--num_classes',     default=10,         type=int)
    parser.add_argument('--resume',          action='store_true')
    parser.add_argument('--epochs',          default=70,         type=int)
    parser.add_argument('--patience',        default=3,          type=int)
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'   if torch.backends.mps.is_available() else
        'cpu'
    )
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    pin_memory  = (device.type == 'cuda')

    train_dir = os.path.join(args.data_parent_dir, args.partition, 'train')
    test_dir  = os.path.join(args.data_parent_dir, args.partition, 'test')

    common_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.ImageFolder(train_dir, common_tf),
        batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True
    )
    test_loader = DataLoader(
        datasets.ImageFolder(test_dir, common_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True
    )

    model = get_resnet18(num_classes=args.num_classes)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Only compile on CUDA to avoid MPS Inductor issues
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        model = torch.compile(model)

    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt_path = f'./{args.partition}_results/checkpoint_res/ckpt.pth'
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get('model', ckpt)
            model.load_state_dict(state)
            best_acc    = ckpt.get('acc', 0.0)
            start_epoch = ckpt.get('epoch', 0)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'Epoch {epoch+1}/{start_epoch+args.epochs} on {device.type} (workers={num_workers})')
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)
            total   += targets.size(0)
            correct += preds.eq(targets).sum().item()
            progress_bar(i, len(train_loader),
                f'Loss: {running_loss/(i+1):.3f} | Acc: {100.*correct/total:.2f}%')

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                total   += targets.size(0)
                correct += preds.eq(targets).sum().item()
                progress_bar(i, len(test_loader),
                    f'Val Acc: {100.*correct/total:.2f}%')

        val_acc = 100.*correct/total
        if val_acc > best_acc:
            save_dir = f'./{args.partition}_results/checkpoint_res'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {'model': model.state_dict(), 'acc': val_acc, 'epoch': epoch},
                os.path.join(save_dir, 'ckpt.pth')
            )
            best_acc = val_acc

        scheduler.step()
        if epoch - start_epoch > args.patience and val_acc <= best_acc:
            print('Early stopping')
            break

    print('Training complete')

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
