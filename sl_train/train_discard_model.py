import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.models import DiscardModel
from dataset.data import TenhouDataset, TenhouIterableDataset, process_data, collate_fn_discard

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import tqdm


@torch.no_grad()
def model_test(model, dataset: TenhouDataset):
    acc = 0
    total = 0
    length = len(dataset)
    while len(dataset) > 0:
        data = dataset()
        if len(data) == 0:
            break
        features, labels = process_data(data, label_trans=lambda x: x // 4)
        features, labels = features.to(device), labels.to(device)
        output = model(features).softmax(1)
        available = features[:, :4].sum(1) != 0
        pred = (output * available).argmax(1)
        correct = (pred == labels).sum()
        acc += correct
        total += len(labels)
        print(f"Testing {length - len(dataset)} / {length} acc: {correct.item() / len(labels):.3f}".center(50, '-'), end='\r')
    dataset.reset()
    return acc / total

if __name__ == '__main__':
    mode = 'discard'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', '-n', default=50, type=int)
    parser.add_argument('--epochs', '-e', default=10, type=int)
    parser.add_argument('--batch_size', '-b', default=512, type=int)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default=f'output/{mode}-model/checkpoints')
    parser.add_argument('--log-dir', default=f'logs/{mode}')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)
    train_set = TenhouDataset(data_dir=args.data_dir, batch_size=128, mode=mode, target_length=2)
    test_set = TenhouDataset(data_dir=args.data_dir, batch_size=128, mode=mode, target_length=2)
    length = len(train_set)
    len_train = int(0.8 * length)
    train_set.data_files, test_set.data_files = train_set.data_files[:len_train], train_set.data_files[len_train:]

    num_layers = args.num_layers
    in_channels = 291
    model = DiscardModel(num_layers=num_layers, in_channels=in_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    optim = Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
    loss_fcn = CrossEntropyLoss()
    epochs = args.epochs

    os.makedirs(args.output_dir, exist_ok=True)
    max_acc = 0
    global_step = 0
    dataset = TenhouIterableDataset(
        data_dir=args.data_dir,
        exclude_files=set(test_set.data_files),
        mode='discard',
        target_length=2,
        shuffle=True
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn_discard,
        pin_memory=True,
        prefetch_factor=10
    )
    for epoch in range(epochs):
        for features, labels in tqdm.tqdm(train_loader):
            features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            output = model(features)
            loss = loss_fcn(output, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            global_step += 1
            writer.add_scalar('Loss/train', loss.item(), global_step)

        train_set.reset()

        torch.save(
            {"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels},
            os.path.join(args.output_dir, f'epoch_{epoch + 1}.pt')
        )
        model.eval()
        acc = model_test(model, test_set)
        if acc > max_acc:
            max_acc = acc
            torch.save(
                {"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels},
                os.path.join(args.output_dir, 'best.pt')
            )
        model.train()

        writer.add_scalar('Metrics/accuracy', acc, epoch + 1)
        writer.add_scalar('LR', optim.param_groups[0]['lr'], epoch + 1)
        scheduler.step(acc)

    writer.close()
