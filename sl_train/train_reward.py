import argparse
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.data import process_reward_data, TenhouDataset
from model.models import RewardPredictor


@torch.no_grad()
def model_test(model, dataset: TenhouDataset):
    total_error = 0
    total = 0
    length = len(dataset)
    while len(dataset) > 0:
        data = dataset()
        if len(data) == 0:
            break
        features, labels = process_reward_data(data)
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        error = (output - labels).pow(2).sum()
        total_error += error
        total += len(labels)
        print(f"Testing {length - len(dataset)} / {length} Error: {error:.3f}".center(50, '-'), end='\r')
    dataset.reset()
    return total_error / total


if __name__ == '__main__':
    mode = 'reward'
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dims', '-hd', default=50, type=int)
    parser.add_argument('--num_layers', '-n', default=2, type=int)
    parser.add_argument('--epochs', '-e', default=10, type=int)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default=f'output/{mode}-model/checkpoints')
    parser.add_argument('--log-dir', default=f'logs/{mode}')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)
    train_set = TenhouDataset(data_dir=args.data_dir, batch_size=128, mode=mode, target_length=4)
    test_set = TenhouDataset(data_dir=args.data_dir, batch_size=128, mode=mode, target_length=4)
    length = len(train_set)
    len_train = int(0.8 * length)
    train_set.data_files, test_set.data_files = train_set.data_files[:len_train], train_set.data_files[len_train:]

    hidden_dims = args.hidden_dims
    epochs = args.epochs
    num_layers = args.num_layers
    model = RewardPredictor(74, hidden_dims, num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    optim = Adam(model.parameters())
    loss_fcn = MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=1)

    os.makedirs(args.output_dir, exist_ok=True)
    min_mse = torch.inf
    global_step = 0
    for epoch in range(epochs):
        while len(train_set) > 0:
            data = train_set()
            if len(data) == 0:
                break
            features, labels = process_reward_data(data)
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            loss = loss_fcn(output, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            global_step += 1
            writer.add_scalar('Loss/train', loss.item(), global_step)

        train_set.reset()

        torch.save(
            {"state_dict": model.state_dict(), "num_layers": num_layers, "hidden_dims": hidden_dims},
            os.path.join(args.output_dir, f'epoch_{epoch + 1}.pt')
        )
        model.eval()
        mse = model_test(model, test_set)
        if mse < min_mse:
            min_mse = mse
            torch.save(
                {"state_dict": model.state_dict(), "num_layers": num_layers, "hidden_dims": hidden_dims},
                os.path.join(args.output_dir, 'best.pt')
            )
        model.train()

        writer.add_scalar('Metrics/mse', mse, epoch + 1)
        writer.add_scalar('LR', optim.param_groups[0]['lr'], epoch + 1)
        scheduler.step(mse)

    writer.close()
