import os
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.models import FuroModel
from dataset.data import TenhouDataset, process_data


@torch.no_grad()
def model_test(model, dataset: TenhouDataset, epoch, writer: SummaryWriter):
    length = len(dataset)
    y_true = []
    y_score = []
    while len(dataset) > 0:
        data = dataset()
        if len(data) == 0:
            break
        features, labels = process_data(data, label_trans=lambda x: x.float())
        features, labels = features.to(device), labels.to(device)
        output = model(features).sigmoid().flatten()
        y_true.extend(labels.tolist())
        y_score.extend(output.tolist())
        print(f"Testing {length - len(dataset)} / {length}".center(50, '-'), end='\r')
    dataset.reset()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    writer.add_figure('ROC', fig, global_step=epoch)
    plt.close(fig)

    y_pred = list(map(lambda x: int(x > threshold), y_score))
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=[1], average='binary')
    return recall, precision, acc, f_score, threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default='chi', type=str, choices=['chi', 'pon', 'kan'])
    parser.add_argument('--num_layers', '-n', default=20, type=int)
    parser.add_argument('--epochs', '-e', default=10, type=int)
    parser.add_argument('--pos_weight', '-w', default=None, type=int)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--log-dir', default=None)
    args = parser.parse_args()
    mode = args.mode
    output_dir = args.output_dir or f'output/{mode}-model/checkpoints'
    log_dir = args.log_dir or f'logs/{mode}'

    writer = SummaryWriter(log_dir=log_dir)
    train_set = TenhouDataset(data_dir=args.data_dir, batch_size=128, mode=mode, target_length=2)
    test_set = TenhouDataset(data_dir=args.data_dir, batch_size=128, mode=mode, target_length=2)
    length = len(train_set)
    len_train = int(0.8 * length)
    train_set.data_files, test_set.data_files = train_set.data_files[:len_train], train_set.data_files[len_train:]

    num_layers = args.num_layers
    in_channels = 291 + 22
    model = FuroModel(num_layers=num_layers, in_channels=in_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    optim = Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
    if args.pos_weight is not None:
        loss_fcn = BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    else:
        loss_fcn = BCEWithLogitsLoss()
    epochs = args.epochs

    os.makedirs(output_dir, exist_ok=True)
    max_f1 = 0
    global_step = 0
    for epoch in range(epochs):
        while len(train_set) > 0:
            data = train_set()
            if len(data) == 0:
                break
            features, labels = process_data(data, label_trans=lambda x: x.float())
            features, labels = features.to(device), labels.to(device)
            output = model(features).flatten()
            loss = loss_fcn(output, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            global_step += 1
            writer.add_scalar('Loss/train', loss.item(), global_step)

        train_set.reset()

        model.eval()
        recall, precision, acc, f_score, threshold = model_test(model, test_set, epoch + 1, writer)
        torch.save({
            "state_dict": model.state_dict(),
            "num_layers": num_layers,
            "in_channels": in_channels,
            "threshold": threshold
        }, os.path.join(output_dir, f'epoch_{epoch + 1}.pt'))
        if f_score > max_f1:
            max_f1 = f_score
            torch.save({
                "state_dict": model.state_dict(),
                "num_layers": num_layers,
                "in_channels": in_channels,
                "threshold": threshold
            }, os.path.join(output_dir, 'best.pt'))
        model.train()

        writer.add_scalar('Metrics/f1', f_score, epoch + 1)
        writer.add_scalar('Metrics/recall', recall, epoch + 1)
        writer.add_scalar('Metrics/precision', precision, epoch + 1)
        writer.add_scalar('Metrics/accuracy', acc, epoch + 1)
        writer.add_scalar('Metrics/threshold', threshold, epoch + 1)
        writer.add_scalar('LR', optim.param_groups[0]['lr'], epoch + 1)
        scheduler.step(f_score)

    writer.close()
