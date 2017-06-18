import torch
from datetime import datetime
import os
import csv
import shutil
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, log_folder, is_best):
    filename = os.path.join(log_folder,
                            "model_current_%03d.pth.tar" % state['epoch'])
    torch.save(state, filename)
    if is_best:
        new_name = os.path.join(log_folder,
                                'model_best_%03d.pth.tar' % state['epoch'])
        shutil.copyfile(filename, new_name)


def adjust_learning_rate(optimizer, epoch, initial_lr, regime=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if regime is not None and epoch in regime.keys():
        lr = regime[epoch]
        print "Adust LR to %f" % lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_save(logs, log_folder):
    with open(os.path.join(log_folder, 'log.csv'), 'w') as f:
        for idx in range(len(logs)):
            train_acc, train_loss, val_acc, val_loss = logs[idx]
            f.write('%d,%f,%f,%f,%f\n' %
                    (idx, train_acc, train_loss, val_acc, val_loss))

    logs = np.array(logs)
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(logs[:, 0], label='train acc')
    plt.plot(logs[:, 2], label='val acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.subplot(122)
    plt.plot(logs[:, 1], label='train loss')
    plt.plot(logs[:, 3], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(log_folder, 'plot.png'))
    plt.close()


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 config,
                 train_loader,
                 val_loader,
                 regime=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.regime = regime
        self.log_folder = "logs/" + "_".join([
            config['name'], config['model'],
            str(config['batch_size']),
            str(config['initial_lr']),
            str(config['pretrain']),
            str(datetime.now())
        ])
        os.mkdir(self.log_folder)
        with open(os.path.join(self.log_folder, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    def run(self):
        best_prec1 = 0.0
        logs = []
        for epoch in range(self.config['epochs']):
            adjust_learning_rate(self.optimizer, epoch, self.regime)
            # train for one epoch
            train_acc, train_loss = self.train(epoch)
            # evaluate on validation set
            val_acc, val_loss = self.validate(epoch)
            logs.append((train_acc, train_loss, val_acc, val_loss))
            # remember best prec@1 and save checkpoint
            is_best = val_acc > best_prec1
            best_prec1 = max(val_acc, best_prec1)
            save_checkpoint({
                'epoch': epoch,
                'arch': self.config['model'],
                'name': self.config['name'],
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
            }, self.log_folder, is_best)
            plot_save(logs, self.log_folder)

    def train(self, epoch):
        top1 = AverageMeter()
        losses = AverageMeter()
        # switch to train mode
        self.model.train()
        with tqdm(total=len(self.train_loader)) as pbar:
            for i, (input, target) in enumerate(self.train_loader):
                # measure data loading time
                target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output = self.model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, _ = accuracy(output.data, target, topk=(1, 2))
                losses.update(loss.data[0], input.size(0))
                top1.update(prec1[0], input.size(0))
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                info = 'Train: epoch: %03d, acc: %02.2f, loss:%02.4f' % (
                    epoch, top1.avg, losses.avg)
                pbar.update(1)
                pbar.set_description(info)
        return top1.avg, losses.avg

    def validate(self, epoch):
        top1 = AverageMeter()
        losses = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        with tqdm(total=len(self.val_loader)) as pbar:
            for i, (input, target) in enumerate(self.val_loader):
                target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input, volatile=True)
                target_var = torch.autograd.Variable(target, volatile=True)

                # compute output
                output = self.model(input_var)
                loss = self.criterion(output, target_var)
                prec1, _ = accuracy(output.data, target, topk=(1, 2))
                losses.update(loss.data[0], input.size(0))
                top1.update(prec1[0], input.size(0))
                info = '***Validation***: epoch: %03d, acc: %02.2f, loss:%02.4f' % (
                    epoch, top1.avg, losses.avg)
                pbar.set_description(info)
                pbar.update(1)
        return top1.avg, losses.avg
