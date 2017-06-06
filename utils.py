import torch
from datetime import datetime
import os
import csv
import shutil
from tqdm import tqdm


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
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
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


class Trainer(object):
    def __init__(self, model, optimizer, criterion, config, 
            train_loader, val_loader, regime=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.regime = regime
        self.log_folder = "logs/" + "_".join([config['name'],
            config['model'], str(config['batch_size']), str(config['initial_lr']), 
            str(datetime.now())])
        os.mkdir(self.log_folder)

        self.train_writer = open(os.path.join(self.log_folder, 'train.csv'), 'w')
        self.val_writer = open(os.path.join(self.log_folder, 'val.csv'), 'w')

        self.train_writer.write('epoch,step,loss,acc\n')
        self.val_writer.write('epoch,step,loss,acc\n')

    def run(self):
        best_prec1 = 0.0
        for epoch in range(self.config['epochs']):
            adjust_learning_rate(self.optimizer, epoch, 0.1)
            # train for one epoch
            self.train(epoch)
            # evaluate on validation set
            prec1 = self.validate(epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.config['model'],
                'name': self.config['name'],
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
            }, self.log_folder, is_best)
            print "       "
        self.train_writer.close()
        self.val_writer.close()


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
                self.train_writer.write('%d,%d,%f,%f\n' % (epoch, i, losses.val, 
                    top1.val))
                info = 'Train: epoch: %d, loss: %.4f' %  (epoch, losses.avg)
                pbar.update(1)
                pbar.set_description(info)


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
                self.val_writer.write('%d,%d,%f,%f\n' % (epoch, i, losses.val, 
                    top1.val))
                info = 'Val: epoch: %d, accuracy: %.3f' % (epoch, top1.avg)
                pbar.set_description(info)
                pbar.update(1)
        return top1.avg