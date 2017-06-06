import argparse
import yaml
import os
import csv
import random
import time

from dataset.imglist import ImageList
import models
from models.others import FinetuneModel
from utils.trainer import Trainer


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms


"""
Configuration
"""
parser = argparse.ArgumentParser(description='PyTorch Image Training')
parser.add_argument('--config', default='./config/invasive.yaml', type=str,
                    help='training configuration file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f)

"""
Data Loading
"""
root_dir = config['data_folder']
label_file = os.path.join(root_dir, 'train_labels.csv')

imgs = []
with open(label_file, 'rb') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img = os.path.join(root_dir, 'train', row['name'] + '.jpg')
        label = int(row['invasive'])
        imgs.append((img, label))

random.shuffle(imgs)
split_index = int(config['train_val_split'] * len(imgs))
train_list = imgs[:split_index]
val_list = imgs[split_index:]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

train_data = ImageList(
    train_list, 
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
)

val_data = ImageList(
    val_list, 
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


"""
Training Setup
"""
model = models.__dict__[config['model']](pretrained=True)
model = FinetuneModel(model, config['model'], config['num_classes'])
model = torch.nn.DataParallel(model).cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(
    model.module.classifier.parameters(), 
    config['initial_lr'], 
    momentum=config['momentum'],
    weight_decay=config['weight_declay'])


trainer = Trainer(model,
    optimizer, 
    criterion,
    config,
    train_loader, 
    val_loader,
    )


trainer.run()