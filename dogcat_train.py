import argparse
import yaml
import os
import csv
import random
import time
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2


from dataset.custom import ImageList
import models

from utils.trainer import Trainer
from utils.utils import finetune, get_input_size, predict

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
parser.add_argument(
    '--config',
    default='./config/dogcat.yaml',
    type=str,
    help='training configuration file')
parser.add_argument(
    '--test',
    default=None,
    type=str,
    help='To test a model')
parser.add_argument(
    '--vis',
    dest='vis',
    action='store_true',
    help='To repeat test')
parser.add_argument(
    '--repeat',
    default=2,
    type=int,
    help='To repeat test')
args = parser.parse_args()
if args.test is None:
    with open(args.config, 'r') as f:
        config = yaml.load(f)
else:
    with open(os.path.join(args.test, 'config.yaml'), 'r') as f:
        config = yaml.load(f)



"""
Data Loading
"""
fold = 5
root_dir = config['data_folder']
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')

imgs = []
img_list = glob.glob(train_dir + '/*.jpg')
for img in img_list:
    img_name = img.split('/')[-1]
    if img_name.startswith('cat'):
        imgs.append((img, 0))
    elif img_name.startswith('dog'):
        imgs.append((img, 1))

random.shuffle(imgs)
split_indexs = [int(0.2*x*len(imgs)) for x in range(fold)]
# split_index = int(config['train_val_split'] * len(imgs))
train_list = imgs[:split_index]
val_list = imgs[split_index:]

test_imgs = sorted(glob.glob(test_dir + '/*.jpg'))
test_list = [(item, 0) for item in test_imgs]


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

scale_size, crop_size = get_input_size(config['model'])

train_data = ImageList(
    train_list,
    transform=transforms.Compose([
        transforms.Scale(scale_size),
        transforms.RandomSizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), )

val_data = ImageList(
    val_list,
    transform=transforms.Compose([
        transforms.Scale(scale_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]), )

test_data = ImageList(
    test_list,
    transform=transforms.Compose([
        transforms.Scale(scale_size),
        transforms.RandomSizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), )

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True, )

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=4,
    pin_memory=True, )

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=config['batch_size']*8,
    shuffle=False,
    num_workers=4,
    pin_memory=True, )

"""
Training Setup
"""

model = models.__dict__[config['model']](pretrained=config['pretrain'])
model, last_layer, feature_layer = finetune(model, config['model'], config['num_classes'])
model = torch.nn.DataParallel(model).cuda()


train_regime = config['regime']
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam([
    {'params': last_layer.parameters(), 'lr': 1e-3},
    {'params': feature_layer.parameters(), 'lr': 1e-5}]
)

if args.test is None:
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        config,
        train_loader,
        val_loader, 
        regime=None)

    trainer.run()

else:
    best_model = sorted(glob.glob(args.test + '/model_best*'))[-1]
    checkpoint = torch.load(best_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    if args.vis:
        preprocess=transforms.Compose([
            transforms.Scale(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])
        for img, label in imgs:
            prob = predict(model, img, preprocess)[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.imread(img)
            info = "cat: %.2f, dog: %.2f" % (prob[0], prob[1])
            cv2.putText(img, info,(20,20), font, 0.5, (0,255,0),1,cv2.LINE_AA)
            cv2.imshow('image', img)
            is_dog = prob[1] > prob[0]
            if is_dog != label:
                cv2.waitKey(0)
                print 'Wrong: %s' % img

    else:
        result = np.zeros((len(test_list), config['num_classes']))
        softmax = nn.Softmax()
        for x in range(args.repeat):
            probs_list = []
            names_list = []
            with tqdm(total=len(test_loader)) as pbar:
                for i, (input, target, names) in enumerate(test_loader):
                    input_var = torch.autograd.Variable(input, volatile=True)
                    prob = np.clip(softmax(model(input_var)).data.cpu().numpy(), 0.01, 0.99)
                    img_id = [item.split('/')[-1].split('.')[0] for item in names]
                    names_list += img_id
                    probs_list += [item for item in prob]
                    pbar.update(1)
            result += np.array(probs_list)
        
        print result / args.repeat


        output = zip(names_list, prob)
        with open(os.path.join(args.test, 'result_%s.csv' % checkpoint['epoch']), 'w') as f:
            f.write('id,label\n')
            for name, prob in output:
                f.write('%s,%f\n' % (name, prob[1]))

