import torch
import torch.nn as nn
from PIL import Image
import sys
import cv2

sys.path.append("../")

from dataset.custom import ImageList
import torchvision.transforms as transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def finetune(model, config):
    arch = config['model']
    num_classes = config['num_classes']
    if arch.startswith('resnet'):
        num_features = list(model.fc.parameters())[0].size(1)
        model.fc = nn.Linear(num_features, num_classes)
        feature_layer = nn.Sequential(*list(model.children())[:-1])
        last_layer = model.fc

    elif arch.startswith('vgg'):
        fc = list(model.classifier.children())[:-1]
        fc.append(nn.Linear(4096, num_classes))
        model.classifier = nn.Sequential(*fc)
        last_layer = model.classifier
        feature_layer = model.features

    elif arch.startswith('inception'):
        model.fc = nn.Linear(2048, num_classes)
        last_layer = model.fc
        feature_layer = nn.Sequential(*list(model.children())[:-1])

    elif arch.startswith('densenet'):
        num_features = list(model.classifier.parameters())[0].size(1)
        model.classifier = nn.Linear(num_features, num_classes)
        last_layer = model.classifier
        feature_layer = nn.Sequential(*list(model.children())[:-1])

    return model, last_layer, feature_layer


def get_input_size(model):
    if model.startswith('inception'):
        scale_size = 334
        crop_size = 299
    else:
        scale_size = 256
        crop_size = 229
    return scale_size, crop_size


def predict(model, img_path, preprocess):
    img = Image.open(img_path)
    input_var = preprocess(img).unsqueeze_(0)
    input_var = torch.autograd.Variable(input_var)
    softmax = nn.Softmax()
    prob = softmax(model(input_var)).data.cpu().numpy()
    return prob


def get_loader(img_list, config, mode):
    scale_size, crop_size = get_input_size(config['model'])
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Scale(scale_size),
            transforms.RandomSizedCrop(crop_size),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize
        ]),
        'val':
        transforms.Compose([
            transforms.Scale(crop_size), transforms.CenterCrop(crop_size),
            transforms.ToTensor(), normalize
        ]),
    }
    data_set = ImageList(img_list, data_transforms[mode])
    batch_size = {
        'train': config['batch_size'],
        'val': config['batch_size'] * 8,
        'vis': 1
    }
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size[mode],
        shuffle=True if mode == 'train' else False,
        num_workers=4,
        pin_memory=True, )

    return data_loader


def play_data(imgs, model, config, sformat):
    scale_size, crop_size = get_input_size(config['model'])
    preprocess = transforms.Compose([
        transforms.Scale(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    for img_, label in imgs:
        prob = predict(model, img_, preprocess)[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.imread(img_)
        info = sformat % tuple(prob)
        cv2.putText(img, info, (20, 20), font, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.imshow('image', img)
        is_invasive = prob[1] > prob[0]
        if is_invasive != label:
            cv2.waitKey(0)
            print 'Wrong: %s' % img_
