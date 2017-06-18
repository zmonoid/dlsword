import torch.nn as nn

def finetune(model, arch, num_classes):
    if arch.startswith('resnet'):
        num_features = list(model.fc.parameters())[0].size(1)
        model.fc = nn.Linear(num_features , num_classes)
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