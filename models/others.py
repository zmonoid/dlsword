import torch.nn as nn
import torch
import torch.nn.functional as F

class FinetuneModel(nn.Module):
    def __init__(self, model, arch, num_classes):
        super(FinetuneModel, self).__init__()
        # Everything except the last linear layer

        if arch.startswith('vgg'):
            self.features = model.features
            self.classifier = nn.Sequential(*list(
                model.classifier.children())[:-1])
            self.classifier = nn.Linear(4096, num_classes)

        elif arch.startswith('resnet'):
            self.features = nn.Sequential(*list(
                model.children())[:-1])
            num_features = list(model.fc.parameters())[0].size(1)
            self.classifier = nn.Sequential(nn.Dropout(), 
                nn.Linear(num_features, num_classes))

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y