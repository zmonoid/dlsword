from PIL import Image, ImageOps
import torch.utils.data as data


def default_loader(path):
    """
    load image from path
    """
    return Image.open(path).convert('RGB')


class ImageList(data.Dataset):
    """
    items should be like a list of form:
    [(img1_path, label1), (img2_path, label2), ...]
    """

    def __init__(self, items, transform=None, target_transform=None):
        self.items = items
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.items[index]
        img = default_loader(path)
        #img = ImageOps.equalize(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path

    def __len__(self):
        return len(self.items)


class NumData(data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = list(X)
        self.Y = list(Y)
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)
