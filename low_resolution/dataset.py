import numpy as np
import os
import torch
from PIL import Image

import utils


def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """
    real, y = next(data_loader)
    # print(f"Type of 'real': {type(real)}, Type of 'y': {type(y)}")
    # print(f"First element of 'real': {real[0] if isinstance(real, (list, tuple)) else real}")
    # print(f"First element of 'y': {y[0] if isinstance(y, (list, tuple)) else y}")

    real, y = real.to(device), y.to(device)

    return real, y


def sample_from_gen(args, device, batch_size, gen_dim_z, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        batch_size, gen_dim_z, device, args.gen_distribution
    )
    pseudo_y = utils.sample_pseudo_labels(
        num_classes, batch_size, device
    )

    fake = gen(z)
    return fake, pseudo_y, z


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, args, root='', transform=None, data_name=None):
        super(FaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.path = self.root
        self.data_name = data_name

        self.label_list = []

        num_classes = len([lists for lists in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, lists))])

        for idx in range(num_classes):
            class_path = os.path.join(self.path, str(idx))
            for _, _, files in os.walk(class_path):
                for img_name in files:
                    image_path = os.path.join(class_path, img_name)
                    # print(image_path)

                    image = Image.open(image_path)
                    image = image.convert('RGB')

                    if self.data_name == 'facescrub':
                        if image.size != (64, 64):
                            image = image.resize((64, 64), Image.LANCZOS)

                    self.images.append(image)
                    self.label_list.append(idx)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.label_list[index]

        if self.transform != None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
