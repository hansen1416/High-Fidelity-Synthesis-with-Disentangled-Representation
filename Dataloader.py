import os

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

project_dir = '/home/hlz/High-Fidelity-Synthesis-with-Disentangled-Representation'

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

def return_data(image_size = 64):

    batch_size = 64
    num_workers = 1

    root = os.path.join(project_dir, 'data/CelebA_{}'.format(image_size))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        ])
    train_kwargs = {'root':root, 'transform':transform}

    train_data = CustomImageFolder(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    return train_loader