import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .custom_transforms import Normalize, ToTensor, RandomHorizontalFlip, RandomGaussianBlur

__all__ = ['HorseDataset', 'make_dataloader']

class HorseDataset(Dataset):
    def __init__(self, args, mode, transforms=None):
        self.root = args.root
        self.imgs = os.path.join(self.root, "horse")
        self.masks = os.path.join(self.root, "mask")
        self.phase = mode
        if self.phase == 'train':
            data_list = args.train_list
        elif self.phase == 'test':
            data_list = args.test_list
        items = []
        with open(data_list,'r') as f:
            for line in f:
                img_name = line.strip('\n')
                items.append(img_name)
        self.items = items
        self.transforms = transforms

    def __getitem__(self, idx):

        img_path = os.path.join(self.imgs, self.items[idx])
        img = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(self.masks, self.items[idx])
        mask = Image.open(mask_path)

        inputs = {'image': img, 'mask': mask}

        if self.transforms is not None:
            inputs = self.transforms(inputs)

        if self.phase == 'train':
            return inputs['image'], inputs['mask']
        else:
            return self.items[idx], np.array(img), inputs['image'], inputs['mask']

    def __len__(self):
        return len(self.items)


def make_dataloader(args):

    seed = args.seed
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    train_dataset = HorseDataset(args, 'train', transforms=transforms.Compose([
                                                    RandomHorizontalFlip(),
                                                    RandomGaussianBlur(),
                                                    normalize,
                                                    ToTensor()
                                                 ]))

    test_dataset = HorseDataset(args, 'test', transforms=transforms.Compose([
                                                    normalize,
                                                    ToTensor()
                                                 ]))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    return train_loader, test_loader
