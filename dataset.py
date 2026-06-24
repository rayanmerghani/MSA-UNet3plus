
import os
from pathlib import Path
from PIL import Image
import PIL.ImageOps

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

class OrganData(Dataset):

    def __init__(self,
                 data_path: str,
                 size: tuple = (256,256),
                 transform=None,
                 mode=None):
        
        self.image_files = self.get_filenames(os.path.join(data_path, 'image'))
        self.label_files = self.get_filenames(os.path.join(data_path, 'label'))
        self.image_files.sort()
        self.label_files.sort()
        

        assert self.check_validity(self.image_files, self.label_files), \
               'inconsistent pairs of data'

        self.size = size
        self.transform = transform
        self.mode = mode

        self.common_transform = transforms.ToTensor()

        if 'JSRT_dataset' in data_path.split('/'):
            self.invert = True
        else:
            self.invert = False


    def check_validity(self,
                       inputs: list,
                       targets: list):
        # image_filenames = [i.split('/')[-1] for i in inputs]
        # label_filenames = [i.split('/')[-1] for i in targets]
        image_filenames = [i.split('\\')[-1] for i in inputs]
        label_filenames = [i.split('\\')[-1] for i in targets]
        # print('image_filenames',image_filenames)
        # print('label_filenames',label_filenames)
        return image_filenames == label_filenames

    def get_filenames(self,
                      path: str):
        filename_list = []
        #print('path',path)
        for filename in os.listdir(path):
            filename_list.append(os.path.join(path, filename))
        return filename_list

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("L")
        if self.invert:
            img = PIL.ImageOps.invert(img)
        img = img.resize(self.size)

        label = Image.open(self.label_files[idx]).convert("L")
        seg_label = label.resize(self.size, resample=Image.NEAREST)
        cont_label = label.resize((128,128), resample=Image.NEAREST)

        if self.transform:
            img = self.transform(img)
            seg_label = self.common_transform(seg_label)
            cont_label = self.common_transform(cont_label)
        else:
            img = self.common_transform(img)
            seg_label = self.common_transform(seg_label)
            cont_label = self.common_transform(cont_label)

        if self.mode == 'test':
            fname = Path(self.image_files[idx]).stem
            return img, seg_label, cont_label, fname
        else:
            return img, seg_label, cont_label

    def __len__(self):
        return len(self.image_files)

