from copy import copy
import os
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class FragmentWithInkCropDataset(Dataset):
    def __init__(self, root_dir, crop_size = 1, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        # load the images
        for image_file in os.listdir(root_dir + '/surface_volume'):
            if int(image_file[:2])<0:
                continue
            self.images.append(image_file)
        # load the mask
        self.mask = Image.open(root_dir + '/mask.png')
        # load the target
        self.target = Image.open(root_dir + '/inklabels.png')

        self.crop_size = crop_size

        print(f"Created FragmentWithInkCropDataset with {len(self)} crops")
        # plot the targ🇪🇹et

        
    def __len__(self):
        #return the number of crops
        return (self.mask.size[0]//self.crop_size)*(self.mask.size[1]//self.crop_size)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get all images and take a crop of size crop_size with offset idx
        images = []
        for image_file in self.images:
            image = Image.open(self.root_dir + '/surface_volume/' + image_file)
            num_horiz_crops = image.size[1]//self.crop_size
            num_vert_crops = image.size[0]//self.crop_size
            image = image.crop(((idx%num_horiz_crops)*self.crop_size, (idx//num_horiz_crops)*self.crop_size, (idx%num_horiz_crops)*self.crop_size+self.crop_size, (idx//num_horiz_crops)*self.crop_size+self.crop_size))
            image = transforms.ToTensor()(image)
            image = transforms.Normalize((0.5,), (0.5,))(image)
            images.append(image)
        images = torch.stack(images)
        mask = self.mask
        # also crop the target to match the image


        target = self.target.crop(((idx%num_horiz_crops)*self.crop_size, (idx//num_horiz_crops)*self.crop_size, (idx%num_horiz_crops)*self.crop_size+self.crop_size, (idx//num_horiz_crops)*self.crop_size+self.crop_size))
        # make target a binary image
        target = np.array(target)
        target[target>0] = 255
        target = Image.fromarray(target)
        target = transforms.ToTensor()(target)
       
        # crop the mask


        mask = mask.crop(((idx%num_horiz_crops)*self.crop_size, (idx//num_horiz_crops)*self.crop_size, (idx%num_horiz_crops)*self.crop_size+self.crop_size, (idx//num_horiz_crops)*self.crop_size+self.crop_size))
        

        if self.transform:
            mask = self.transform(mask)
        
        
        return images, mask, target