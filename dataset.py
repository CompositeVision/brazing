import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as T
import glob


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None) -> None:
        self.image_files = sorted(glob.glob("processed/images/*"))
        self.mask_files = sorted(glob.glob("processed/masks/*"))
        self.transforms = transforms or T.ToImageTensor()

    def __getitem__(self, index):
        image = torchvision.io.read_image(self.image_files[index])
        mask = (torchvision.io.read_image(self.mask_files[index]) / 255).long()
        
        image, mask = self.transforms(image, mask)
        return image, mask

    def __len__(self):
        return len(self.image_files)
    

