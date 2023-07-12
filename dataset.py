import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import Data_Augmentation

class MapDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
    def __len__(self):
        return len(self.list_files)
    def __getitem__(self, index):
        image_file = self.list_files[index]
        image_path = os.path.join(self.root_dir,image_file)
        image = np.array(Image.open(image_path))
        input_image = image[:,:600,:]
        target_image = image[:, 600:, :]

        augmentations = Data_Augmentation.both_transform(image=input_image,image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = Data_Augmentation.transform_only_input(image=input_image)["image"]
        target_image = Data_Augmentation.transform_only_mask(image=target_image)["image"]

        return input_image,target_image

if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset,batch_size=5)
    for x,y in loader:
        print(x.shape)
        save_image(x,"x.png")
        save_image(y,"y.png")
        import sys
        sys.exit()
