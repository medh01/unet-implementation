import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class GrayscaleEmbryoICMDataset(Dataset):
    def __init__(self, image_dir, icm_dir, transform=None, target_size=(512, 512)):
        self.image_dir = image_dir
        self.icm_dir = icm_dir
        self.target_size = target_size
        self.transform = transform

        self.image_files = []
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".bmp")])

        for f in image_files:
            base = f.rsplit('.', 1)[0]
            expected_mask = f"{base} ICM_Mask.bmp"
            for mask_file in os.listdir(icm_dir):
                if mask_file.lower() == expected_mask.lower():
                    self.image_files.append(f)
                    break

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        base_name = image_file.rsplit('.', 1)[0]

        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("L").resize(self.target_size)
        image = transforms.ToTensor()(image)

        expected_mask = f"{base_name} ICM_Mask.bmp"
        for mask_file in os.listdir(self.icm_dir):
            if mask_file.lower() == expected_mask.lower():
                mask_path = os.path.join(self.icm_dir, mask_file)
                break

        mask = Image.open(mask_path).convert("L").resize(self.target_size)
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)

        return image, mask_tensor
