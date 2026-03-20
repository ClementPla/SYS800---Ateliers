import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir: str | Path, split: str = "train"):
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        self.root_dir = root_dir
        
        image_dir = root_dir / "train"
        labels = pd.read_csv(root_dir / "trainLabels.csv")
        categories = sorted(labels['label'].unique())
        self.classes = categories
        self.classes_to_idx = {category: idx for idx, category in enumerate(categories)}
        train_labels, test_labels = train_test_split(labels, test_size=0.3, random_state=42)
        train_labels, val_labels = train_test_split(train_labels, test_size=0.1, random_state=42)
        
        self.data_augmentation_pipeline = None
        match split:
            case "train":
                self.labels = train_labels
                self.data_augmentation_pipeline = A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.25),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                ]
            ) 
            case "val":
                self.labels = val_labels
            case "test":
                self.labels = test_labels
            case _:
                raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")
        self.image_paths = [image_dir / f"{row['id']}.png" for _, row in self.labels.iterrows()]
        
        
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels.iloc[idx]['label']
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        # Augmentation de données
        if self.data_augmentation_pipeline is not None:
            transformed = self.data_augmentation_pipeline(image=image)
            image = transformed['image']
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W) format
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        return image, torch.tensor(self.classes_to_idx[label], dtype=torch.long)
    
    
    
    def show(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self))
        image, label = self[idx]
        image = image.permute(1, 2, 0).numpy()  # Convert back to (H, W, C) format
        plt.imshow(image)
        plt.title(f"Label: {self.classes[label]}")
        plt.axis('off')
        plt.show()