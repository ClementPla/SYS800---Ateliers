from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
ARCADE: Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs Dataset Phase 2 consist of two folders with 300 images in each of them as well as annotations. 


The structure of ".JSON" contains three top-level fields: "images", "categories", and "annotations". 
The "images" field contains the unique "id" of the image in the dataset, its "width" and "height" in pixels, and the "file_name" sub-field, 
which contains specific information about the image. 
The "categories" field contains a unique "id" from 1 to 26, and a "name", relating it to the SYNTAX descriptions. 
The "annotations" field contains a unique "id" of the annotation, "image_id" value, relating it to the specific image from the "images" field, and a "category_id" 
relating it to the specific category from the "categories" field. The "segmentation" sub-field contains coordinates of mask edge points in "XYXY" format. 
Bounding box coordinates are given in the "bbox" field in the "XYWH" format, where the first 2 values represent the x and y coordinates of the left-most 
and top-most points in the segmentation mask. The height and width of the bounding box are determined by the difference between the right-most and bottom-most points and the first two values. 
Finally, the "area" field provides the total area of the bounding box, calculated as the area of a rectangle.
"""
class ArcadeDataset(Dataset):
    def __init__(self, root_dir, resolution=(128, 128)):
        if not isinstance(root_dir, Path):
            root_dir = Path(root_dir)
        
        self.root_dir = root_dir
        # Listons les images dans le dossier:
        self.image_paths = list((root_dir / "images").glob("*.png"))
        self.resolution = resolution
        # Chargons le fichier JSON d'annotations pour construire un mapping image_id -> masque
        annotations_folder = self.root_dir / "annotations"
        json_file = (annotations_folder / self.root_dir.name).with_suffix(".json")
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
        
        
        if self.root_dir.name == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Resize(*self.resolution),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()]
                      )
        
    def get_mask(self, image_name, scale_factor):
        # Trouvons l'ID de l'image correspondant au nom donné
        image_id = None
        for img in self.annotations['images']:
            if img['file_name'] == image_name:
                image_id = img['id']
                break
        if image_id is None:
            raise ValueError(f"Image '{image_name}' not found in annotations.")
        
        mask = np.zeros(self.resolution, dtype=np.uint8)
        # Find all annotations for this image_id
        annotations_for_image = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_id]
        for ann in annotations_for_image:
            # We can retrieve the category_id and find the corresponding class value (or name) for the mask
            category_id = ann['category_id']
            for cat in self.annotations['categories']:
                if cat['id'] == category_id:
                    class_value = cat['id']  # We can also use cat['name'] if we want to map to class names instead of IDs
                    break
            coords = np.asarray(ann['segmentation']).squeeze()  # Liste de toutes les annotations
            x_coords = coords[0::2] * scale_factor[0] # x1, x2, ..., xn
            y_coords = coords[1::2] * scale_factor[1]  # y1, y2, ..., yn
            points = np.column_stack((x_coords, y_coords)).astype(np.int32)
            
            cv2.fillPoly(mask, [points], class_value)  # Remplir le masque avec du blanc (255)
            
        return (mask > 0).astype(np.uint8)  # Convertir en masque binaire (0 et 1)
         
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Charger l'image, la redimensionner et la normaliser
        # (cette partie est à implémenter, par exemple avec PIL ou OpenCV)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        image = cv2.resize(image, self.resolution)
        # Rescale factor 
        scale_x = self.resolution[1] / w
        scale_y = self.resolution[0] / h
        mask = self.get_mask(image_path.name, (scale_x, scale_y))
        data = self.transform(image=image, mask=mask)
        image = data['image']
        mask = data['mask']
        mask = mask.long()
        return image, mask
    
    def show(self, idx):
        image, mask = self[idx]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Image")
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='jet')
        axes[1].set_title("Masque de segmentation")
        axes[1].axis('off')
        plt.show()
    

if __name__ == "__main__":
    
    root_train = Path("C:\\Users\\cleme\\OneDrive\\PostDoc\\data\\arcade\\syntax\\train")
    dataset = ArcadeDataset(root_train)
    print(len(dataset))
