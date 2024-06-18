import os
import cv2
from torch.utils.data import ConcatDataset, Dataset

# CLASSES = ["Free", "Blowhole", "Break", "Crack", "Fray", "Uneven"]
CLASSES = ["Free", "Blowhole", "Crack"]


class MagneticTilesDatasetClass(Dataset):
    def __init__(self, root, fold="train", class_name="Crack", segmentation=True, transform=None):
        """
        fold: train, val or test
        class_name: one of CLASSES
        """
        self.root = root
        self.fold = fold
        self.class_name = class_name
        self.class_id = CLASSES.index(class_name)   # 0 = background
        self.segmentation = segmentation

        self.class_path = os.path.join(root, f"MT_{class_name}", fold, "ROIs")
        self.images = sorted(os.listdir(os.path.join(self.class_path, "image")))
        self.masks = sorted(os.listdir(os.path.join(self.class_path, "mask")))

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = cv2.imread(os.path.join(self.class_path, "image", image))
        mask = cv2.imread(os.path.join(self.class_path, "mask", mask), cv2.IMREAD_GRAYSCALE) / 255
        mask *= self.class_id

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if not self.segmentation:
            mask = mask.max().item()  # classification label

        return image, mask, self.images[idx]


class MagneticTilesDataset(ConcatDataset):
    def __init__(self, root, fold="train", segmentation=True, **kwargs):
        assert fold in ["train", "val", "test"], f"fold {fold} not in train, val or test"

        # initialize a concat dataset with the corresponding fold
        super().__init__(
            [
                MagneticTilesDatasetClass(root, fold, class_name, segmentation, **kwargs)
                for class_name in CLASSES
            ]
        )
