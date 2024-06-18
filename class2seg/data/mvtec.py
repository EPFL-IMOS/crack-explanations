import os
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

CLASSES = ["good", "hole"]
# CLASSES = ["good", "crack"]
# CLASSES = ["good", "cut"]

class MVTecDataset(Dataset):

    def __init__(self, root, fold="train", transform=None, random_state=42, **kwargs):
        """
        fold: train, val or test
        """
        assert fold in ["train", "val", "test"], f"fold {fold} not in train, val or test"

        self.root = root
        images = []
        labels = []

        for class_id, class_name in enumerate(CLASSES):
            for filename in sorted(os.listdir(os.path.join(self.root, class_name))):
                images.append(os.path.join(class_name, filename))
                labels.append(class_id)

        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=random_state)

        if fold == "train":
            self.images = images_train
            self.labels = labels_train
        elif fold == "test" or fold == "val":
            self.images = images_test
            self.labels = labels_test

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(os.path.join(self.root, image), cv2.IMREAD_COLOR)

        if self.transform is not None:
            image, _ = self.transform(image)

        return image, label, self.images[idx]
