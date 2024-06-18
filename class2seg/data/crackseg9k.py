import os
import cv2
from torch.utils.data import Dataset


class CrackSeg9kDataset(Dataset):
    def __init__(self, root, fold="train", transform=None, **kwargs):
        """
        fold: train, val or test
        """
        self.root = root
        self.fold = fold
        if fold == "test":
            filename = "test.txt"
        else:
            filename = "train_shuffled.txt"
        with open(os.path.join(root, filename), "r") as f:
            self.samples = f.readlines()
        self.samples = [s.replace("\n", "") for s in self.samples]

        # Remove non-existent files
        images = os.listdir(os.path.join(root, "Images"))
        self.samples = [s for s in self.samples if s in images]

        split = 0.9
        if fold == "train":
            self.samples = self.samples[:int(split*len(self.samples))]
        elif fold == "val":
            self.samples = self.samples[int(split*len(self.samples)):]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]

        image = cv2.imread(os.path.join(self.root, "Images", filename))
        mask = cv2.imread(os.path.join(self.root, "Masks", filename), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(int)
        # mask = (mask > 0).astype(int)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, self.samples[idx]
