import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class PatientDataset(Dataset):
    def __init__(self, images_path, labels: pd.DataFrame):
        self.images_path = images_path
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        pid = row["patient_id"]
        labels = row.drop(["patient_id"]).values
        labels = labels.astype(np.float64)
        labels_tensor = torch.tensor(labels)

        images = torch.zeros((100, 128, 128), dtype=torch.float64)
        image_files = os.listdir(f"{self.images_path}/{pid}")
        image_files = [os.path.splitext(fname)[0] for fname in image_files]
        image_files = sorted([int(x) for x in image_files])
        image_files = [f"{f}.png" for f in image_files]

        for idx, fname in enumerate(image_files):
            image = Image.open(f"{self.images_path}/{pid}/{fname}").convert("L")
            tensor = self.transform(image)
            images[idx] = tensor

        images = images.unsqueeze(0)

        return images, labels_tensor
