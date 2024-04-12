import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class PatientDataset(Dataset):
    def __init__(self, images_path, data, n_slices=64):
        self.images_path = images_path
        self.transform = transforms.ToTensor()
        self.n_slices = n_slices

        self.data = self.__process_slices_proportionally(images_path, data, n_slices)

    def __process_slices_proportionally(self, original_images_path, data, n_slices):
        file_slices = []
        for row in data.itertuples():
            patient_dir = os.path.join(original_images_path, str(row.patient_id))
            files = os.listdir(patient_dir)
            files_indices = np.int32(np.linspace(0, len(files) - 1, n_slices))
            files = list(map(lambda fname: os.path.splitext(fname)[0], files))
            files = sorted([int(x) for x in files])
            files = np.array(files)
            files = files[files_indices]
            files = [os.path.join(patient_dir, f"{file}.png") for file in files]
            file_slices.append(files)

        data["slices"] = file_slices
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # pid = row["patient_id"]
        labels = row.drop(["patient_id", "slices"]).values
        labels = labels.astype(np.float64)
        labels_tensor = torch.tensor(labels)

        images = torch.zeros((self.n_slices, 128, 128), dtype=torch.float64)
        for idx, fname in enumerate(row["slices"]):
            image = Image.open(fname).convert("L")
            tensor = self.transform(image)
            images[idx] = tensor

        images = images.unsqueeze(0)

        return images, labels_tensor
