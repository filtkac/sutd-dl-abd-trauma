import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import models
from data.dataset import PatientDataset


def train(args):
    df = pd.read_csv(args.labels_path)

    # Create an empty list to store file paths
    file_paths = []

    # Iterate over files in images_path directory
    for filename in os.listdir(args.images_path):
        file_path = os.path.join(args.images_path, filename)
        # Append the file path to the list
        file_paths.append(file_path)

    # Add a new column 'file_path' to the DataFrame
    df["file_path"] = file_paths

    # Split the dataset into train, validation, and test sets
    train_data, val_test_data = train_test_split(df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

    train_dataset = PatientDataset(args.images_path, train_data)
    val_dataset = PatientDataset(args.images_path, val_data)
    test_dataset = PatientDataset(args.images_path, test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # get one sample
    inputs, labels = train_dataset[0]

    if args.model == "cnn":
        model = models.ConvNet3D(
            in_channels=inputs.shape[0],
            out_channels=labels.shape[0],
            depth=inputs.shape[1],
            height=inputs.shape[2],
            width=inputs.shape[3],
        )
    elif args.model == "unet":
        model = ...
    elif args.model == "vit":
        model = ...
    else:
        raise ValueError("Invalid model selected for training.")

    if args.cuda:
        model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()  # multi-label classification
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0

        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = batch
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero gradients for every batch
            optimizer.zero_grad()
            outputs = model(inputs.float())
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for abdominal trauma detection.")
    parser.add_argument("--images_path", type=str, required=True, help="path to directory of images")
    parser.add_argument("--labels_path", type=str, required=True, help="path to labels csv file")
    parser.add_argument(
        "--model", type=str, default="cnn", choices=["cnn", "unet", "vit"], help="model to train with (default: cnn)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--cuda", action="store_true", help="enables GPU training with CUDA")
    args = parser.parse_args()

    train(args)
