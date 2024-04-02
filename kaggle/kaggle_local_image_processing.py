import os
import shutil

import numpy as np
from tqdm import tqdm


def get_images_information(images_path, minimum_images):
    patients = os.listdir(images_path)

    min_images = None
    max_images = None
    patients_too_few_images = []

    for patient in tqdm(patients):
        files = os.listdir(f"{images_path}/{patient}")
        files_amount = len(files)
        if min_images is None or files_amount < min_images:
            min_images = files_amount
        if max_images is None or files_amount > max_images:
            max_images = files_amount
        if files_amount < minimum_images:
            patients_too_few_images.append(patient)

    return min_images, max_images, patients_too_few_images


def copy_n_images_proportionally(original_images_path, images_path_reduced, number_of_images, patients_to_skip):
    if not os.path.exists(images_path_reduced):
        os.makedirs(images_path_reduced)

    patients = os.listdir(original_images_path)
    for patient in tqdm(patients):
        if patient in patients_to_skip:
            continue

        files = os.listdir(f"{original_images_path}/{patient}")
        files_indices = np.int32(np.linspace(0, len(files) - 1, number_of_images))
        files = list(map(lambda fname: os.path.splitext(fname)[0], files))
        files = sorted([int(x) for x in files])
        files = np.array(files)
        files = files[files_indices]
        files = [f"{file}.png" for file in files]

        new_patient_folder = f"{images_path_reduced}/{patient}"
        if not os.path.exists(new_patient_folder):
            os.makedirs(new_patient_folder)

        for file in files:
            shutil.copy(f"{original_images_path}/{patient}/{file}", f"{new_patient_folder}/{file}")


def get_reduced_df(df, patients_to_skip):
    patients_too_few_images_int = [int(pid) for pid in patients_to_skip]
    reduced_df = df[~df['patient_id'].isin(patients_too_few_images_int)]
    return reduced_df
