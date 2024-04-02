import os
from glob import glob

import cv2
import dicomsdl
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def visualize_image(orig_image, image):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_image, cmap='gray')
    axes[0].set_title('Original DICOM Image')
    axes[1].imshow(image, cmap='gray')
    axes[1].set_title('Processed DICOM Image')
    plt.show()


def __dataset__to_numpy_image(self, index=0):
    info = self.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    self.copyFrameData(index, outarr)
    return outarr


def glob_sorted(path):
    return sorted(glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0]))


def get_rescaled_image(dcm, img):
    resI, resS = dcm.RescaleIntercept, dcm.RescaleSlope
    img = resS * img + resI
    return img


# abdominal window is set as the default arguments
def get_windowed_image(img, WL=50, WW=400):
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X * 255.0).astype('uint8')

    return X


def load_volume(dcms):
    volume = []
    pos_zs = []
    k = 1

    for dcm_path in dcms:
        pydcm = pydicom.dcmread(dcm_path)

        pos_z = pydcm[(0x20, 0x32)].value[-1]
        pos_zs.append(pos_z)

        dcm = dicomsdl.open(dcm_path)

        orig_image = dcm.to_numpy_image()

        image = get_rescaled_image(dcm, orig_image)
        image = get_windowed_image(image)
        if np.min(image) < 0:
            image = image + np.abs(np.min(image))
        image = image / image.max()
        image = (image * 255).astype(np.uint8)

        volume.append(image)

        # Visualize the image
        # if k == 1:
        #    k +=1
        #    print(image.shape)
        #    visualize_image(orig_image, image)

    return np.stack(volume)


def save_volume(volume, patient, output_folder_root, output_dims):
    volume = np.stack([cv2.resize(x, output_dims) for x in volume])

    # to visualize the squished image
    # print(volume.shape)
    # image = volume[0]
    # visualize_image(image, image)

    output_folder = f"{output_folder_root}/{patient}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(volume)):
        filename = os.path.join(output_folder, f"{i}.png")
        plt.imsave(filename, volume[i], cmap='gray')
