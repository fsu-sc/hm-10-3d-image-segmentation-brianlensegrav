import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
image_dir = os.path.join(data_dir, "imagesTr")
label_dir = os.path.join(data_dir, "labelsTr")

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.header.get_zooms(), img.shape

def show_slices(img, title=""):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img[img.shape[0] // 2, :, :], cmap='gray')
    axs[1].imshow(img[:, img.shape[1] // 2, :], cmap='gray')
    axs[2].imshow(img[:, :, img.shape[2] // 2], cmap='gray')
    fig.suptitle(title)
    plt.show()

def main():
    filenames = sorted(os.listdir(image_dir))
    print(f"Number of training images: {len(filenames)}")

    img_path = os.path.join(image_dir, filenames[0])
    img_data, spacing, shape = load_nifti(img_path)
    print(f"Image shape: {shape}")
    print(f"Voxel spacing: {spacing}")

    show_slices(img_data, title="Sample Image")

if __name__ == "__main__":
    main()
