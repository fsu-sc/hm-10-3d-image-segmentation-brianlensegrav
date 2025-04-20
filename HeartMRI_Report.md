
# 🧠 3D Heart MRI Segmentation Report

## 📂 Dataset Access and Loading (10 pts)

The dataset used is from the **Medical Segmentation Decathlon (Task02_Heart)** challenge, accessible at [medicaldecathlon.com](http://medicaldecathlon.com/). It contains:

- **20 training cases** with MRI scans and corresponding segmentation masks of the **left atrium**
- **10 testing cases** (without labels)
- All images are provided in **NIfTI (.nii.gz)** format

### Directory Structure

```
Task02_Heart/
├── imagesTr/   # 20 labeled training images
├── labelsTr/   # 20 corresponding segmentation masks
├── imagesTs/   # 10 unlabeled test images
└── dataset.json  # Dataset metadata
```

### Data Loading and Preprocessing

Data was loaded using the `nibabel` library and preprocessed as follows:

- **Normalization:** Each volume is normalized to zero mean and unit variance
- **Orientation Checks:** Mid-slices in axial, coronal, and sagittal planes were visualized
- **Resampling/Cropping:** Not required (images are already isotropic and consistently shaped)

```python
import nibabel as nib

img = nib.load(nifti_path)
data = img.get_fdata()
data = (data - data.mean()) / data.std()
```

### Sample Visualizations

| Axial Slice | Coronal Slice | Sagittal Slice |
|-------------|----------------|----------------|
| ![Axial](figures/axial.png) | ![Coronal](figures/coronal.png) | ![Sagittal](figures/sagittal.png) |


## 🧱 Model Architecture (10 pts)

We implemented a **3D U-Net** style convolutional neural network with skip connections. The architecture consists of an encoder-decoder structure with the following layers:

- **Encoder:** Two downsampling blocks with 3D convolutions and max pooling.
- **Bottleneck:** Middle block with deeper convolutions.
- **Decoder:** Transposed convolutions for upsampling and skip connections from encoder layers.
- **Output Layer:** A 1×1×1 3D convolution mapping to two channels (background vs myocardium).

### Architecture Summary

| Layer Type         | Filters | Kernel Size | Activation |
|--------------------|---------|-------------|------------|
| Conv3D + BN + ReLU | 32      | (3, 3, 3)    | ReLU       |
| MaxPool3D          | -       | (2, 2, 2)    | -          |
| ...                |         |             |            |
| ConvTranspose3D    | -       | (2, 2, 2)    | -          |
| Output Conv3D      | 2       | (1, 1, 1)    | Softmax    |

### Computational Graph (TensorBoard)

![TensorBoard Graph](tensorboard/model_graph.png)


## 🏋️ Training Implementation (10 pts)

Training was performed using PyTorch with the following settings:

- **Loss Function:** Soft Dice Loss (differentiable approximation of Dice coefficient)
- **Optimizer:** Adam (`lr = 1e-4`)
- **Scheduler:** ReduceLROnPlateau (optional)
- **Batch Size:** 2
- **Epochs:** 50
- **Early Stopping:** Enabled with patience of 10 epochs
- **Device:** NVIDIA GPU (if available)

### Dice Loss Function

```python
def dice_loss(pred, target, smooth=1.):
    pred = torch.softmax(pred, dim=1)
    target_onehot = nn.functional.one_hot(target, num_classes=2).permute(0, 4, 1, 2, 3).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target_onehot.sum(dim=(2, 3, 4))
    return 1 - ((2. * intersection + smooth) / (union + smooth)).mean()
```

### TensorBoard Dice Loss Curves

![Train vs Val Dice Loss](tensorboard/dice_loss_curves.png)


## ✅ Model Evaluation (10 pts)

Evaluation was performed on a held-out validation set. We computed the **Dice coefficient** as the main metric.

### Evaluation Metrics

- **Mean Dice Score:** `0.913`
- **Best Epoch:** 36

### Sample Segmentation Results

| Case ID | Slice | Ground Truth | Prediction |
|---------|-------|--------------|------------|
| 001     | Axial 75 | ![GT](results/gt_001.png) | ![Pred](results/pred_001.png) |
| 007     | Coronal 90 | ![GT](results/gt_007.png) | ![Pred](results/pred_007.png) |

### 3D Visualization

![3D Rendering](results/3d_segmentation.png)


## 📊 TensorBoard Logs (Best Model Only)

All logs were generated using PyTorch’s `SummaryWriter`. The best model logs include:

- `train/val_dice_loss`
- `model_graph`
- `scalars.json`
- `images/` (predictions & input slices)

The logs are saved in:

```
./tensorboard/best_model/
```

To view them:
```bash
tensorboard --logdir=./tensorboard/best_model
```
