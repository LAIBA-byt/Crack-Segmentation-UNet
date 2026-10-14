# Crack-Segmentation-UNet

End-to-end semantic segmentation pipeline using U-Net with a pre-trained ResNet34 encoder (PyTorch) for pixel-level surface crack detection. Uses a combined BCE + Dice loss with positive class weighting and empirical threshold tuning to handle severe foreground-background class imbalance.
# Dataset
- Custom **synthetic** surface crack dataset, generated to test and validate the segmentation pipeline on continuous, thin, linear crack patterns.
- Size: [N] images, resolution [e.g. 256x256], train/val split [e.g. 80/20]
- Class imbalance: crack pixels are less than 3% of all pixels.
## Method
- Model: U-Net with ResNet34 encoder (Segmentation Models PyTorch)
- Loss: BCE + Dice Loss with `pos_weight`
- Augmentation: Albumentations
- Framework: PyTorch, trained on Google Colab
#Results (validation set)
| Threshold | Dice | Precision | Recall |
|-----------|------|-----------|--------|
| 0.5 | 0.4896 | 0.3244 | 0.9983 |
| 0.9 (best) | 0.6659 | 0.5122 | 0.9642 |
Low threshold gives very high recall but many false positives; raising the threshold improves precision and Dice.
## Sample Predictions
(Add 3-4 images: input | ground truth | prediction)
## Limitations
- Synthetic data does not fully capture real-world texture, shadows and noise.
- The model over-segments near boundaries and struggles with ultra-thin micro-cracks.
## Future Work
- Evaluate on a real dataset (e.g. CRACK500 / DeepCrack)
- Try Focal or Tversky loss to improve precision
## How to Run
1. Open the notebook in Google Colab
2. Install requirements: `pip install segmentation-models-pytorch albumentations opencv-python`
3. Run all cells
