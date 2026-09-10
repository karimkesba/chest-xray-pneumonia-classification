# Chest X-Ray Pneumonia Classification

A deep learning project for binary classification of chest X-ray images into:

- NORMAL
- PNEUMONIA

## Workflow

1. Load and extract the dataset.
2. Split the data into:
   - 70% Training
   - 15% Validation
   - 15% Testing
3. Resize images to `224×224` and use grayscale images.
4. Apply data augmentation to the training set:
   - Rotation
   - Width/height shifting
   - Zoom
   - Horizontal flipping
5. Build and train a custom CNN from scratch.
6. Handle class imbalance using:
   - Class weights
   - Equal-class downsampling
7. Train a second custom CNN on the balanced dataset.
8. Apply Transfer Learning using MobileNetV2 pretrained on ImageNet.
9. Fine-tune the last 20 layers of MobileNetV2.
10. Evaluate the models using:
    - Accuracy
    - Confusion Matrix
    - Precision
    - Recall
    - F1-score
11. Analyze misclassified test images.

## Final Result

The best model was **MobileNetV2 with Transfer Learning and Fine-Tuning**.

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 90.56% |
| Custom CNN + Downsampling | 91.39% |
| MobileNetV2 + Fine-Tuning | **93.07%** |

Final confusion matrix:

```text
[[232   6]
 [ 27 211]]
```

## Technologies

- Python
- TensorFlow / Keras
- CNN
- MobileNetV2
- Transfer Learning
- Fine-Tuning
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Author

**Karim Mohamed Kesba**

AI / Machine Learning Engineer
