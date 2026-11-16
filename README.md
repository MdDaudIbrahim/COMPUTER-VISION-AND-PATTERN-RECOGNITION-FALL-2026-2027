
# COMPUTER-VISION-AND-PATTERN-RECOGNITION-FALL-2026-2027

# Fast Food Classification Using Custom CNN (PyTorch)

### Computer Vision and Pattern Recognition (CVPR) - Mid Assignment

**American International University-Bangladesh (AIUB)**

---

## Project Overview

This project focuses on designing, training, and evaluating a custom Convolutional Neural Network (CNN) using **PyTorch** to classify images into 10 different fast-food categories.

- **Student Name:** Md. Daud Ibrahim
- **Student ID:** 22-49760-3
- **Course:** Computer Vision and Pattern Recognition (CVPR)
- **Submission Notebook:** [CNN_22-49760-3.ipynb](CNN_22-49760-3.ipynb)
- **Model Weights:** [model.pth](model.pth) (Approximately 5.51 MB)

---

## Dataset Information

- **Dataset:** [Fast Food Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset)
- **Number of Classes:** 10
- **Total Images:** 20,000

**Food Categories:**
Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, Pizza, Sandwich, Taco, and Taquito.

**Dataset Split:**
- Training: 15,000 images
- Validation: 3,500 images
- Testing: 1,500 images

---

## Custom CNN Architecture

The CNN was built from scratch using four convolutional blocks followed by a fully connected classification head.

```text
Input: (3 × 128 × 128)

├── Block 1: Conv2D (3 → 32) + BatchNorm + ReLU + MaxPool
│   Output: (32 × 64 × 64)
│
├── Block 2: Conv2D (32 → 64) + BatchNorm + ReLU + MaxPool
│   Output: (64 × 32 × 32)
│
├── Block 3: Conv2D (64 → 128) + BatchNorm + ReLU + MaxPool
│   Output: (128 × 16 × 16)
│
├── Block 4: Conv2D (128 → 256) + BatchNorm + ReLU + MaxPool
│   Output: (256 × 8 × 8)
│
├── AdaptiveAvgPool2d((4, 4))
│
└── Classifier:
    ├── Flatten (4096)
    ├── Dropout (0.4)
    ├── Linear (4096 → 256) + ReLU
    ├── Dropout (0.3)
    └── Linear (256 → 10)
```

- **Total Trainable Parameters:** 1,440,778 (~1.44 Million)
- **Model Weight Size:** Approximately 5.51 MB

---

## Hyperparameters & Training Configuration

- **Input Resolution:** 128 × 128
- **Batch Size:** 32
- **Epochs:** 25
- **Optimizer:** Adam
- **Initial Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Loss Function:** CrossEntropyLoss with calculated class weights. All weights were 1.0 due to equal class representation in the training dataset.
- **LR Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=2)
- **Dropout:** 0.4 and 0.3
- **Checkpointing:** Saves the model with the highest validation accuracy.

---

## Experimental Results

### Model Performance

- **Training Accuracy:** 76.16%
- **Best Validation Accuracy:** 73.37% (Epoch 24)
- **Test Accuracy:** 72.13%
- **Test Loss:** 0.90

### Performance Comparison

The initial experiment using 64 × 64 images achieved 60.93% test accuracy. In the improved experiment, the image resolution was increased to 128 × 128 and additional training techniques were applied.

The test accuracy improved by **11.20 percentage points**, reaching 72.13%.

### Best Performing Classes

| Food Category | F1-Score |
|---|---:|
| Donut | 0.87 |
| Pizza | 0.85 |
| Crispy Chicken | 0.83 |
| Fries | 0.81 |
| Burger | 0.77 |

The model performed less effectively on visually similar categories, particularly Taco and Taquito.

---

## Repository Contents

1. **`CNN_22-49760-3.ipynb`**  
   Complete Jupyter Notebook containing dataset preparation, preprocessing, CNN architecture, training, validation, test evaluation, visualizations, classification report, confusion matrix, and discussion.

2. **`model.pth`**  
   Saved CNN model weights (approximately 5.51 MB).

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

## Conclusion

This project demonstrates the use of a custom CNN for multi-class fast-food image classification. The model achieved 72.13% test accuracy while maintaining a compact size of approximately 5.51 MB.

Future improvements could explore transfer learning, additional training data, and more advanced CNN architectures.
