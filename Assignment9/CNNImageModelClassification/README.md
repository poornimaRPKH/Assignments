# CNN Model for Flower Image Classification 🌸

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to automatically
classify flower images into one of five categories, for a horticulture company.

## Problem Statement

Build and evaluate a CNN that classifies flower images into: **daisy, dandelion,
roses, sunflowers, tulips**.

## Dataset

- **Source:** [TensorFlow Flower Photos dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
- **Size:** 3,670 images across 5 classes
- Downloaded automatically inside the notebook using `tf.keras.utils.get_file(...)` — no manual download needed.

## Project Files

| File | Description |
|---|---|
| `Assignment_9_CNNModelforImageClassification.ipynb` | Complete notebook: data loading, CNN model, training, evaluation, analysis |
| `trained_model.h5` | Saved trained CNN model |
| `confusion_matrix.png` | Confusion matrix on the validation set |
| `training_curves.png` | Training vs. validation accuracy/loss curves |
| `requirements.txt` | Python dependencies |

## Methodology

1. **Data Preparation:** Images resized to 100×100, normalized to [0,1], split 80% train / 20% validation.
2. **CNN Architecture:** Rescaling → 2×(Conv2D + ReLU + MaxPooling2D) → Flatten → Dense(64, ReLU) → Dense(5, softmax).
3. **Training:** Optimizer: Adam · Loss: Sparse Categorical Crossentropy · 5 epochs.
4. **Evaluation:** Accuracy, loss, confusion matrix, sample predictions.
5. **Analysis:** Training/validation curves reviewed for overfitting, with improvement suggestions.

## Results

- **Final training accuracy:** 82.4%
- **Final validation accuracy:** 64.6%
- **Final validation loss:** 0.97
- **Classes most confused:** Tulips were frequently misclassified as Roses (50 out of 202 tulip images), likely due to similar petal shapes and color palettes.
- Dandelion and Sunflower were the most accurately classified (127 correct each), thanks to their distinctive colors/shapes.

See `training_curves.png` for the accuracy/loss plots and `confusion_matrix.png` for the full breakdown.

## Key Findings & Improvement Suggestions

The gap between training accuracy (82%) and validation accuracy (65%) shows the model is **overfitting** — memorizing training images rather than learning fully general flower features.

**Recommended next steps:**
1. Use **transfer learning** (e.g. MobileNetV2 pretrained on ImageNet) instead of training from scratch.
2. Add **data augmentation** (random flips, rotations, zooms) to reduce overfitting.
3. Add **Dropout** layers to force more general learning.
4. Use **EarlyStopping** to stop training at the best point.
5. Collect **more training images**, especially for tulips and roses, to help the model tell them apart.

## Tech Stack

Python · TensorFlow/Keras · NumPy · Matplotlib · Seaborn · scikit-learn

## How to Run

1. Open the notebook in Google Colab (Runtime → GPU recommended).
2. Run all cells top to bottom — the dataset downloads automatically.
3. The trained model and output plots will be generated in the working directory.

