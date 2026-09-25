# AI-Based Handwritten Digit Generation Using GANs

**Module End Project: Generative Deep Learning Applications**

## Problem Statement
This project builds a **Generative Adversarial Network (GAN)** using TensorFlow/Keras to generate new handwritten digit images that resemble the **MNIST** dataset. It demonstrates how a Generator and a Discriminator compete during adversarial training, and how to evaluate the quality of generated images over time.

## Objectives
- Understand the basic concept of Generative AI and GANs
- Preprocess the MNIST image dataset
- Build Generator and Discriminator networks
- Train a GAN using adversarial learning
- Generate new handwritten digit images
- Analyze GAN training using generated image samples and loss curves

## Dataset
**MNIST**, loaded directly via TensorFlow/Keras:
```python
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
```
- 60,000 grayscale training images of handwritten digits (0–9)
- Original size: 28×28 pixels, single channel
- Only the images (not labels) are used, since GAN training is unsupervised

## Preprocessing
- Pixel values normalized from `[0, 255]` to `[-1, 1]` (to match the Generator's `tanh` output)
- Images reshaped to `(28, 28, 1)` to add an explicit channel dimension

## Model Architecture

### Generator
`Dense → BatchNormalization → LeakyReLU → Conv2DTranspose → Conv2DTranspose → Output (Tanh)`
- Input: 100-dimensional random noise vector
- Output: 28×28×1 grayscale image

### Discriminator
`Conv2D → LeakyReLU → Dropout → Conv2D → LeakyReLU → Dropout → Flatten → Dense (Sigmoid)`
- Input: 28×28×1 image (real or generated)
- Output: probability that the image is real

## Training Configuration
| Setting | Value |
|---|---|
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Beta_1 | 0.5 |
| Batch Size | 256 |
| Noise Vector Size | 100 |
| Epochs | 20 (minimum) |

## Repository Structure
```
├── GAN_MNIST_Digit_Generation.ipynb   # Main notebook: full code + explanations
├── gan_samples/                       # Generated image grids + loss curve plot (created on run)
└── README.md                          # This file
```

## How to Run
1. Open `GAN_MNIST_Digit_Generation.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. (Recommended) Use a GPU runtime (e.g. Colab: *Runtime → Change runtime type → GPU*) for reasonable training speed.
3. Run all cells in order. The notebook will:
   - Load and preprocess MNIST
   - Build the Generator and Discriminator
   - Train for 20 epochs, showing a sample image grid every 5 epochs
   - Plot the Generator/Discriminator loss curves
   - Display a final grid of generated digits

## Requirements
```
tensorflow>=2.x
numpy
matplotlib
```

## Results Summary
- Sample generated digit grids are saved every few epochs to `gan_samples/`.
- The Generator/Discriminator loss curve (`gan_samples/loss_curves.png`) shows how both networks' losses evolve, used to judge training stability.
- A final 5×5 grid of freshly generated digits (`gan_samples/final_generated_grid.png`) shows overall output quality after training.

## Key Learnings
- How adversarial training works: two networks improving by competing against each other.
- The importance of matching data normalization (`[-1, 1]`) to the Generator's output activation (`tanh`).
- How to read GAN loss curves to judge whether training is stable, or whether one network is dominating the other.

## Possible Extensions
- Train for more epochs / on more data for sharper digits.
- Add label conditioning (Conditional GAN) to generate a *specific* requested digit.
- Try a DCGAN with more layers, or a WGAN loss for improved training stability.

