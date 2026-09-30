# CS5720: Neural Networks and Deep Learning - Home Assignment 3

### Student Information
- **Student Name:** Md Shakiful Islam Khan
- **Course:** CS5720 Neural Networks and Deep Learning (Fall 2026)
- **Assignment:** Home Assignment 3

---

## Repository Contents
This repository contains the implementations and analysis for **Part II (Programming Questions)** of Home Assignment 3.

### Part II - Question 1: Implement Convolution from Scratch

#### 1. Implementation Overview
A custom 2D convolution operation (`convolve2d`) was built from scratch utilizing Python and `NumPy` without relying on high-level deep learning library functions (like PyTorch or TensorFlow's convolution APIs). 

- **Input Matrix Size:** 5 × 5
- **Kernel/Filter Size:** 3 × 3
- **Padding:** 0
- **Stride Options Evaluated:** Stride = 1 and Stride = 2

#### 2. Spatial Dimension Math
The output size of a convolutional layer is determined by:
$$\text{Output Size} = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1$$
- For **Stride = 1**:
  $$\text{Output Size} = \left\lfloor \frac{5 - 3 + 0}{1} \right\rfloor + 1 = 3 \times 3$$
- For **Stride = 2**:
  $$\text{Output Size} = \left\lfloor \frac{5 - 3 + 0}{2} \right\rfloor + 1 = 2 \times 2$$

#### 3. Execution Results
- **Stride = 1 Output Feature Map:**
  ```
  [[4. 3. 4.]
   [2. 4. 3.]
   [2. 3. 4.]]
  ```
  *Shape:* `(3, 3)`

- **Stride = 2 Output Feature Map:**
  ```
  [[4. 4.]
   [2. 4.]]
  ```
  *Shape:* `(2, 2)`

#### 4. Stride Impact Analysis
Changing the stride from 1 to 2 increases the step size of the sliding filter across the input. As a result, the receptive fields overlap less, leading to a downsampled feature map. Spatially, this reduces the dimensions from $3 \times 3$ to $2 \times 2$, effectively compressing the representation and lowering down-stream computational complexity, but potentially discarding fine-grained spatial information.

---

### Part II - Question 2: Transfer Learning (Freeze vs. Fine-Tune)

#### 1. Experimental Setup
We trained and compared two different transfer learning paradigms using a pretrained **ResNet18** backbone on a 2-class subset of the CIFAR-10 dataset (Airplanes vs. Automobiles):

- **Experiment A (Frozen Feature Extractor):** All ResNet18 convolutional layer weights were frozen (`requires_grad = False`). Only the newly initialized linear classifier header (`fc`) was trained.
- **Experiment B (Fine-Tuned Block 4):** Standard convolutional base layers were initially frozen, but the final residual convolutional block (`layer4`) and the custom classifier layer (`fc`) were unfrozen (`requires_grad = True`) and updated simultaneously with differential learning rates.

#### 2. Quantitative Comparison Table

| Method | Trainable Parameters | Training Time | Validation Accuracy |
| :--- | :---: | :---: | :---: |
| **Frozen Feature Extractor** | 1,026 | 8.06 seconds | 93.00% |
| **Fine-Tuned Network (Layer 4 + FC)** | 8,394,754 | 8.11 seconds | 96.50% |

#### 3. In-Depth Discussion of Transfer Learning Paradigms

In our experiments, the **Fine-Tuned Network** achieved a higher validation accuracy (**96.50%**) compared to the **Frozen Feature Extractor** (**93.00%**). This performance gain is due to the unfreezing of the final deep convolutional block (`layer4`). While lower layers of a CNN capture general features like edges and textures, the deeper layers capture highly abstract, domain-specific semantic details. Fine-tuning these deeper layers allows the model to adapt its feature representation directly to our dataset's target classes (Airplanes and Automobiles).

Although the Fine-Tuned model required optimizing significantly more parameters (**8,394,754** compared to only **1,026** in the frozen case), the training times remained very close (approx. 8 seconds) due to GPU acceleration. Under larger dataset regimes, fine-tuning generally demands higher computational resources and backpropagation overhead. Freezing layers is computationally efficient and prevents overfitting on small datasets, but fine-tuning selectively yields superior accuracy by refining high-level decision boundaries.
