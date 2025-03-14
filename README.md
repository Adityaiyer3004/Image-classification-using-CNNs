Deeper Networks for Image Classification
========================================
Author: Aditya Iyer

Project Overview
----------------
This repository evaluates state-of-the-art Convolutional Neural Network (CNN) architectures—VGG13, VGG16, ResNet-18, ResNet-50, GoogLeNet (Inception V1), and Inception V3—on MNIST and CIFAR-10 datasets. The goal is to assess their robustness, accuracy, and efficiency for image classification, providing insights valuable for real-world applications, such as autonomous driving and medical diagnostics.

CNN Models and Architectures
----------------------------
- GoogLeNet (Inception V1): 22-layer network using parallel convolutional filters (Inception modules), dimensionality reduction (1x1 convolutions), and global average pooling.
- Inception V3: Improved GoogLeNet architecture with factorized convolutions (3x3), asymmetric convolutions (1x7, 7x1), and optimized pooling layers.
- ResNet-18: 18-layer residual network using skip connections to mitigate the vanishing gradient problem.
- ResNet-50: Enhanced residual network with 50 layers using bottleneck residual blocks to maintain efficiency.
- VGG13: Simple yet effective CNN architecture comprising 13 convolutional layers with uniform 3x3 filters.
- VGG16: Expanded version of VGG13 with 16 convolutional layers, known for consistent and robust performance.

Datasets Evaluated
------------------
1. MNIST:
   - 70,000 grayscale images (28x28 pixels)
   - 10 classes of handwritten digits (0-9)
2. CIFAR-10:
   - 60,000 color images (32x32 pixels)
   - 10 diverse classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

Key Observations & Insights
---------------------------
- MNIST: All models quickly achieved approximately 98% accuracy due to dataset simplicity. CNN architectures intended for more complex datasets demonstrated rapid convergence.
- CIFAR-10: Accuracy around 80% was achievable; however, this required careful optimizer selection and early stopping to avoid overfitting. VGG models performed best with the SGD optimizer, whereas other models benefitted from Adam optimizer.

Recommended Training Setup
--------------------------
**Important:**  
It is highly advised to train these CNN architectures using Google's free GPU resources available through Google Colab to significantly reduce training times.

Google Colab Link: https://colab.research.google.com/

---

## 📁 Repository Structure

```bash
Deeper-Networks-Image-Classification/
├── notebooks/
│   ├── MNIST/
│   │   ├── VGG13_MNIST.ipynb
│   │   ├── VGG16_MNIST.ipynb
│   │   ├── ResNet18_MNIST.ipynb
│   │   ├── ResNet50_MNIST.ipynb
│   │   ├── GoogLeNet_MNIST.ipynb
│   │   └── InceptionV3_MNIST.ipynb
│   │
│   ├── CIFAR10/
│   │   ├── VGG13_CIFAR10.ipynb
│   │   ├── VGG16_CIFAR10.ipynb
│   │   ├── ResNet18_CIFAR10.ipynb
│   │   ├── ResNet50_CIFAR10.ipynb
│   │   ├── GoogLeNet_CIFAR10.ipynb
│   │   └── InceptionV3_CIFAR10.ipynb
│   │
│   └── Analysis_and_Comparison.ipynb *(Planned)*
│
├── report/
│   └── [Image_Classification_CNN_Report.pdf](report/Image_Classification_CNN_Report.pdf)
│
├── requirements.txt
└── README.md
Setup and Dependencies
----------------------
To run the notebooks, install the following dependencies:

torch
torchvision
numpy
matplotlib
pandas
pytorch-lightning
torchmetrics
scikit-learn

Install via:
pip install -r requirements.txt

Future Directions
-----------------
- Testing models on complex real-world datasets (e.g., ImageNet, CIFAR-100).
- Hyperparameter optimization.
- Transfer learning approaches for applications in autonomous driving and medical diagnostics.

Contact Information
-------------------
Aditya Iyer
- GitHub: https://github.com/<your-username>
- LinkedIn: https://linkedin.com/in/<your-profile>

License
-------
This project is licensed under the MIT License.





