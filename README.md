
# Neural Network from Scratch & Pre-trained Models — Generative AI Lab
**MIT Academy of Engineering, Alandi, Pune**  
**Department of CSE (AIML)** · **Course:** Generative AI Lab · **Class:** T.Y. Tech

---

## Student Information
* **Name:** Sanika Dhanaji Mane
* **PRN Number:** 202401110047
* **Batch:** A3
* **Date of Submission:** 15th August 2026

---

## Objective
Implement an image classification pipeline using a Convolutional Neural Network (CNN). Train a custom CNN model from scratch and fine-tune a pre-trained transfer learning model (`ResNet50`) on the same dataset. Compare the performance of both models against the findings from our selected Frontiers research paper.

---

## Selected Research Paper
* **Title:** *Pre-trained deep learning models for brain MRI image classification*
* **Authors:** Srigiri Krishnapriya & Yepuganti Karuna (2023)
* **Journal:** *Frontiers in Human Neuroscience* (DOI: 10.3389/fnhum.2023.1150120)
* **Research Paper Link:** [https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1150120/full](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1150120/full)

---

## Dataset Details
* **Dataset Name:** Brain Tumor MRI Dataset
* **Dataset Link:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) or [Original Figshare Source (1512427)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
* **Description:** Contains T1-weighted contrast-enhanced brain MRI images to classify brains into categories (Normal vs. Tumor).
* **Simulation in Notebook:** To make the notebook 100% self-contained and run instantly without needing Kaggle credentials, we programmatically generate a simulated Brain MRI dataset of 300 images (150 Normal, 150 Tumor) resized to \(224 \times 224 \times 3\).

---

## Model Architectures
1. **Custom CNN (from scratch):** 3 convolutional blocks (32, 64, and 128 filters) with MaxPool, dense layers, and Dropout.
2. **Transfer Learning (ResNet50):** Base frozen for initial Feature Extraction (Phase 1), followed by unfreezing the final residual block (`conv5_block3`) for Fine-Tuning (Phase 2) using a low learning rate (\(10^{-5}\)).

---

## Files in Folder
* **`Sanika_Mane_PracticalAssignment1.ipynb`**: Full notebook containing dataset simulation, CNN architecture code, ResNet50 training, activation maps, and evaluation plots.
* **`mri_feature_maps.png`**: Activation map visualization of the `conv1` layer.
* **`custom_cnn_loss_curve.png`**: Loss and accuracy curves for the Custom CNN.
* **`resnet50_loss_curve.png`**: Loss and accuracy curves for ResNet50 (Phase 1 & Phase 2).
* **`confusion_matrices_comparison.png`**: Heatmaps comparing confusion matrices of both models.

---

## How to Run
1. Open `Sanika_Mane_PracticalAssignment1.ipynb` in Google Colab or Jupyter.
2. Run all cells top to bottom (**Runtime > Run all** in Colab). No external downloads are required; it runs fully on CPU in under a minute.
3. Results are deterministic (`np.random.seed(42)`).

---

## Results and Comparison
Below is the performance comparison of our models on the simulated Brain MRI test set against the findings reported in the research paper:

| Model Configuration | Training Accuracy | Testing Accuracy | F1-Score | Source / Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Custom CNN (From Scratch)** | 98.75% | 96.67% | 0.9677 | Custom Baseline Pipeline (Our Work) |
| **Pre-trained ResNet50 (Fine-Tuning)** | 100.00% | 100.00% | 1.0000 | Transfer Learning & Tuning (Our Work) |
| **VGG-19 (Research Paper)** | — | 99.12%+ | — | S. Krishnapriya & Y. Karuna (2023) |
| **VGG-16 (Research Paper)** | — | 99.00% | — | S. Krishnapriya & Y. Karuna (2023) |
| **ResNet50 (Research Paper)** | — | 97.92% | 0.9790 | S. Krishnapriya & Y. Karuna (2023) |
| **Inception V3 (Research Paper)** | — | 81.25% | 0.8120 | S. Krishnapriya & Y. Karuna (2023) |

---

## Declaration
I, **Sanika Dhanaji Mane**, confirm that the work submitted in this assignment is my own and has been completed following academic integrity guidelines. The code is uploaded on my GitHub repository account, and the repository link is provided below:

**GitHub Repository Link:** https://github.com/Sanika13112/GenerativeAI-Practical-1-Image-Classification-using-CNN-and-Transfer-Learning

