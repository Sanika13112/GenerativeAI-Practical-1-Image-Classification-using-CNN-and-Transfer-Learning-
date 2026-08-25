


# Tomato Leaf Disease Classification using CNN & Transfer Learning — Generative AI Lab

**MIT Academy of Engineering, Alandi, Pune**  
**Department of CSE (AIML)** · **Course:** Generative AI Lab · **Class:** T.Y. Tech

---

## Student Information

- **Name:** Sanika Dhanaji Mane
- **PRN Number:** 202401110047
- **Batch:** A3
- **Date of Submission:** 21th August 2026

---

## Objective

Implement an image classification pipeline using a Convolutional Neural Network (CNN).

A Custom CNN is trained from scratch and compared with three pre-trained transfer learning models:

1. MobileNetV2
2. ResNet50
3. EfficientNetB0

All four models are trained and evaluated on the same tomato leaf disease dataset.

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Number of parameters
- Training and validation curves
- Confusion matrices

The main objective is to determine whether transfer learning provides better classification performance than training a CNN completely from scratch and to study the trade-off between model performance and computational complexity.

---

# Selected Research Paper

- **Title:** *Comparing pre-trained models for efficient leaf disease detection: a study on custom CNN*
- **Authors:** Touhidul Seyam Alam, Chandni Barua Jowthi & Abhijit Pathak
- **Journal:** *Journal of Electrical Systems and Information Technology*
- **Year:** 2024
- **Volume:** 11
- **Article Number:** 12
- **DOI:** 10.1186/s43067-024-00137-1
- **Publisher:** Springer Nature
- **Research Paper:** https://link.springer.com/article/10.1186/s43067-024-00137-1

### Research Paper Relevance

The selected research paper presents a comparative study between a custom CNN trained from scratch and nine pre-trained CNN models for leaf disease detection.

The paper evaluates pretrained architectures including:

- DenseNet201
- EfficientNetB3
- EfficientNetB4
- InceptionResNetV2
- MobileNetV2
- ResNet50
- ResNet152
- VGG16
- Xception

The models are evaluated using classification metrics such as accuracy, precision, recall and F1-score, along with computational characteristics such as parameter count, training time and memory requirements.

The research is highly relevant to this practical because it investigates leaf disease detection using the PlantVillage dataset and compares custom CNN architectures with pretrained models.

Following a similar comparative methodology, this practical evaluates:

1. Custom CNN
2. MobileNetV2
3. ResNet50
4. EfficientNetB0

The main difference is that this practical uses its own experimental configuration, preprocessing pipeline, dataset split and training settings.

Therefore, the results obtained in this practical are independent experimental results and are not claimed to reproduce the exact results reported in the research paper.

---

# Dataset Details

- **Dataset Name:** PlantVillage Tomato Leaf Disease Dataset
- **Dataset Source:** PlantVillage
- **Number of Classes:** 10
- **Total Images:** 18,160
- **Image Size:** 224 × 224 pixels
- **Image Type:** RGB

The dataset contains tomato leaf images representing healthy leaves and leaves affected by different diseases.

---

# Dataset Classes

The project uses the following 10 tomato leaf classes:

1. Tomato___Bacterial_spot
2. Tomato___Early_blight
3. Tomato___Late_blight
4. Tomato___Leaf_Mold
5. Tomato___Septoria_leaf_spot
6. Tomato___Spider_mites Two-spotted_spider_mite
7. Tomato___Target_Spot
8. Tomato___Tomato_Yellow_Leaf_Curl_Virus
9. Tomato___Tomato_mosaic_virus
10. Tomato___healthy

---

# Dataset Distribution

| Class | Number of Images |
| :--- | ---: |
| Tomato___Bacterial_spot | 2,127 |
| Tomato___Early_blight | 1,000 |
| Tomato___Late_blight | 1,909 |
| Tomato___Leaf_Mold | 952 |
| Tomato___Septoria_leaf_spot | 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 |
| Tomato___Target_Spot | 1,404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 |
| Tomato___Tomato_mosaic_virus | 373 |
| Tomato___healthy | 1,591 |
| **Total** | **18,160** |

The dataset is imbalanced because the number of images varies significantly between classes. Class weights were therefore used during model training.

---

# Dataset Preprocessing

The following preprocessing operations were performed:

- Images resized to 224 × 224 pixels
- RGB image format used
- Training, validation and testing datasets created
- Data augmentation applied to training images
- Model-specific preprocessing applied to pretrained architectures
- Class weights used to handle class imbalance
- Images loaded using TensorFlow data pipelines

### Data Augmentation

Data augmentation was applied to the training data to increase image variation and improve model generalization.

The augmentation pipeline includes transformations such as:

- Random horizontal flipping
- Random rotation
- Random zoom
- Other training-time image transformations

---

# Model Architectures

## 1. Custom CNN — From Scratch

The Custom CNN is trained completely from scratch without using pre-trained ImageNet weights.

### Architecture

```text
Input Image
     ↓
224 × 224 × 3
     ↓
Data Augmentation
     ↓
Rescaling
     ↓
Conv2D — 32 filters
     ↓
MaxPooling
     ↓
Conv2D — 64 filters
     ↓
MaxPooling
     ↓
Conv2D — 128 filters
     ↓
MaxPooling
     ↓
Conv2D — 256 filters
     ↓
MaxPooling
     ↓
Global Average Pooling
     ↓
Dense — 128 neurons
     ↓
Dropout
     ↓
Softmax
     ↓
10 Classes
````

The model learns visual features directly from the tomato leaf images.

---

## 2. MobileNetV2 — Transfer Learning

MobileNetV2 is a lightweight convolutional neural network pre-trained on ImageNet.

### Why MobileNetV2?

* Lightweight architecture
* Computationally efficient
* Strong pretrained visual features
* Suitable for transfer learning
* Lower complexity compared with many deeper architectures

### Training Strategy

The model was trained in two stages:

1. Feature extraction using the frozen pretrained base.
2. Fine-tuning of deeper layers using a low learning rate.

---

## 3. ResNet50 — Transfer Learning

ResNet50 is a deep convolutional neural network based on residual connections and pre-trained on ImageNet.

### Why ResNet50?

* Strong deep feature extraction
* Residual connections
* Effective training of deep networks
* Strong image classification capability
* Widely used transfer learning architecture

### Training Strategy

The model was trained in two stages:

1. Feature extraction using the frozen pretrained base.
2. Fine-tuning of deeper layers using a low learning rate.

---

## 4. EfficientNetB0 — Transfer Learning

EfficientNetB0 is a pretrained CNN designed to provide an effective balance between classification performance and computational complexity.

### Why EfficientNetB0?

* Efficient architecture
* Good parameter utilization
* Strong pretrained features
* Suitable for image classification
* Good balance between performance and model complexity

### Training Strategy

The model was trained in two stages:

1. Feature extraction using the frozen pretrained base.
2. Fine-tuning of deeper layers using a low learning rate.

---

# Training Methodology

The same dataset and evaluation procedure were used for all four models to provide a controlled comparative experiment.

```text
                 PlantVillage Tomato Dataset
                           |
                           ↓
                  Data Preprocessing
                           |
                           ↓
                   Data Augmentation
                           |
             ┌─────────────┴─────────────┐
             ↓                           ↓
        Custom CNN                 Transfer Learning
       From Scratch              /       |       \
             |                  /        |        \
             |          MobileNetV2   ResNet50   EfficientNetB0
             |                  \        |        /
             |                   \       |       /
             └────────────────────┴───────┴──────┘
                           |
                           ↓
                      Test Dataset
                           |
                           ↓
        Accuracy / Precision / Recall / F1-score
                           |
                           ↓
            Training Time / Model Parameters
                           |
                           ↓
                 Comparative Evaluation
```

---

# Training Configuration

* **Input Size:** 224 × 224 × 3
* **Number of Classes:** 10
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Cross-Entropy
* **Activation Function:** ReLU
* **Output Activation:** Softmax
* **Data Augmentation:** Applied to training data
* **Class Weighting:** Applied
* **Early Stopping:** Used
* **Learning Rate Reduction:** ReduceLROnPlateau
* **Transfer Learning:** ImageNet pretrained weights
* **Fine-Tuning:** Applied to deeper layers of pretrained models

---

# Evaluation Metrics

## Accuracy

Accuracy represents the proportion of correctly classified images among all evaluated images.

## Precision

Precision measures how many of the images predicted as a particular class actually belong to that class.

## Recall

Recall measures how many images belonging to a particular class were correctly identified by the model.

## F1-Score

F1-score provides a balanced measure of precision and recall.

## Training Time

Training time measures the computational time required to train each model.

## Number of Parameters

The number of parameters provides an indication of model size and complexity.

---

# Results and Comparison

The final models are evaluated on the same unseen test dataset.

| Model              | Accuracy | Precision | Recall | F1-Score | Training Time | Total Parameters |
| :----------------- | :------: | :-------: | :----: | :------: | :-----------: | ---------------: |
| **Custom CNN**     |    TBD   |    TBD    |   TBD  |    TBD   |      TBD      |              TBD |
| **MobileNetV2**    |    TBD   |    TBD    |   TBD  |    TBD   |      TBD      |              TBD |
| **ResNet50**       |    TBD   |    TBD    |   TBD  |    TBD   |      TBD      |              TBD |
| **EfficientNetB0** |    TBD   |    TBD    |   TBD  |    TBD   |      TBD      |              TBD |

> The final values should be updated using the final comparison table generated by the notebook.

---

# Custom CNN Training Result

The Custom CNN achieved the following validation performance during training:

* **Best Validation Accuracy:** 94.05%
* **Best Epoch:** 18

Early stopping was used during training and the model weights from the best validation epoch were restored.

The 94.05% value represents validation accuracy and is therefore not used as the final test accuracy in the comparative study.

The final comparison is based on test-set performance.

---

# Training and Validation Curves

Training and validation accuracy and loss curves were generated for the models.

The curves are used to analyze:

* Learning behavior
* Convergence
* Generalization
* Overfitting
* Training stability

---

# Confusion Matrix

A confusion matrix is generated for each model using the test dataset.

It provides class-wise information about:

* Correct predictions
* Incorrect predictions
* Disease classes that are difficult to distinguish
* Overall classification behavior

---

# Comparative Study

The Custom CNN and pretrained models are compared based on both classification performance and computational complexity.

## Custom CNN

### Advantages

* Simple architecture
* Easy to understand
* Trained specifically for the target dataset
* No dependency on pretrained weights
* Can provide lower model complexity

### Limitations

* Learns all features from scratch
* May require more training
* More sensitive to dataset size
* Architecture may require additional tuning

---

## Transfer Learning Models

### Advantages

* Use features learned from a large-scale dataset
* Faster feature adaptation
* Strong classification performance
* Useful for image datasets
* Fine-tuning allows adaptation to the target problem

### Limitations

* Some pretrained models have high parameter counts
* Fine-tuning requires careful learning-rate selection
* Larger models can require more computational resources
* Training time can increase with model complexity

---

# Research Paper Comparison

The selected Springer research paper compares a custom CNN with nine pretrained architectures for leaf disease detection.

The paper evaluates models using classification performance and computational characteristics such as accuracy, precision, recall, F1-score, parameter count, training time and memory requirements.

The research paper reported very high classification performance for several pretrained models, while its proposed custom LDDTA CNN provided a strong efficiency advantage with substantially fewer parameters.

The following values are reported by the research paper and are included only as reference values:

| Model             | F1-Score | Trainable Parameters |
| :---------------- | -------: | -------------------: |
| DenseNet201       |    0.997 |           18,824,010 |
| EfficientNetB3    |    0.998 |           11,185,721 |
| InceptionResNetV2 |    0.998 |           54,738,922 |
| MobileNetV2       |    0.998 |            2,556,938 |
| ResNet50          |    0.998 |           24,123,018 |
| EfficientNetB4    |    0.999 |           18,142,569 |
| ResNet152         |    0.999 |           58,906,250 |
| Xception          |    0.999 |           21,396,786 |
| VGG16             |    0.981 |           14,850,634 |
| Proposed LDDTA    |    0.975 |              184,890 |

**Note:** These values are from the selected research paper and are not the results obtained in this practical.

---

# Research Paper vs Our Experiment

| Aspect            | Research Paper                                    | Our Practical                                     |
| :---------------- | :------------------------------------------------ | :------------------------------------------------ |
| Dataset           | PlantVillage tomato leaf images                   | PlantVillage tomato leaf images                   |
| Custom Model      | LDDTA Custom CNN                                  | Custom CNN                                        |
| Pretrained Models | 9 architectures                                   | 3 architectures                                   |
| MobileNetV2       | Yes                                               | Yes                                               |
| ResNet50          | Yes                                               | Yes                                               |
| EfficientNet      | B3/B4                                             | B0                                                |
| Evaluation        | Accuracy, Precision, Recall, F1, Parameters, Time | Accuracy, Precision, Recall, F1, Parameters, Time |
| Fine-Tuning       | Used                                              | Used                                              |
| Main Goal         | Compare performance and efficiency                | Compare CNN vs Transfer Learning                  |

The research paper provides the methodological motivation for this comparative study, while the results in this project are independently obtained using the project's own experimental setup.

---

# Best Model

The final best model is selected using:

1. Test accuracy
2. F1-score
3. Precision
4. Recall
5. Training time
6. Number of parameters
7. Overall computational efficiency

The model with the strongest overall test performance will be considered the best model for this experiment.

The final selection is based on experimentally obtained results rather than assuming that a particular architecture will always perform best.

---

# Files in Repository

```text
Tomato-Leaf-Disease-Classification/
│
├── README.md
│
├── Sanika_Mane_PracticalAssignment1.ipynb
│
├── results/
│   ├── custom_cnn_accuracy.png
│   ├── custom_cnn_loss.png
│   ├── mobilenetv2_accuracy.png
│   ├── mobilenetv2_loss.png
│   ├── resnet50_accuracy.png
│   ├── resnet50_loss.png
│   ├── efficientnetb0_accuracy.png
│   ├── efficientnetb0_loss.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_mobilenetv2.png
│   ├── confusion_matrix_resnet50.png
│   ├── confusion_matrix_efficientnetb0.png
│   └── model_comparison.png
│
└── report/
    └── Practical_Assignment_1.pdf
```

The complete dataset is not included in the GitHub repository because of its large size.

---

# How to Run

## Google Colab

1. Open the notebook in Google Colab.
2. Select an available GPU runtime if available.
3. Connect or upload the PlantVillage dataset.
4. Set the dataset path.
5. Run the notebook cells sequentially.
6. Generate the evaluation and comparison results.

## VS Code / Jupyter Notebook

1. Install Python.
2. Install Jupyter Notebook or use VS Code with the Jupyter extension.
3. Install the required libraries.
4. Place the dataset in the local system.
5. Update the dataset path.
6. Run the notebook cells sequentially.

### Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

# Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab
* Jupyter Notebook
* VS Code

---

# Future Scope

The project can be further extended by:

* Testing additional pretrained architectures
* Using real-world field images
* Applying Grad-CAM for explainable predictions
* Improving minority-class performance
* Developing a web-based disease detection application
* Developing a mobile application
* Using model compression and quantization
* Deploying the best-performing model on edge devices

---

# Conclusion

This project demonstrates the application of deep learning for automated tomato leaf disease classification.

A Custom CNN trained from scratch was compared with three ImageNet pretrained transfer learning models: MobileNetV2, ResNet50 and EfficientNetB0.

The experiment evaluates both predictive performance and computational complexity using accuracy, precision, recall, F1-score, training time and parameter count.

The selected Springer research paper demonstrates the importance of comparing custom CNN architectures with pretrained models while considering both classification performance and computational efficiency.

Similarly, this practical investigates the trade-off between learning features from scratch and reusing pretrained features.

The final model selection is based on the experimentally obtained test results and considers both classification performance and computational efficiency.

---

# Declaration

I, **Sanika Dhanaji Mane**, confirm that the work submitted in this assignment is my own work and has been completed in accordance with academic integrity guidelines.

The project implementation, experiments, analysis, graphs and results presented in this repository were developed as part of the Generative AI Lab Practical Assignment.

The selected research paper has been used as a methodological reference for the comparative study, and the experimental results reported in this repository are obtained from our own implementation and dataset configuration.

The code and project documentation are uploaded to my GitHub repository.

### GitHub Repository : https://github.com/Sanika13112/GenerativeAI-Practical-1-Image-Classification-using-CNN-and-Transfer-Learning-



---

# Reference

Alam, T. S., Jowthi, C. B., & Pathak, A. (2024).

*Comparing pre-trained models for efficient leaf disease detection: a study on custom CNN.*

Journal of Electrical Systems and Information Technology, 11, 12.

DOI: 10.1186/s43067-024-00137-1

Springer Nature:

[https://link.springer.com/article/10.1186/s43067-024-00137-1](https://link.springer.com/article/10.1186/s43067-024-00137-1)

---

# Acknowledgement

I would like to express my sincere gratitude to the Department of Computer Science and Engineering (AIML), MIT Academy of Engineering, Alandi, Pune, for providing the opportunity and resources to carry out this practical assignment.

I also acknowledge the PlantVillage dataset and the selected research paper for providing the dataset and methodological reference used in this project.

---

## Author

**Sanika Dhanaji Mane**

T.Y. Tech — CSE (AIML)
MIT Academy of Engineering, Alandi, Pune

**Generative AI Lab — Practical Assignment 1**

