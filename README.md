# Brain Tumor Detection using CNN

## About This Project
This project aims to detect brain tumors from MRI images using a Convolutional Neural Network (CNN). It provides a web-based interface for users to upload MRI scans and receive predictions. The model is trained using TensorFlow and Keras, and the web application is built with Flask.

## Project Workflow
The project workflow consists of the following steps:

1. **Data Collection and Preprocessing:**
   - MRI images are collected and categorized into two classes: `Tumor` and `No Tumor`.
   - Images are resized to 64x64 pixels.
   - Normalization is applied to scale pixel values between 0 and 1.
   - Data is split into training and testing sets.

2. **Model Training:**
   - A CNN model is built with the following layers:
     - Convolutional layers to extract features.
     - MaxPooling layers to reduce dimensionality.
     - Dropout layers to prevent overfitting.
     - Dense layers for classification.
   - The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
   - Training is conducted with the prepared dataset, and performance metrics are recorded.

3. **Model Evaluation:**
   - The trained model is evaluated on test data using accuracy, precision, recall, and F1-score.
   - A confusion matrix is generated to analyze misclassifications.

4. **Deployment:**
   - The trained model is saved as `BrainTumorCategorical.h5`.
   - Flask is used to build a web application where users can upload MRI images.
   - Predictions are displayed with a user-friendly interface.

5. **Testing:**
   - The web app is tested by uploading sample images to validate performance.
   - Various test cases are considered, such as handling invalid inputs and checking response times.

## Model Flow Diagram
Below is the visual representation of the complete flow from loading MRI scan data to final deployment:

```
Load MRI Scan Data ➜ Preprocessing (Resizing & Normalization) ➜ CNN Layers (Convolution, Activation) ➜ Filters (Feature Extraction) ➜ Pooling (Dimensionality Reduction) ➜ Epochs & Training ➜ Evaluation (Accuracy, Loss) ➜ Prediction & Deployment
```

## Concepts Learned in This Project

### 1. Convolutional Neural Networks (CNN)
**What is CNN?**
- CNN is a deep learning algorithm designed for processing structured grid data such as images.
- It consists of convolutional layers that automatically learn spatial hierarchies of features.

**Why CNN and not RNN?**
- CNN is better suited for image processing tasks, whereas RNNs are effective for sequential data like time series and text.
- CNN captures spatial dependencies, while RNNs are designed for temporal dependencies.

### 2. Filters in CNN
**What is a Filter?**
- A filter (or kernel) is a matrix applied to an image to detect specific patterns such as edges, textures, or colors.

**Why Use Filters?**
- Filters help reduce the complexity of the image by highlighting important features.

**Why Here?**
- In this project, filters are used in convolutional layers to extract relevant patterns from MRI images to distinguish tumors from healthy tissues.

### 3. MaxPooling
**What is MaxPooling?**
- MaxPooling is a downsampling technique that reduces the spatial dimensions of the feature maps.
- It helps in reducing computation and controlling overfitting.

**Why Used?**
- To retain the most important features while minimizing the amount of data processed.

### 4. Dropout
**What is Dropout?**
- Dropout is a regularization technique where randomly selected neurons are ignored during training to prevent overfitting.

**Why Used?**
- To ensure the model generalizes well to unseen data.

### 5. Adam Optimizer
**What is Adam?**
- Adam (Adaptive Moment Estimation) is an optimization algorithm that adjusts learning rates dynamically.

**Why Used?**
- It combines the benefits of RMSprop and Momentum, ensuring faster convergence with minimal tuning.

### 6. Categorical Cross-Entropy
**What is Categorical Cross-Entropy?**
- It is a loss function used for multi-class classification problems.

**Why Used?**
- Since the project classifies images into multiple categories (tumor/no tumor), categorical cross-entropy is the appropriate loss function.

### 7. Flask Framework
**What is Flask?**
- Flask is a lightweight web framework for Python.

**Why Used?**
- To provide a web interface for the trained model, enabling users to upload images and get predictions.

## Installation and Setup

To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/KAMRANKHANALWI/BrainTumorProject.git
   cd BrainTumorDetection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open the browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Future Improvements
- Integrating transfer learning with pre-trained models for better accuracy.
- Deploying on cloud services for broader accessibility.
- Implementing real-time MRI image processing via API integration.

## Conclusion
This project demonstrates the potential of deep learning in medical imaging applications. It provides an end-to-end solution for brain tumor detection with an easy-to-use web interface. Further improvements and optimizations can make it more robust and accurate in real-world scenarios.

