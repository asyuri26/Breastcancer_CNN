# Multi-Layer Perceptron - Multimodal Breast Cancer Classification

This repository contains a project focused on implementing a Convolutional Neural Network (CNN) for breast cancer classification using a private dataset. The application is built using Streamlit for easy accessibility and interaction during local testing.

## Features

- User-friendly Streamlit interface for local experimentation.
- Convolutional Neural Network for image classification.
- Provides training and validation metrics, including accuracy and loss visualizations.
- Split the dataset into training and testing sets dynamically.
- Visualizes loss and accuracy across epochs.


## Dataset Information

The dataset used in this project is private and cannot be shared due to copyright restrictions. To replicate the results, users can prepare their own dataset structured in folders based on class labels.

### Dataset Structure

Ensure your dataset folder is structured as follows:
```
<dataset_root>/
    class_0/
        image1.jpg
        image2.jpg
        ...
    class_1/
        image1.jpg
        image2.jpg
        ...
```
The dataset root folder path can be set in the code to dynamically load and split the data.

## File Descriptions

- **app.py**: The main Python file containing the CNN implementation and Streamlit app interface.
- **requirements.txt**: A list of Python dependencies required to run the project.
- **modelCNN.h5**: Saved model weights (if any) for evaluation or inference.

## How to Use

1. Run the application and navigate to the local web interface.
2. From the sidebar, select a menu option:
   - **Home**: Displays project information and team members.
   - **Machine Learning**: Provides functionality to load dataset, configure the model, and train it.
3. Upload an example image to visualize preprocessing and generate predictions using the trained model.
4. Train the model and monitor performance metrics, including accuracy and loss.
5. Visualize training results with confusion matrix and epoch plots.

