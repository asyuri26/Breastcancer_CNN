import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F #Adds some efficiency
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # For evaluating results

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plt

from PIL import Image
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def predict(image, model):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()
    
    return prediction

def predict_from_dataset(dataset, model):
    predictions = []
    labels = []
    
    # Iterate through the dataset
    for image, label in dataset:
        # Make prediction for each image
        prediction = predict(image, model)
        
        # Append the prediction and true label to lists
        predictions.append(prediction)
        labels.append(label)
    
    return predictions, labels

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 14 * 14, 64)    
        self.fc2 = nn.Linear(64, 32)      
        self.fc3 = nn.Linear(32, 16) 
        self.fc4 = nn.Linear(16, 1)     
        self.dropout = nn.Dropout(0.3)   
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 2) 
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = 2) 
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size = 2)  
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Create an instance of the SimpleCNN model

st.title("Image Classification with SimpleCNN")
st.write("Upload an image to classify")
menu = ["Home", "Machine Learning"]
choice = st.sidebar.selectbox("Menu", menu)

state_dict = torch.load(f="modelCNN.h5", map_location=device) #device is torch.device('cpu') or torch.device("cuda")
#model.load_state_dict(state_dict['model_state_dict'])
model = 'modelCNN.h5'
if choice == "Home":
    st.subheader("Home")
    st.text("Anggota kelompok")
    st.text("Salsabiela Khairunnisa Siregar - 5023201020")
    st.text("Irgi Azarya Maulana - 502332023")
    st.text("Muhammad Asyarie Fauzie - 5023201049")
    st.text("Andini Vira Salsabilla Z. P. - 5023201065")
    st.text("Reynard Prasetya Savero - 5023211042")

else : 
    st.subheader("Machine Learning")
    st.text("Contoh Input Breast Cancer")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        main_folder_path = "breast_cancer"
        
        dataset = datasets.ImageFolder(root=main_folder_path, transform=transform)

        # Split dataset into training and testing sets
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset,[train_size, test_size])

        # DataLoaders for training and testing sets
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
        
        st.write(f"Jumlah sampel dataset: {len(dataset)}")
        st.write(f"Jumlah sampel training set: {len(train_data)}")
        st.write(f"Jumlah sampel testing set: {len(test_data)}")
        st.write(f"Kelas dataset: {dataset.classes}")
        
        st.text("Model Convolutional Neural Network :")
        st.code("""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                # Define convolutional layers
                self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1)   # input channels: 1, output channels: 16, kernel size: 3x3, stride: 1
                self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1)  # input channels: 16, output channels: 32, kernel size: 3x3, stride: 1
                self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)  # input channels: 32, output channels: 64, kernel size: 3x3, stride: 1
                self.fc1 = nn.Linear(64 * 14 * 14, 120)    
                self.fc2 = nn.Linear(120, 84)        # input size: 120, output size: 84
                self.fc3 = nn.Linear(84, 1)         # input size: 84, output size: 10

            def forward(self, x):
                # Apply first convolutional layer followed by ReLU activation and max pooling
                x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 2) 
                # Apply second convolutional layer followed by ReLU activation and max pooling
                x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size = 2) 
                # Apply third convolutional layer followed by ReLU activation and max pooling
                x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size = 2)  
                # Flatten the output for fully connected layers
                x = x.view(-1, self.num_flat_features(x))
                # Apply first fully connected layer followed by ReLU activation
                x = F.relu(self.fc1(x))
                # Apply second fully connected layer followed by ReLU activation
                x = F.relu(self.fc2(x))
                # Apply the output layer
                x = self.fc3(x)
                return x
        """)
        
        train_losses = []
        test_losses = []
        train_correct = []
        test_correct = []
        
        model = SimpleCNN()
        st.write("Model CNN", model)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        if st.button("Train Model"):
            num_epochs = 20
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    labels = labels.float().unsqueeze(1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                train_losses.append(running_loss / len(train_loader))
                train_correct.append(correct / total)

                # Evaluation on test set
                model.eval()
                test_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        labels = labels.float().unsqueeze(1)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                test_losses.append(test_loss / len(test_loader))
                test_correct.append(correct / total)

                st.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_correct[-1]:.2%}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_correct[-1]:.2%}')
                    # Plotting the training and testing metrics
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot training and test loss
            axs[0].plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
            axs[0].plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
            axs[0].set_title('Loss vs Epochs')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Loss')
            axs[0].legend()
            
            # Plot training and test accuracy
            axs[1].plot(range(1, num_epochs + 1), train_correct, label='Training Accuracy')
            axs[1].plot(range(1, num_epochs + 1), test_correct, label='Testing Accuracy')
            axs[1].set_title('Accuracy vs Epochs')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            axs[1].legend()
            
            st.pyplot(fig)
        # conf_matrix = np.zeros((2, 2), dtype=int)
        # with torch.no_grad():
        #     correct = 0
        #     for inputs, labels in test_loader:
        #         outputs = model(inputs)
        #         predicted = torch.round(torch.sigmoid(outputs))
        #         correct += (predicted == labels).sum().item()
        #         for p, t in zip(predicted.view(-1), labels.view(-1)):
        #             conf_matrix[t.long(), p.long()] += 1

        # # Print confusion matrix
        # np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}')) 
        # st.write(np.arange(2).reshape(1,2))
        # st.write(conf_matrix)
        # predictions, labels = predict_from_dataset(test_data, model)

        # # Convert predictions and labels to numpy arrays for further processing
        # predictions = np.array(predictions)
        # labels = np.array(labels)

        # # Calculate accuracy
        # accuracy = (predictions == labels).mean()
        # st.write(f'Accuracy: {accuracy:.2%}')

        # # Calculate confusion matrix
        # conf_matrix = confusion_matrix(labels, np.round(predictions))
        # st.write('Confusion Matrix:')
        # st.write(conf_matrix)

        # # Visualize the confusion matrix
        # plt.figure(figsize=(8, 6))
        # #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        # plt.xlabel('Predicted Labels')
        # plt.ylabel('True Labels')
        # plt.title('Confusion Matrix')
        # st.pyplot()
        
    #     # Display the result
    #     st.write(f'Prediction: {prediction:.4f}')
    #     if prediction > 0.5:
    #         st.write('Class: 1')
    #     else:
    #         st.write('Class: 0')
