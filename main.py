'''
    Neural Network for Binary classification
'''
# Importing libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
 
# Dataset preprocessing
# -> Setting up our weights initialization
np.random.seed(123)
torch.manual_seed(123)

# -> Importing datasets
samples = pd.read_csv('entradas_breast.csv') # Samples for fed our NN
classes = pd.read_csv('saidas_breast.csv') # 0 or 1 
print(f'Dataset dimensions: {samples.shape}')

# -> Training and test split (25% for test and 75% for training)
train_samples, test_samples, train_classes, test_classes = train_test_split(samples, 
                                                                            classes,
                                                                            test_size=0.25) 
print(f'Training dataset dimensions: {train_samples.shape}')
print(f'Testing dataset dimensions: {test_samples.shape}')

# -> Transforming data to pytorch tensor
train_samples = torch.tensor(np.array(train_samples), dtype=torch.float)
train_classes = torch.tensor(np.array(train_classes), dtype=torch.float)
test_samples = torch.tensor(np.array(test_samples), dtype=torch.float)

# -> Creating Tensor dataset for feed our NN
dataset = torch.utils.data.TensorDataset(train_samples, train_classes)

# -> Creating DataLoader with mini-batch = 10 for train our NN
train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=10, 
                                           shuffle=True)
print('-'*50)
# Create Neural Network model
''' NN Architecture:
        30 Neurons at the input layer 
        16 Neurons at the hidden layer
        16 Neurons at the hidden layer
        1 Neurons at the output layer
    The numbers of neurons at the hidden layers follow this structure:
        n_neurons = (n_classes_total - n_classes_pred) / 2 ::: (30 - 1) / 2 = 16 
'''
model = nn.Sequential(  # Sequential architecture
    nn.Linear(30, 16, bias=True),  # Fully connected Neurons 30->16            (input layer)
    nn.ReLU(),          # Activation function into neurons output 
    nn.Linear(16, 16, bias=True),  # Fully connected Neurons 16->16            (hidden layer)
    nn.ReLU(),          # Activation function into neurons
    nn.Linear(16, 1, bias=True),   # Fully connected Neurons 16->1             (output layer)
    nn.Sigmoid()        # Activation function into neuron                       
)
print('Model architecture:')
print(model.parameters)
print('-'*50)

# -> Error function
criterion = nn.BCELoss() # Binary Cross Entropy

# -> Optimizer
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=0.001,
                             weight_decay=0.0001)

# Training model
'''
    Training_size = 426
    batch_size = 10
    N_iterations per epoch = Training_size/batch_size
    N_iterations per epoch = 42,6 or 43 iterations per epoch
'''
for epoch in range(100): # 100 epochs with mini-batch 10
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data       # input: Training samples
                                    # labels: True classes
        optimizer.zero_grad()
        outputs = model(inputs)     # Predict outputs ( model.forward(inputs) )
        
        loss = criterion(outputs,
                         labels)    # Error calculation
        
        loss.backward()             # Backpropagation algorithm
        
        optimizer.step()            # Weights update
        
        running_loss += loss.item() # Loss acumutation 
    print(f'Epoch: {epoch + 1} || Loss: {running_loss/len(train_loader):.4f} ') 

# Model Evaluation
model.eval()                        # Configurate model for evaluation part
predict = model(test_samples)       # Or model.forward(test_samples)
predict = np.array(predict > 0.5)
# -> Model Accuracy
accuracy = accuracy_score(test_classes, predict)
print(f'Accuracy: {accuracy*100:.2f} %')

# -> Confusion Matrix
matrix = confusion_matrix(test_classes, predict)
sns.heatmap(matrix, annot=True)
plt.show()