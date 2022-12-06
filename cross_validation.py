import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skorch
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import train_test_split, cross_val_score

INPUT   = pd.read_csv('entradas_breast.csv') 
CLASSES = pd.read_csv('saidas_breast.csv')

INPUT = np.array(INPUT, dtype='float')
CLASSES = np.array(CLASSES, dtype='float')

class model(nn.Module):
    def __init__(self,):
        super().__init__()
        # 30 -> 16 -> 16 -> 1
        dense0 = nn.Linear(30, 16)
        torch.nn.init.uniform(self.dense0.weight)
        activation0 = torch.nn.ReLU()
        
        dense1 = nn.Linear(16, 16)
        torch.nn.init.uniform(self.dense1.weight)
        activation1 = torch.nn.ReLU()
        
        dense2 = nn.Linear(16, 1)
        torch.nn.init.uniform(self.dense2.weight)
        output = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.dense0(x)
        x = self.activation0(x)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dense2(x)
        x = self.output(x)
        return x

neural_network = NeuralNetBinaryClassifier(module=model,
                                  criterion=nn.BCELoss(),
                                  optimizer=torch.optim.Adam,
                                  lr=1e-3,
                                  optimizer__weight_decay=1e-4,
                                  epochs=100,
                                  batch_size=10,
                                  train_split=False)

results = cross_val_score(neural_network,
                          INPUT,
                          CLASSES,
                          cv=10,
                          scoring='accuracy')
