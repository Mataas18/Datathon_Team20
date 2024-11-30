import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from schedulefree import AdamWScheduleFree
from sklearn.model_selection import train_test_split


# Load the CSV file
file_path = 'Data Files\V2\\train_data.csv'
df = pd.read_csv(file_path)

# Detect columns with -1 in some entry, or a string with -1 in it
columns_with_minus_one = df.columns[(df == -1).any()]
print(columns_with_minus_one)

## Taking the first column with -1 entry and delete the rest of the columns
idx = 4
column = columns_with_minus_one[idx]
print(column)
df = df.drop(columns=[col for i, col in enumerate(columns_with_minus_one) if i != idx])  # Eliminar las demás columnas
df = df.drop(columns='ind_launch_date') # Drop the date column '-1'

## Split the data into two parts: one with -1 and one without -1
df_with_minus_one = df[df[column] == -1]
df_without_minus_one = df[df[column] != -1]

## Take only the numerical data 
X_minus_one = df_with_minus_one.select_dtypes(include=[np.number])
y_minus_one = X_minus_one[column]


X = df_without_minus_one.select_dtypes(include=[np.number])
y = X[column]

## Split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Create the dataloaders
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Dataloaders
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Model to predict the missing numerical data
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

loss_fn = nn.MSELoss()
optimizer = AdamWScheduleFree(model.parameters(), lr=1e-6)

## Training the model
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    optimizer.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
## Testing the model
model.eval()
optimizer.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch.view(-1, 1))
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f'Test Loss: {test_loss}')

## Predict the missing data
model.eval()
optimizer.eval()
X_minus_one = torch.tensor(X_minus_one.values, dtype=torch.float32)
y_pred = model(X_minus_one)
y_pred = y_pred.detach().numpy()

print(y_pred)
