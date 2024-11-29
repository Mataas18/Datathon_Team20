from Functions.load_data import load_data
from metric_files.helper import compute_metric, compute_zero_actuals

data_path = 'Data Files/V2/train_data.csv'
# Load data
df = load_data(data_path)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from schedulefree import AdamWScheduleFree
from Functions.cyme_loss import CYMELoss
### CAMBIOS IVAN
# from Functions.categorical_cleaning import convert_categorical_to_numerical
from Preprocessing import convert_categorical_to_multilabel, date_codification, generate_unique_codes, format_dataframe2train

# Define Dataset class
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Define Feedforward Neural Network
class FeedForwardNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeedForwardNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)  # Assuming binary classification
        )

    def forward(self, x):
        return self.network(x)

 # Replace with your data source
df.replace(-1, np.nan, inplace=True)

cluster_nl = df["cluster_nl"]
# Drop the column 'cluster_nl'
if 'cluster_nl' in df.columns:
    df.drop(columns=['cluster_nl'], inplace=True)

# Split features and target
target_column = 'target'  # Replace with your target column
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle missing values
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:  # Numerical columns
        X[col].fillna(X[col].mean(), inplace=True)

# if there are still nan values, delete the rows and count them
nan_count = X.isnull().sum().sum()
X.dropna(inplace=True)
y = y[X.index]

print(f"Deleted {nan_count} rows with missing values")

# Handle categorical variables
dates = X["date"]
### CAMBIOS IVAN
# Codification of date column --> month and month + year * 12
X = date_codification(X)

X = X.drop(columns=['launch_date','ind_launch_date']) ## M

## Take the unique codes for each categorical column
indication_codes = generate_unique_codes(X, 'indication')
brand_codes = generate_unique_codes(X, 'brand')
drug_id_codes = generate_unique_codes(X, 'drug_id')
country_codes = generate_unique_codes(X, 'country')
corporation_codes = generate_unique_codes(X, 'corporation')
therapeutic_area_codes = generate_unique_codes(X, 'therapeutic_area')

# X = pd.get_dummies(X, drop_first=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = convert_categorical_to_multilabel(X_train, 'indication', indication_codes) # Multi-label binary matrix
X_test = convert_categorical_to_multilabel(X_test, 'indication', indication_codes) # Multi-label binary matrix
X_train = convert_categorical_to_multilabel(X_train, 'brand', brand_codes) # Multi-label binary matrix
X_test = convert_categorical_to_multilabel(X_test, 'brand', brand_codes) # Multi-label binary matrix
X_train = convert_categorical_to_multilabel(X_train, 'drug_id', drug_id_codes) # Multi-label binary matrix
X_test = convert_categorical_to_multilabel(X_test, 'drug_id', drug_id_codes) # Multi-label binary matrix
X_train = convert_categorical_to_multilabel(X_train, 'country', country_codes) # Multi-label binary matrix
X_test = convert_categorical_to_multilabel(X_test, 'country', country_codes) # Multi-label binary matrix
X_train = convert_categorical_to_multilabel(X_train, 'corporation', corporation_codes) # Multi-label binary matrix
X_test = convert_categorical_to_multilabel(X_test, 'corporation', corporation_codes) # Multi-label binary matrix
X_train = convert_categorical_to_multilabel(X_train, 'therapeutic_area', therapeutic_area_codes) # Multi-label binary matrix
X_test = convert_categorical_to_multilabel(X_test, 'therapeutic_area', therapeutic_area_codes) # Multi-label binary matrix

# Normalize data by dividing by the standard deviation of the training set only for numerical columns
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train_std = X_train[numerical_cols].std()
X_train_mean = X_train[numerical_cols].mean()
X_train[numerical_cols] = (X_train[numerical_cols] - X_train_mean )/ X_train_std
X_test[numerical_cols] = (X_test[numerical_cols] - X_train_mean) / X_train_std  # Normalize test set with training set std
y_train_std = y_train.std()
y_train = y_train / y_train_std  # Normalize target by dividing by the std

## Convert the multi-label arrays in multiple columns with 0 and 1 values
X_train = format_dataframe2train(X_train)
X_test = format_dataframe2train(X_test)

# convert Trues to 1 and Falses to 0
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Create DataLoader
train_dataset = TabularDataset(X_train.values, y_train.values)
test_dataset = TabularDataset(X_test.values, y_test.values)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
input_dim = X_train.shape[1]
model = FeedForwardNN(input_dim).to(device)
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.L1Loss()
criterion = CYMELoss()
optimizer = AdamWScheduleFree(model.parameters(), lr=0.0005)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# Training and Test Loss lists
train_losses = []
test_losses = []

# Training loop with test loss computation
epochs = 15
for epoch in range(epochs):
    model.train()
    optimizer.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluate on the test set
    model.eval()
    optimizer.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Plot training and test losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()


# get all predictions of the test set
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    outputs = model(inputs)
    predictions.extend(outputs.flatten().tolist())


# add predictions to data
X_test['prediction'] = [p * y_train_std for p in predictions]
X_test['target'] = y_test

# insert dates again in their original place
X_test.insert(0, "date", dates[X_test.index])
X_test.insert(1, "cluster_nl", cluster_nl[X_test.index])

compute_zero_actuals(X_train, X_test, cluster_nl)

metric = compute_metric(X_test)

print(metric)

# load test data
test_data_path = 'Data Files/V2/submission_data.csv'

# Load data
df_test = load_data(test_data_path)

# check target column
# df_test.replace(-1, np.nan, inplace=True)

# visualize data in the colum 'target'
plt.hist(df_test['target'], bins=100)
plt.title("Histogram of Target Column")
plt.xlabel("Target Value")
plt.ylabel("Frequency")
plt.show()
