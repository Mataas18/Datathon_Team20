from Functions.load_data import load_data
from metric_files.helper import compute_metric, compute_zero_actuals, prepare_submission

data_path = 'Data Files/V2/train_data.csv'
# Load data
df = load_data(data_path)

test_data_path = 'Data Files/V2/submission_data.csv'
df_test = load_data(test_data_path)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from schedulefree import AdamWScheduleFree
from Functions.cyme_loss import CYMELoss
### CAMBIOS IVAN
# from Functions.categorical_cleaning import convert_categorical_to_numerical
from Preprocessing import date_codification, generate_unique_codes, convert_categorical_to_multilabel, process_dataframe_multiple_columns

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
    def __init__(self, input_dim, hidden_dim=16):
        super(FeedForwardNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(input_dim, hidden_dim, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1)  # Assuming binary classification
        )

    def forward(self, x):
        return self.network(x)

 # Replace with your data source
df.replace(-1, np.nan, inplace=True)
df_test.replace(-1, np.nan, inplace=True)

cluster_nl = df["cluster_nl"]
cluster_nl_test = df_test["cluster_nl"]
# Drop the column 'cluster_nl'
if 'cluster_nl' in df.columns:
    df.drop(columns=['cluster_nl'], inplace=True)
if 'cluster_nl' in df_test.columns:
    df_test.drop(columns=['cluster_nl'], inplace=True)

# Split features and target
target_column = 'target'  # Replace with your target column
X = df.drop(columns=[target_column])
y = df[target_column]
X_f_test = df_test.drop(columns=[target_column])

# Handle missing values
for col in X_f_test.columns:
    if X_f_test[col].dtype in ['float64', 'int64', 'float32']:  # Numerical columns
        X_f_test[col].fillna(X[col].mean(), inplace=True)

for col in X.columns:
    if X[col].dtype in ['float64', 'int64', 'float32']:  # Numerical columns
        X[col].fillna(X[col].mean(), inplace=True)


# if there are still nan values, delete the rows and count them
nan_count = X.isnull().sum().sum()
X.dropna(inplace=True)
y = y[X.index]

print(f"Deleted {nan_count} rows with missing values")

# Handle categorical variables
dates = X["date"]
dates_test = X_f_test["date"]

X.drop(columns=['date'], inplace=True)
X_f_test.drop(columns=['date'], inplace=True)

# get dummies from the union of X and X_test and separate them afterwards
X_union = pd.concat([X, X_f_test], axis=0)

# drop columns that are not useful
X_union.drop(columns=['launch_date', 'ind_launch_date'], inplace=True)
# X_union.drop(columns=['country'], inplace=True)
# X_union.drop(columns=['corporation'], inplace=True)

# indication_codes = generate_unique_codes(X_union, 'indication')
# X_union = convert_categorical_to_multilabel(X_union, 'indication', indication_codes) 

X_union = process_dataframe_multiple_columns(X_union, ['brand', 'corporation', 'country', 'indication'], [420, 80, 39, 210])
# X_union = process_dataframe_multiple_columns(X_union, ['brand', 'indication'], [420, 210])

X_union = pd.get_dummies(X_union)

# add second and third order numerical values
numerical_cols = X_union.select_dtypes(include=['float64', 'float32']).columns
for col in numerical_cols:
    X_union[col + '_2'] = X_union[col] ** 2
    X_union[col + '_3'] = X_union[col] ** 3

X = X_union.iloc[:X.shape[0], :]
X_f_test = X_union.iloc[X.shape[0]:, :]

# add dates as year * 12 + month. compute the first date as the reference
dates = pd.to_datetime(dates)
dates_test = pd.to_datetime(dates_test)
first_date = dates.min()
dates = (dates.dt.year * 12 + dates.dt.month) - (first_date.year * 12 + first_date.month)
dates_test = (dates_test.dt.year * 12 + dates_test.dt.month) - (first_date.year * 12 + first_date.month)

X.insert(0, "date", dates)
X_f_test.insert(0, "date", dates_test)

# Codification of date column --> month and month + year * 12
# X = date_codification(X)
# X = convert_categorical_to_numerical(X, 'indication') # Multi-label binary matrix

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Normalize data by dividing by the standard deviation of the training set only for numerical columns
numerical_cols = X_train.select_dtypes(include=['float64', 'float32']).columns
X_train_std = X_train[numerical_cols].std()
X_train_mean = X_train[numerical_cols].mean()
X_train[numerical_cols] = (X_train[numerical_cols] - X_train_mean )/ X_train_std
X_test[numerical_cols] = (X_test[numerical_cols] - X_train_mean) / X_train_std  # Normalize test set with training set std
X_f_test[numerical_cols] = (X_f_test[numerical_cols] - X_train_mean) / X_train_std  # Normalize test set with training set std
y_train_std = y_train.std()
y_train = y_train / y_train_std  # Normalize target by dividing by the std

# convert Trues to 1 and Falses to 0
X_train = X_train.astype(float)
X_test = X_test.astype(float)
X_f_test = X_f_test.astype(float)

# Create DataLoader
train_dataset = TabularDataset(X_train.values, y_train.values)
test_dataset = TabularDataset(X_test.values, y_test.values)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

test_f_dataset = TabularDataset(X_f_test.values)
test_f_loader = DataLoader(test_f_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
input_dim = X_train.shape[1]
model = FeedForwardNN(input_dim).to(device)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
# criterion = CYMELoss()
optimizer = AdamWScheduleFree(model.parameters(), lr=0.00025, weight_decay=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# Training and Test Loss lists
train_losses = []
test_losses = []

# Training loop with test loss computation
epochs = 100
optimizer.train()
for epoch in range(epochs):
    model.train()
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
X_test['date'] = dates[X_test.index]
X_test.insert(1, "cluster_nl", cluster_nl[X_test.index])

compute_zero_actuals(X_train, X_test, cluster_nl)

metric = compute_metric(X_test)

print(metric)


# eval in test_f
predictions = []
for inputs in test_f_loader:
    inputs = inputs.to(device)
    outputs = model(inputs)
    predictions.extend(outputs.flatten().tolist())

# add predictions to data
X_f_test['prediction'] = [p * y_train_std for p in predictions]

# load again test data
df_test = load_data(test_data_path)

df_test['prediction'] = X_f_test['prediction']

prepare_submission(df_test)
