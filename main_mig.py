from Functions.load_data import load_data


data_path = 'Data Files/train_data.csv'
# Load data
data = load_data(data_path)

print(data.head())