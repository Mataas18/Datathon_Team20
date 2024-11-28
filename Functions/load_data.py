import pandas as pd

def load_data(path):
    # Replace 'your_file.csv' with the path to your CSV file
    data = pd.read_csv(path)
    print("Data loaded successfully!")

    return data



if __name__ == "__main__":
    
    file_path = 'Data Files/train_data.csv'

    # Load the CSV file into a DataFrame
    data = load_data(file_path)

    # Display the first few rows of the DataFrame
    print("Preview of the data:")
    print(data.head())

    # Display information about the dataset
    print("\nDataset Information:")
    print(data.info())

    # Check for missing values
    print("\nMissing values in the dataset:")
    print(data.isnull().sum())
