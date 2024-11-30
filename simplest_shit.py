from Functions.load_data import load_data
from metric_files.helper import compute_metric, compute_zero_actuals, prepare_submission

data_path = 'Data Files/V2/train_data.csv'
# Load data
data = load_data(data_path)


class simple_model:
    def __init__(self):
        self.therapeutic_area_means = {}

    def train(self, train_data):
        """
        Compute the mean of the output variable for each therapeutic_area in the training set.
        
        :param train_data: Pandas DataFrame with columns ['therapeutic_area', 'output']
        """
        # Group by 'therapeutic_area' and compute the mean of 'output'
        self.therapeutic_area_means = (
            train_data.groupby('therapeutic_area')['target']
            .mean()
            .to_dict()
        )

    def predict(self, data):
        """
        Predict the mean output for each therapeutic_area in the input data.
        
        :param data: Pandas DataFrame with a column ['therapeutic_area']
        :return: List of predictions
        """
        predictions = []
        for index, row in data.iterrows():
            # Lookup the mean value for the therapeutic_area, default to None if not found
            predictions.append(self.therapeutic_area_means.get(row['therapeutic_area'], None))
        return predictions

    
model = simple_model()

# split data into train and validation
split = 0.9
train_data = data.sample(frac=split, random_state=0)
validation_data = data.drop(train_data.index)

model.train(train_data)

predictions = model.predict(validation_data)

# add predictions to data
validation_data["zero_actuals"] = 0
validation_data['prediction'] = predictions

metric = compute_metric(validation_data)

print(metric)

test_data_path = 'Data Files/V2/submission_data.csv'

df_test = load_data(test_data_path)

predictions = model.predict(df_test)

df_test['prediction'] = predictions

compute_zero_actuals(train_data, df_test, data['cluster_nl'])

prepare_submission(df_test)