# -*- coding: utf-8 -*-
"""Compute metrics for Novartis Datathon 2024.
   This auxiliar file is intended to be used by participants in case
   you want to test the metric with your own train/validation splits."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def _CYME(df: pd.DataFrame) -> float:
    """ Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))"""

    yearly_agg = df.groupby("cluster_nl")[["target", "prediction"]].sum().reset_index()
    yearly_error = abs((yearly_agg["target"] - yearly_agg["prediction"])/yearly_agg["target"]).median()

    monthly_error = abs((df["target"] - df["prediction"])/df["target"]).median()

    return 1/2*(yearly_error + monthly_error)

def _CYME_LOSS(prediction, target, clusters_nl) -> float:
    """ Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))"""

    yearly_agg = df.groupby("cluster_nl")[["target", "prediction"]].sum().reset_index()
    yearly_error = abs((yearly_agg["target"] - yearly_agg["prediction"])/yearly_agg["target"]).median()

    monthly_error = abs((df["target"] - df["prediction"])/df["target"]).median()

    return 1/2*(yearly_error + monthly_error)

def _metric(df: pd.DataFrame) -> float:
    """Compute metric of submission.

    :param df: Dataframe with target and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split 0 actuals - rest
    zeros = df[df["zero_actuals"] == 1]
    recent = df[df["zero_actuals"] == 0]

    # weight for each group
    zeros_weight = len(zeros)/len(df)
    recent_weight = 1 - zeros_weight

    # Compute CYME for each group
    return round(recent_weight*_CYME(recent) + zeros_weight*min(1,_CYME(zeros)),8)


def compute_metric(submission: pd.DataFrame) -> Tuple[float, float]:
    """Compute metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric.
    """

    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[['cluster_nl', 'date', 'target', 'prediction', 'zero_actuals']]

    return _metric(submission)

def compute_zero_actuals(train_data: pd.DataFrame, submission: pd.DataFrame, cluster_nl) -> pd.DataFrame:
    """Compute zero actuals.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Dataframe with zero_actuals column.
    """
    # Divide cluster_nl in tests and train by the indexes
    train_cluster_nl = cluster_nl[train_data.index]
    test_cluster_nl = cluster_nl[submission.index]

    # Check if cluster_nl in test has appears in train
    submission["zero_actuals"] = test_cluster_nl.isin(train_cluster_nl)

    # print the percentage of zero_actuals
    print(f"Percentage of zero actuals: {submission['zero_actuals'].mean()}")
       
    return submission

import os
def prepare_submission(submission):
    SAVE_PATH = "./Data Files/submissions/"
    # add date tp submission
    ATTEMPT = "attempt_" + str(len([f for f in os.listdir(SAVE_PATH) if "attempt" in f]) + 1)
    submission.to_csv(SAVE_PATH + f"submission_{ATTEMPT}.csv", sep=",", index=False)

if __name__ == "__main__":
    # Load data
    PATH = Path("path/to/data/folder")
    train_data = pd.read_csv(PATH / "train_data.csv")

    # Split into train and validation set

    validation = 0.1

    # Train your model

    # Perform predictions on validation set
    # validation["prediction"] = model.predict(validation[features])

    # Assign column ["zero_actuals"] in the depending if in your
    # split the cluster_nl has already had actuals on train or not

    validation["zero_actuals"] = True

    # Optionally check performance
    print("Performance:", compute_metric(validation))

    # Prepare submission
    submission_data = pd.read_parquet(PATH / "submission_data.csv")
    submission = pd.read_csv(PATH / "submission_template.csv")

    # Fill in 'prediction' values of submission
    submission["prediction"] = model.predict(submission_data[features])

    # ...

    # Save submission
    SAVE_PATH = Path("path/to/save/folder")
    ATTEMPT = "attempt_x"
    submission.to_csv(SAVE_PATH / f"submission_{ATTEMPT}.csv", sep=",", index=False)
