import pandas as pd
import os
import torch

def preprocess_data():
    df = pd.read_pickle("new_data.pkl")
    df = df.dropna()
    df = df.reset_index(drop=True)
    # split data into train and test
    train_df = df.sample(frac=0.8, random_state=0)
    test_df = df.drop(train_df.index)
    val_df = test_df.sample(frac=0.5, random_state=0)
    test_df = test_df.drop(val_df.index)
    train_df.to_pickle("train_data.pkl")
    test_df.to_pickle("test_data.pkl")
    val_df.to_pickle("val_data.pkl")


preprocess_data()

def make_model():
    # Sanya this is for you!! WOOWOWOW THANKS BUDDY!! YAYYAYAYAY
    model = None
    # i found this for saving the model!
    # torch.save(model.state_dict(), "model.pkl")


def predict():
    model = torch.load("model.pkl")
    # return the prediction


df_train, df_test, df_val = pd.read_pickle("train_data.pkl"), pd.read_pickle(
    "test_data.pkl"), pd.read_pickle("val_data.pkl")

print(len(df_train), len(df_val), len(df_test))
