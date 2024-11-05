import pandas as pd

from tools.preprocess import split_csv_data
from tools.training import train_basic_regression, train_yolo, train_optimized_regression
from tools.predicting import eval_basic_regression

def basic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame):
    model = train_basic_regression(train_df)
    mse, r2 = eval_basic_regression(test_df, model)
    return mse, r2

def optimized_regression(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    model = train_optimized_regression(train_df, val_df)
    mse, r2 = eval_basic_regression(test_df, model)
    return mse, r2

if __name__ == "__main__":
    csv_path = "kaggle/pokedex.csv"
    train_df, val_df, test_df = split_csv_data(csv_path)
    mse, r2 = basic_regression(train_df, test_df)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    
    mse, r2 = optimized_regression(train_df, val_df, test_df)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
    