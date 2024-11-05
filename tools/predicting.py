import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def eval_basic_regression(test_df: pd.DataFrame, model):
    X_test = test_df[['HP', 'Attack', 'Defense', 'SP. Atk.', 'SP. Def', 'Speed']]
    y_test = test_df['Total']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2
    