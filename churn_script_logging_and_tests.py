# test_churn_library.py

import os
import logging
from sklearn.model_selection import train_test_split
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

def test_import(import_data):
    '''
    Test data import
    '''
    try:
        df = import_data(r"C:\Users\hp\Downloads\customer_churn_eda\data\bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: Dataframe is empty")
        raise err

    return df

def test_eda(perform_eda, df):
    '''
    Test EDA function
    '''
    try:
        perform_eda(df)
        assert os.path.exists("./images/eda/churn_hist.png")
        assert os.path.exists("./images/eda/customer_age_hist.png")
        assert os.path.exists("./images/eda/total_trans_ct.png")
        assert os.path.exists("./images/eda/trans_amt_boxplot.png")
        assert os.path.exists("./images/eda/heatmap.png")
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: FAILED")
        raise err

def test_encoder_helper(encoder_helper, df):
    '''
    Test encoder helper
    '''
    category_lst = [
        'Gender', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category'
    ]
    try:
        df_encoded = encoder_helper(df.copy(), category_lst, response='Churn')
        for col in category_lst:
            assert f"{col}_Churn" in df_encoded.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err

    return df_encoded

def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    Test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, response='Churn')

        assert all(X_train.dtypes != 'object'), "Non-numeric feature in training set"
        assert all(X_test.dtypes != 'object'), "Non-numeric feature in test set"

        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err

    return X_train, X_test, y_train, y_test

def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    Test model training
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("./images/results/roc_curve.png")
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: FAILED")
        raise err

if __name__ == "__main__":
    df = test_import(cls.import_data)
    test_eda(cls.perform_eda, df)
    df_encoded = test_encoder_helper(cls.encoder_helper, df)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, df_encoded)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
