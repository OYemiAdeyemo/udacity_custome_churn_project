# library doc string
"""
Churn Library

This module provides functions to load data, perform EDA, encode features, engineer features,
train models, and visualize results for churn prediction tasks.
"""

# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    '''
    try:
        df = pd.read_csv(pth)
        return df
    except FileNotFoundError as err:
        print("File not found")
        raise err


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    '''
    if not os.path.exists("./images/eda"):
        os.makedirs("./images/eda")

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(6, 4))
    df['Churn'].hist()
    plt.title("Churn Histogram")
    plt.savefig('./images/eda/churn_hist.png')

    plt.figure(figsize=(6, 4))
    df['Customer_Age'].hist()
    plt.title("Customer Age Histogram")
    plt.savefig('./images/eda/customer_age_hist.png')

    plt.figure(figsize=(6, 4))
    sns.histplot(df['Total_Trans_Ct'], kde=True)
    plt.title("Total Transactions Count")
    plt.savefig('./images/eda/total_trans_ct.png')

    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Churn', y='Total_Trans_Amt', data=df)
    plt.title("Transaction Amount by Churn")
    plt.savefig('./images/eda/trans_amt_boxplot.png')

    # plt.figure(figsize=(10, 6))
    # numeric_df = df.select_dtypes(include='number')
    # sns.heatmap(numeric_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    # plt.title("Feature Correlation Heatmap")
    # plt.savefig('./images/eda/heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category
    '''
    for cat in category_lst:
        means = df.groupby(cat)[response].mean()
        df[f"{cat}_{response}"] = df[cat].map(means)
    return df


def perform_feature_engineering(df, response='Churn'):
    """
    Validates that feature engineering outputs expected format
    """
    y = df[response]

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    X = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test



def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results
    and stores report as image in images folder
    '''
    if not os.path.exists("./images/results"):
        os.makedirs("./images/results")

    def save_report(report, filename):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')
        ax.text(
            0,
            1,
            report,
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=10,
            family='monospace')
        plt.savefig(f'./images/results/{filename}', bbox_inches='tight')

    save_report(
        classification_report(y_train, y_train_preds_lr),
        'lr_train_report.png'
    )
    save_report(
        classification_report(y_test, y_test_preds_lr),
        'lr_test_report.png'
    )
    save_report(
        classification_report(y_train, y_train_preds_rf),
        'rf_train_report.png'
    )
    save_report(
        classification_report(y_test, y_test_preds_rf),
        'rf_test_report.png'
    )


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    '''
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_pth, 'feature_importance.png'))


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train and save models
    '''

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./images/results"):
        os.makedirs("./images/results")

    # Logistic Regression pipeline with scaling and increased max_iter
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    # Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Fit models
    lr_pipeline.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Predictions
    y_train_preds_lr = lr_pipeline.predict(X_train)
    y_test_preds_lr = lr_pipeline.predict(X_test)
    y_train_preds_rf = rf_model.predict(X_train)
    y_test_preds_rf = rf_model.predict(X_test)

    # Save models
    joblib.dump(lr_pipeline, './models/logistic_model.pkl')
    joblib.dump(rf_model, './models/rfc_model.pkl')

    # Save classification report image
    classification_report_image(
        y_train, y_test,
        y_train_preds_lr, y_train_preds_rf,
        y_test_preds_lr, y_test_preds_rf
    )

    # ROC Curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(rf_model, X_test, y_test, ax=ax, alpha=0.8, name="Random Forest")
    RocCurveDisplay.from_estimator(lr_pipeline, X_test, y_test, ax=ax, alpha=0.8, name="Logistic Regression")
    plt.title("ROC Curves")
    plt.savefig('./images/results/roc_curve.png')
    plt.close()
