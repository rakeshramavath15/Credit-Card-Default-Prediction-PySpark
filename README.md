# Predictive Modeling of Credit Scores Through Logistic Regression

## Project Overview

This project is about predicting whether a credit card customer will default on their next payment or not. Credit card default is a big problem for banks and financial companies because they lose a lot of money when customers don't pay back. I built a machine learning model that can predict this before it happens.

I used the UCI Credit Card Default dataset which has 30,000 customer records from a Taiwan bank. The dataset has 24 features including credit limit, age, education, marriage status, payment history for 6 months and bill amounts for 6 months.

I built a Logistic Regression model using PySpark on Google Colab. Before building the model I cleaned the data, handled the class imbalance using class weights, applied One Hot Encoding for categorical columns and used PCA to reduce features. I used Pipeline to chain all steps together and CrossValidator to automatically find the best hyperparameters.

The model achieved 73.21% AUC-ROC score on test data. The biggest finding was that PAY_1  whether someone paid last month is by far the strongest signal of whether they will default next month.

## Key Features

- Distributed Processing: Utilizes PySpark MLlib to handle 30,000 customer records
- Class Imbalance Handling: Applied class weights to handle 78% vs 22% imbalance
- Feature Engineering: Used VectorAssembler to combine 16 selected features
- Dimensionality Reduction: Applied PCA to reduce features from 16 to 10 components
- Predictive Modeling: Uses Logistic Regression for binary classification
- Hyperparameter Tuning: CrossValidator tested 6 combinations with 5 fold cross validation
- Evaluation: Validated using AUC-ROC, Accuracy, Precision, Recall and F1 Score

## Tools and Technologies

- Language: Python 3.12
- Library: PySpark 4.0.2 (MLlib, SQL)
- Environment: Google Colab
- Dataset: UCI Credit Card Default Dataset (30,000 records)
- Other Libraries: Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn

## Results

| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| AUC-ROC | 0.7321 |
| Accuracy | 68.25% |
| Precision | 76.66% |
| Recall | 68.25% |
| F1 Score | 70.69% |
| Defaulters Caught | 864 out of 1303 (66.31%) |

## How to Run

1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/16CAHkWvlYYGMoKrdWv9urL3FC-mgEPz8?usp=sharing)
2. Upload the dataset file when prompted
3. Run all cells in order from top to bottom
4. ## Project Structure

    TCS-IonProject Code.ipynb    - Main project notebook
    TCS-IonProject Report.docx   - Project report
    class_distribution.png       - Class distribution chart
    confusion_matrix.png         - Confusion matrix chart
    correlation_heatmap.png      - Feature correlation heatmap
    feature_coefficients.png     - Feature importance chart
    roc_curve.png                - ROC curve chart
    sigmoid_function.png         - Sigmoid function chart
    README.md                    - Project documentation

## Credits

- Author: Rakesh Ramavath
- Organization: TCS iON Industry Project
