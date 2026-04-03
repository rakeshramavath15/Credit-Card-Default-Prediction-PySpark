# Credit-Card-Default-Prediction-PySpark
# Credit Card Default Prediction using Logistic Regression

## Project Overview
This project predicts whether a credit card customer will default on their next payment using a Logistic Regression model built with PySpark.

**Student:** Rakesh Ramavath  
**Program:** BCA 3rd Year - 5th Semester  
**University:** Amity University Online  
**Project:** TCS iON Industry Project  

---

## Dataset
- **Source:** UCI Machine Learning Repository
- **Records:** 30,000 credit card customers
- **Features:** 23 input features + 1 target column
- **Default Rate:** 22.12%

---

## Tools Used
- Google Colab
- PySpark 4.0.2
- Python 3.12
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn

---

## Project Steps
1. Data Loading and Cleaning
2. Handling Class Imbalance with Class Weights
3. One Hot Encoding for Categorical Columns
4. Feature Selection using Correlation Threshold
5. PCA - Dimensionality Reduction
6. Train Test Split (80/20)
7. Pipeline with VectorAssembler + StandardScaler + LogisticRegression
8. Hyperparameter Tuning with CrossValidator
9. Model Evaluation
10. Model Saving with Timestamp Versioning

---

## Results
| Metric | Score |
|---|---|
| AUC-ROC | 0.7321 |
| Accuracy | 68.25% |
| Precision | 76.66% |
| Recall | 68.25% |
| F1 Score | 70.69% |

---

## How to Run

### Option 1 - Google Colab (Recommended)
- Open the .ipynb file in Google Colab
- Run all cells in order
- Upload the dataset when prompted

### Option 2 - Local Machine
- Install Python 3.6+
- Run: pip install pyspark pandas matplotlib seaborn scikit-learn
- Remove the google.colab import cell
- Change file path to your local path
- Run in Jupyter Notebook

### Option 3 - Databricks
- Upload notebook to Databricks
- Remove pip install pyspark cell
- Remove google.colab upload cell
- SparkSession already exists in Databricks
- Upload dataset to DBFS first

---

## Charts Generated
1. Class Distribution
2. Correlation Heatmap
3. Feature Coefficients
4. Sigmoid Function
5. ROC Curve
6. Confusion Matrix

---

## Key Finding
PAY_1 (last month payment status) is the strongest predictor of credit default with a coefficient of 0.5053. If a customer missed last month's payment, they are very likely to default next month.Predicting credit card default using Logistic Regression with PySpark - TCS iON Industry Project
