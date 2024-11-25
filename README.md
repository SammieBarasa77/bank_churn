# Machine Learning Project: Bank Customer Churn Analysis

![Logo](https://github.com/SammieBarasa77/bank_churn/blob/main/assets/images/Screenshot%202024-11-23%20222057.png)

# Table of Contents

1. [Introduction](#introduction)
2. [Data Import and Loading](#data-import-and-loading)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Customer Demographics](#customer-demographics)
   - [Account Tenure and Balance](#account-tenure-and-balance)
   - [Financial Behavior](#financial-behavior)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Building and Evaluation](#model-building-and-evaluation)
7. [Feature Importance Analysis](#feature-importance-analysis)
8. [Customer Retention Strategies](#customer-retention-strategies)

## 1. Introduction
Project objective and scope

Customer churn is a pressing challenge for banks, threatening profitability and growth. In this project, I leverage machine learning to identify high-risk customers, uncovering patterns that signal potential churn. By turning data into actionable insights, this analysis equips banks with the tools to address churn proactively.

Transforming Risk into Retention
Beyond prediction, this project focuses on providing clear, actionable recommendations. The goal is to help banks not only understand churn but also implement strategies that enhance customer satisfaction and loyalty.

Importance of customer churn analysis in the banking sector

Understanding Customer Retention: Customer churn analysis helps banks identify why customers leave, enabling targeted retention strategies to improve loyalty and satisfaction.

Improving Profitability: Retaining existing customers is more cost-effective than acquiring new ones, making churn analysis critical for sustaining profitability.

Enhancing Competitive Advantage: By addressing churn drivers, banks can differentiate themselves in a competitive market by offering tailored services and better customer experiences.

Optimizing Resource Allocation: Insights from churn analysis guide banks in prioritizing resources and investments toward high-value customers at risk of churning.


2. Data Import and Loading
For this project, I used **Google Colab** the IDE (Integrated Development Environment).

Necessary Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from google.colab import drive
```
Mounting Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Uploading and reading the dataset
```python
# Importing the dataset from my computer hard drive
from google.colab import files
uploaded = files.upload()
```
```python
import io
df = pd.read_csv(io.BytesIO(uploaded['Bank Customer Churn Prediction.csv']))
df
```
![Data Preview]()

Handle Missing Values
```python
import io

# Check for missing values
print(df.isnull().sum())

# Handle missing values (e.g., by replacing with mean, median, or mode)
# For numerical features, replace missing values with the mean
for column in df.columns:
  if pd.api.types.is_numeric_dtype(df[column]):
    df[column].fillna(df[column].mean(), inplace=True)

# For categorical features, replace missing values with the mode
for column in df.columns:
  if not pd.api.types.is_numeric_dtype(df[column]):
    df[column].fillna(df[column].mode()[0], inplace=True)


# Verify if missing values are handled
print(df.isnull().sum())
```

4. Exploratory Data Analysis (EDA)
A. Customer Demographics
```python
import seaborn as sns
import matplotlib.pyplot as plt

churn_column = 'churn'
```
Age: Analyzing churn trends by age
```python
# Age
plt.figure(figsize=(8, 6))
sns.boxplot(x=churn_column, y='age', data=df)
plt.title('Churn Rate by Age')
plt.show()
```

Gender: Gender-based churn analysis
```python
# Gender
gender_churn = df.groupby(['gender', churn_column])[churn_column].count().unstack()
gender_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Gender')
plt.show()
```
Geography: Churn trends by regions
```python
# Geography
geography_churn = df.groupby(['country', churn_column])[churn_column].count().unstack()
geography_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Geography')
plt.show()
```

B. Account Tenure and Balance

Tenure: Churn trends by customer tenure
```python
# Tenure
plt.figure(figsize=(8, 6))
sns.boxplot(x=churn_column, y='tenure', data=df)
plt.title('Churn Rate by Tenure')
plt.show()

```
Balance: Correlation between balance and churn
```python
# Balance
plt.figure(figsize=(8, 6))
sns.boxplot(x=churn_column, y='balance', data=df)
plt.title('Churn Rate by Balance')
plt.show()
```
C. Financial Behavior

Credit Score: Relationship between credit score and churn
```python
# Credit Score
plt.figure(figsize=(8, 6))
sns.boxplot(x=churn_column, y='credit_score', data=df)
plt.title('Churn Rate by Credit Score')
plt.show()
```
Number of Products: Impact of product holdings on churn
```python
# Number of Products
product_churn = df.groupby(['products_number', churn_column])[churn_column].count().unstack()
product_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Number of Products')
plt.show()
```
Estimated Salary: Effect of income levels on churn
```python
# Estimated Salary
plt.figure(figsize=(8, 6))
sns.boxplot(x=churn_column, y='estimated_salary', data=df)
plt.title('Churn Rate by Estimated Salary')
plt.show()

```

5. Feature Engineering

Creating interaction terms
Developing a custom risk score
Generating new features based on domain knowledge
```python
# Feature Engineering

# 1. Interaction Terms
df['Tenure_Balance'] = df['tenure'] * df['balance']
df['CreditScore_Balance'] = df['credit_score'] * df['balance']
df['Age_EstimatedSalary'] = df['age'] * df['estimated_salary']

# 2. Risk Score (Example)
df['RiskScore'] = (
    (df['balance'] < 5000) * 1 +
    (df['tenure'] < 2) * 1 +
    (df['products_number'] == 1) * 1
)
print(df)
```

6. Model Building and Evaluation

Splitting data into training and testing sets
```python
# Defining features (X) and target variable (y)
X = df.drop(churn_column, axis=1)
y = df[churn_column]

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE (if needed)
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)


```
Training models: Logistic Regression, Decision Tree, Random Forest, XGBoost
```python
# Choose models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier()
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)[:, 1]

  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_prob)
  confusion = confusion_matrix(y_test, y_pred)

```
Evaluating models with metrics (accuracy, precision, recall, F1-score, ROC-AUC)
```python
  results[model_name] = {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "roc_auc": roc_auc,
      "confusion_matrix": confusion
  }

```
Comparing model performance
```python

# Print evaluation results for each model
for model_name, metrics in results.items():
  print(f"Model: {model_name}")
  print(f"Accuracy: {metrics['accuracy']:.4f}")
  print(f"Precision: {metrics['precision']:.4f}")
  print(f"Recall: {metrics['recall']:.4f}")
  print(f"F1-score: {metrics['f1']:.4f}")
  print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
  print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
  print("-" * 30)
```
7. Feature Importance Analysis

Analyzing feature importance for tree-based models
```python
# 1. Random Forest

if 'Random Forest' in results:
  model = models['Random Forest']
  feature_importances = pd.Series(model.feature_importances_, index=X.columns)
  top_features = feature_importances.nlargest(10)  # Get the top 10 most important features
  print("Top 10 Important Features (Random Forest):")
  print(top_features)
  plt.figure(figsize=(10, 6))
  sns.barplot(x=top_features.values, y=top_features.index)
  plt.title('Feature Importance (Random Forest)')
  plt.xlabel('Importance Score')
  plt.ylabel('Feature')
  plt.show()
```
```python
# 2. XGBoost

if 'XGBoost' in results:
  model = models['XGBoost']
  feature_importances = pd.Series(model.feature_importances_, index=X.columns)
  top_features = feature_importances.nlargest(10)
  print("Top 10 Important Features (XGBoost):")
  print(top_features)
  plt.figure(figsize=(10, 6))
  sns.barplot(x=top_features.values, y=top_features.index)
  plt.title('Feature Importance (XGBoost)')
  plt.xlabel('Importance Score')
  plt.ylabel('Feature')
  plt.show()
```

The Best-performing model

The XGBoost model prioritizes the most predictive features more clearly, suggesting it may provide better churn predictions if the data aligns with its feature importance. However, the best model should be determined by evaluating performance metrics like accuracy, precision, recall, F1-score, or AUC-ROC. If XGBoost outperforms in these metrics, it is the better choice.

Visualizing the top features influencing churn

```python
# Churn Probability
# Use the best-performing model to predict churn probabilities for the test set.(XGBoost in this case)
if 'XGBoost' in results:
  best_model = models['XGBoost']
  y_prob = best_model.predict_proba(X_test)[:, 1]  # Get the probability of churn (class 1)

  # Create a DataFrame with customer IDs, actual churn, and predicted churn probability
  churn_predictions = pd.DataFrame({'Actual_Churn': y_test, 'Predicted_Churn_Probability': y_prob})

  # Identify high-risk customers (e.g., those with a churn probability above a threshold)
  threshold = 0.7  # You can adjust this threshold based on your business needs
  high_risk_customers = churn_predictions[churn_predictions['Predicted_Churn_Probability'] > threshold]

  print("Number of High-Risk Customers:", len(high_risk_customers))
  print("High-Risk Customers:")
high_risk_customers
```
8. Customer Retention Strategies
   
Findings

High-Risk Segments

1. Low Balance: Based on EDA and potential feature importance, customers with lower account balances might be at a higher risk of churn.

2. Short Tenure: Customers with a shorter tenure with the bank could also be at a higher risk.
 
3. Only One Bank Product: Customers who only utilize one bank product may be more likely to switch banks.
 
4. Specific Demographics: Based on EDA (age, gender, or country), certain demographics might be more prone to churn.
   
For instance, I found that younger customers and specific geographic regions had higher churn rates, these segments would be considered high-risk.

Recommendations for reducing churn and retaining customers

Retention Strategies

1. Personalized Offers

- Target customers with only one bank product with offers to bundle additional services (credit card, loan, investment).

- Provide tailored promotions based on identified needs and preferences of customers in high-risk segments.

2. Loyalty Programs

- Design reward programs for customers who have been with the bank for longer durations.

- Offer benefits and discounts for customers with higher account balances.

3. Proactive Communication

- Reach out to customers identified as high-risk through personalized messages or targeted campaigns.

- Offer incentives for them to stay with the bank

4. Improved Customer Service

- Focus on enhancing customer service for high-risk segments to address their needs and concerns proactively.

5. Targeted Incentives

- Offer discounts on fees or interest rates to high-risk customers who hold multiple products.

- Reward customers with exceptional behavior with special offers.
6. Customer Feedback and Surveys

- Conduct regular customer surveys to understand the reasons behind churn and improve overall customer experience.
