## Logistic-Regression-Model
Diabetes Prediction using Logistic Regression 📊🏥 Project Overview  This project builds a Logistic Regression Machine Learning model to predict whether a patient is diabetic based on diagnostic health measurements. The project demonstrates the complete data science workflow from data analysis to model evaluation.

## Tech Stack
-Python
-Pandas
-NumPy
-Seaborn & Matplotlib
-Scikit-learn

##Dataset Features
-Pregnancies
-Glucose
-BloodPressure
-SkinThickness
-Insulin
-BMI
-DiabetesPedigreeFunction
-Age
-Outcome (Target variable)

## Project Steps

EDA

-Distribution plots
-Feature relationships
-Correlation heatmap
-Outlier detection

Data Preprocessing

-Replaced invalid zero values in Glucose
-Feature scaling using MinMaxScaler
-Train-test split (80-20)

## Evaluation Metrics

-Confusion Matrix
-Precision
-Recall
-F1 Score
-Accuracy

## Model Performance

The Logistic Regression model was evaluated using classification metrics to ensure balanced performance rather than relying only on accuracy.

Key Insights
-Glucose is the strongest predictor of diabetes
-BMI shows moderate correlation
-Age slightly influences risk
-Some features contain outliers
