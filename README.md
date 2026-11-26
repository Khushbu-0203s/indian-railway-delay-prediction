# Indian Railway Train Delay Prediction Using Linear Regression

## 1. Project Overview

This project focuses on predicting Indian railway train delays using Machine Learning, specifically Linear Regression.

The main objective is to estimate train delays in minutes using historical railway stop records, train and station information, and historical delay patterns.

The project involves data cleaning, exploratory data analysis, feature engineering, model development, and performance evaluation.

## 2. Project Objectives

- Analyze historical Indian railway train delay records.
- Clean and preprocess large-scale railway data.
- Perform Exploratory Data Analysis (EDA).
- Identify patterns and relationships in train delays.
- Develop historical features for delay prediction.
- Build a Linear Regression model.
- Evaluate prediction performance using regression metrics.

## 3. Dataset Description

The project uses approximately 36.5 million historical railway stop records covering February 2025 to February 2026.

### Data Processing Summary

| Stage                                   | Number of Records |
| --------------------------------------- | ----------------: |
| After cleaning and journey-jump removal |        36,480,745 |
| Final modelling table                   |        35,524,132 |
| Training data                           |        25,879,636 |
| Validation data                         |         2,971,571 |
| Testing data                            |         6,672,925 |

The modelling table excludes records without a matching schedule.

**Note:** The original dataset is not included in this repository because of its large size.

## 4. Technologies and Libraries Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 5. Project Methodology

The project follows these major steps:

### Step 1: Data Loading and Understanding

Loading the railway dataset and examining its structure, columns, data types, and records.

### Step 2: Data Cleaning and Preprocessing

Removing invalid records and identifying unusual journey jumps to improve data quality.

### Step 3: Exploratory Data Analysis (EDA)

Analyzing railway delay patterns and examining relationships between relevant variables.

### Step 4: Feature Engineering

Creating useful prediction features, including:

- Station number
- Distance
- Time-of-day indicator variables
- Train-level target encoding
- Station-level target encoding
- Seven-day historical delay features at train, station, and network levels

### Step 5: Time-Based Data Splitting

To avoid future-data leakage, the dataset was divided chronologically:

- Training: February–October 2025
- Validation: November 2025
- Testing: December 2025–February 2026

### Step 6: Model Development

Linear Regression was selected as the final prediction algorithm.

The final model uses 10 features and predicts the raw delay value in minutes.

The historical feature window was selected using validation data.

### Step 7: Model Evaluation

The final model was evaluated on the testing dataset using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Prediction Bias

## 6. Final Model Performance

The final Linear Regression model achieved the following results on the test dataset:

| Evaluation Metric |         Result |
| ----------------- | -------------: |
| MAE               | 30.282 minutes |
| RMSE              | 58.267 minutes |
| R² Score          |          0.361 |
| Prediction Bias   | +0.134 minutes |

### Interpretation of Results

**Mean Absolute Error (MAE):** The model's average absolute prediction error is approximately 30.28 minutes.

**Root Mean Squared Error (RMSE):** The RMSE is approximately 58.27 minutes, indicating that larger prediction errors have a stronger influence on the metric.

**R-squared (R²):** The model explains approximately 36.1% of the variation in the target variable on the test dataset.

**Prediction Bias:** The average prediction bias is approximately +0.13 minutes, indicating very little overall systematic bias.

## 7. Project Limitations

- The dataset covers approximately one year of railway records.
- Weather conditions are not included.
- Railway congestion and disruption information is not included.
- The model has difficulty predicting extremely large delays.
- Linear Regression may not capture all complex and non-linear delay patterns.

## 8. Future Improvements

- Include weather and railway disruption information.
- Explore additional historical and operational features.
- Experiment with non-linear Machine Learning algorithms such as Gradient Boosting.
- Improve prediction accuracy for extreme delays.
- Evaluate the model using additional historical data.

## 9. Repository Structure

```text
indian-railway-delay-prediction/
│
├── Indian_Railway_Delay_Prediction.ipynb
├── README.md
└── .gitignore
```

## 10. How to Run the Project

1. Clone or download this GitHub repository.
2. Install Python and Jupyter Notebook.
3. Install the required Python libraries.
4. Obtain the original dataset separately.
5. Configure the dataset path in the notebook.
6. Open the Jupyter Notebook.
7. Execute the notebook cells in the required order.

**Important:** The notebook uses a local dataset path. It must be updated to match the location of the dataset on another computer before execution.

## 11. Conclusion

This project demonstrates the application of Machine Learning to Indian railway train delay prediction.

Through large-scale data preprocessing, exploratory analysis, historical feature engineering, and Linear Regression modelling, the project investigates historical railway delay patterns and develops a model to estimate train delays.

The final model achieved an MAE of 30.282 minutes and an R² score of 0.361 on the test dataset.

The results demonstrate the potential of historical railway data for delay prediction while highlighting opportunities for further improvement.

## 12. Author

**Khushbu Sharma**

Shoolini University

Machine Learning Project

