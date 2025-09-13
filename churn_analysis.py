# customer_churn_analysis.py
# Complete Customer Churn Analysis Prediction Project

# 1. Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import urllib.request
import os

# Set matplotlib to non-interactive mode
plt.ioff()

print("=" * 60)
print("CUSTOMER CHURN ANALYSIS PREDICTION")
print("=" * 60)

# Download dataset if not available
dataset_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
dataset_filename = "Telco-Customer-Churn.csv"

if not os.path.exists(dataset_filename):
    print("Downloading dataset...")
    try:
        urllib.request.urlretrieve(dataset_url, dataset_filename)
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download the dataset from:")
        print("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
        print("And save it in the same folder as this script")
        exit()
else:
    print("Dataset already exists.")

# Load the dataset
print("\n1. Loading the Dataset...")
dataset = pd.read_csv(dataset_filename)

# Display first few rows
print("\nDataset Head:")
print(dataset.head())

# 2. Understanding the Dataset
print("\n" + "=" * 60)
print("2. UNDERSTANDING THE DATASET")
print("=" * 60)

# Check for missing values
print("\nMissing values in dataset:")
print(dataset.isnull().sum())

# Check dataset statistics
print("\nDataset statistics:")
print(dataset.describe())

# Display dataset information
print("\nDataset info:")
dataset.info()

# 3. Analyzing Churn Distribution
print("\n" + "=" * 60)
print("3. ANALYZING CHURN DISTRIBUTION")
print("=" * 60)

# Check churn distribution
churn_counts = dataset['Churn'].value_counts()
print("\nChurn value counts:")
print(churn_counts)

# Visualize churn distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Churn', data=dataset, hue='Churn', palette='coolwarm', legend=False)
plt.title('Churn Distribution')
plt.xlabel('Churn (No = Stayed, Yes = Left)')
plt.ylabel('Count')
plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
plt.close()  # This will close the plot and allow the script to continue
print("Churn distribution plot saved as 'churn_distribution.png'")

# 4. Data Preprocessing
print("\n" + "=" * 60)
print("4. DATA PREPROCESSING")
print("=" * 60)

# Handle missing and incorrect values in TotalCharges
print("\nHandling TotalCharges column...")
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')

# Check if there are any missing values after conversion
missing_values = dataset['TotalCharges'].isnull().sum()
print(f"Found {missing_values} missing values in TotalCharges")

# Fill missing values with median
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)
print("Missing values filled with median")

# Handle categorical variables
print("\nEncoding categorical variables...")
labelencoder = LabelEncoder()

# List of categorical columns to encode
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod', 'Churn']

# Encode each categorical column
for col in categorical_cols:
    if col in dataset.columns:
        dataset[col] = labelencoder.fit_transform(dataset[col].astype(str))
        print(f"Encoded {col}")

print("All categorical variables encoded successfully")

# 5. Feature Selection and Splitting Data
print("\n" + "=" * 60)
print("5. FEATURE SELECTION AND DATA SPLITTING")
print("=" * 60)

# Separate features and target variable
X = dataset.drop(['customerID', 'Churn'], axis=1)
y = dataset['Churn']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# 6. Feature Scaling
print("\n" + "=" * 60)
print("6. FEATURE SCALING")
print("=" * 60)

# Apply standardization to features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed")

# 7. Model Training and Prediction
print("\n" + "=" * 60)
print("7. MODEL TRAINING AND PREDICTION")
print("=" * 60)

# Initialize and train Random Forest Classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

print("Model training and prediction completed")

# 8. Model Evaluation
print("\n" + "=" * 60)
print("8. MODEL EVALUATION")
print("=" * 60)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="coolwarm")
plt.title('Confusion Matrix - Random Forest')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Feature Importance Analysis
print("\n" + "=" * 60)
print("9. FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
print(feature_importances.head(10))

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature importance plot saved as 'feature_importance.png'")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"Final Model Accuracy: {accuracy:.2f}")
print("Visualizations have been saved as PNG files in the current directory")
print("=" * 60)