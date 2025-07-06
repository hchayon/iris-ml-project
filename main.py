#import os
#print("Current Working Directory:", os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

df = pd.read_csv('iris.csv')

#url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
#df = pd.read_csv(url)

#print(df.head())

#df.info()
#df.describe()

#print(df['species'].value_counts())
#sns.pairplot(df, hue='species')
#plt.show()

#plt.figure(figsize=(6,4))
#sns.heatmap(df.drop('species',axis=1).corr(),annot=True)
#plt.show()

# Seperate features and target variable
X = df.drop('species', axis=1)
y = df['species']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Create and train the Random Forest model
clf = RandomForestClassifier( random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
feature_importances = clf.feature_importances_
feature_names = X.columns

for feat, importance in zip(feature_names, feature_importances):
    print(f"{feat}: {importance:.4f}")

# Plot feature importances
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()