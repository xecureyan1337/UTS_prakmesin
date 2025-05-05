# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# load dataset
df = pd.read_csv('dataset_comp.csv')
print(df.head())

# preprocessing data
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Pisahkan Fitur dan Label
X = df.drop('Buys_Computer', axis=1)
y = df['Buys_Computer']

# Split Data (Train dan Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model Naive bayes (NIM GENAP)
model = GaussianNB()
model.fit(X_train, y_train)

# evaluasi model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("------------------------------------")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("------------------------------------")
print("Classification Report:\n", classification_report(y_test, y_pred))

