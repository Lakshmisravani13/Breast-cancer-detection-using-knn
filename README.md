# Breast-cancer-detection-using-knn
import numpy as np
from sklearn.datasets import load_breast_cancer from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler from sklearn.neighbors import KNeighborsClassifier from sklearn.metrics import accuracy_score
import pandas as pd
# Load the breast cancer dataset data = load_breast_cancer()
# Split the data into features and target X = data.data
y = data.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features scaler =  StandardScaler() X_train =
scaler.fit_transform(X_train),
 X_test = scaler.transform(X_test)
# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Train the classifier knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) print("Accuracy:", accuracy)



# Allow user to input their own data
print("\nEnter the file name containing the data to test  algorithm:") #file_name = input("File name (including extension): ")
# Read user data from file
user_data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

# Extract features from user data X_user=user_data.values

# Standardize the user input data X_user_scaled=scaler.transform(X_use)

# Predict the class of the sample predicted_classes=knn.predict(X_use scaled)

print("\nPredicted Classes for User Data:") 
for predicted_class in =predicted_classes:
    if     
    predicted_class ==0:
print("Malignant")
 else: print("Benign")

