#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data (you can replace this with your own data)
data = {
    'Age': [30, 25, 35, 28, 42, 45, 39, 33],
    'JobSatisfaction': [4, 2, 3, 2, 5, 4, 3, 4],
    'TotalWorkingYears': [8, 3, 10, 5, 20, 22, 12, 9],
    'Attrition': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Data preprocessing
# Encode categorical variables and feature scaling (if needed)
# For simplicity, we are skipping preprocessing in this example.

# Split the data into training and testing sets
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the model's performance
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)

# Now, you can use the trained model to make predictions for new employee data
# For example:
new_employee_data = pd.DataFrame({
    'Age': [38],
    'JobSatisfaction': [3],
    'TotalWorkingYears': [15]
})

# Make predictions for the new employee
prediction = model.predict(new_employee_data)
print(f'Predicted Attrition: {prediction[0]}')


# In[ ]:




