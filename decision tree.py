import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv(r'C:\Users\afar9\Desktop\Customer Churn Prediction using Decision Trees\faraz khan - project4_dataset.csv')


data.drop(columns=['customerID'],inplace=True)

data_encoded  = pd.get_dummies(data)
X=data_encoded.drop(columns=['Churn_No', 'Churn_Yes'])
Y=data_encoded[ 'Churn_Yes']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=45)
model=DecisionTreeClassifier(random_state=45)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,y_pred)
print(accuracy  )
# 0.7 accuracy

print("Classification Report:")
print(classification_report(Y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
from sklearn.tree import plot_tree
plot_tree(model, feature_names=X.columns, class_names=['Not Churned', 'Churned'], filled=True)
plt.show()

