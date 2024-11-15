# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Use pandas for data manipulation, sklearn for dataset and model handling, and matplotlib/seaborn for visualization.
2. Load the Iris dataset using load_iris().
3. Convert the dataset into a DataFrame and add the target column (iris.target).
4. Use SGDClassifier with specified parameters (max_iter=1000, tol=1e-3).
5. Train the model on the training data (X_train, y_train).
6. Generate a Confusion Matrix using confusion_matrix(y_test, y_pred).
7. Print the accuracy and confusion matrix.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MADHANRAJ P
RegisterNumber: 212223220052

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data= iris.data,columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
*/

```

## Output:
![image](https://github.com/user-attachments/assets/82dbe48b-5331-48f5-9cff-b1864cd3c9cc)
![image](https://github.com/user-attachments/assets/9c7f8e41-6d4b-4672-bf66-0ec53b641bfb)
![image](https://github.com/user-attachments/assets/fccff6e7-5e2e-4be1-b362-a0241e2afc57)




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
