# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import pandas, numpy, matplotlib.pyplot, and necessary sklearn modules.
2. Load the dataset using pd.read_csv()Separate features (x) and target (y) using .iloc.
3. Use train_test_split() to divide the data into training and test sets.
4. Create and fit a LinearRegression model with training data.
5. Use the trained model to predict test set results (y_pred).
6. Compute MSE, MAE, and RMSE for performance evaluation.
7. Plot scatter and regression line for training and test sets.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by:    MADHANRAJ P
RegisterNumber:  212223220052
*/
```

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

```
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
```

<img width="750" alt="image" src="https://github.com/user-attachments/assets/551e037b-c209-4ad1-a8e0-93173ede22fb">

```
X=df.drop('target',axis=1)
Y=df['target']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
```

```
sgd_clf.fit(X_train,Y_train)
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/4e8d4a22-41de-47f1-bb8c-59bcd32a0b5c">

```
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(Y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/b215ef46-3d14-4a75-8b8c-33fde6509893">

```
cm=confusion_matrix(Y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```
<img width="650" alt="image" src="https://github.com/user-attachments/assets/60985f71-e36a-4316-bf5f-150b2df345d1">


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
