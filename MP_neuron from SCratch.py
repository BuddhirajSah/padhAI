import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
#Loading DataSets
breast_cancer=sklearn.datasets.load_breast_cancer()
X=breast_cancer.data
Y=breast_cancer.target
#print(X.shape,Y.shape)
data=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
data['class']=breast_cancer.target
#print(data.head())
print(breast_cancer.target_names)
#Train test Split
from sklearn.model_selection import train_test_split
X = data.drop('class', axis=1)
Y=data['class']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
#print(X.mean)
### Binarisation of Input

X_binarised_3_train = X_train['mean area'].map(lambda x: 0 if x < 1000 else 1)
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])

X_binarised_test=X_test.apply(pd.cut, bins=2,labels=[1,0])
#print(type(X_binarised_test))
X_binarised_test=X_binarised_test.values
X_binarised_train=X_binarised_train.values
print(type(X_binarised_test))

#### [] MP  Neuron [] []


#Now Choosing Random b and checking which  of them gave more accuracy
for b in range(X_binarised_train.shape[1]+1):
    Y_pred_train=[]
    accurate_rows=0
    for x, y in zip(X_binarised_train, Y_train):
        Y_pred = (np.sum(x) >= b)
        Y_pred_train.append(Y_pred)
        accurate_rows += (y == Y_pred)
    print(b,accurate_rows, accurate_rows / X_binarised_train.shape[0])
from sklearn.metrics import  accuracy_score
b=27
Y_pred_test=[]
for x in X_binarised_test:
    y_pred=(np.sum(x)>=b)
    Y_pred_test.append(y_pred)
accuracy=accuracy_score(Y_pred_test,Y_test)
print('- - '*15)
print("My first simplest neural network")
print('Bias(b)='+str(b)+'  '+'Accuracy='+ str((accuracy)*100)+'%')


