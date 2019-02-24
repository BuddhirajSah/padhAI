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

#Train test Split
from sklearn.model_selection import train_test_split
X = data.drop('class', axis=1)
Y=data['class']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
#print(X.mean)
### Binarisation of Input
#plt.plot(X_test.T,'.')
#plt.xticks(rotation='vertical')
#plt.show()
X_binarised_3_train = X_train['mean area'].map(lambda x: 0 if x < 1000 else 1)
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1,0])
#plt.plot(X_binarised_train.T,'*')
#plt.xticks(rotation='vertical')
#plt.show()
X_binarised_test=X_test.apply(pd.cut, bins=2,labels=[1,0])
print(type(X_binarised_test))
X_binarised_test=X_binarised_test.values
X_binarised_train=X_binarised_train.values
print(type(X_binarised_test))

#### [] PERCEPTRON (class) [] []
X_test=X_test.values
X_train=X_train.values
from sklearn.metrics import accuracy_score
class Perceptron:
    def __init__(self):
        self.w=None
        self.b=None
    def model(self,x):
        return 1 if (np.dot(self.w,x)>=self.b) else 0
    def predict(self,X):
        Y=[]
        for x in X:
            result=self.model(x)
            Y.append(result)
        return np.array(Y)
    def fit(self,X,Y,epoch=1,lr=1):

        self.w=np.ones(X.shape[1])
        self.b=0

        accuracy={}
        max_accuracy=0

        wt_matrix=[]
        # if mismatch occur between y  and pred_y then adjust the w and b
        for i in range(epoch):
            for x,y in zip(X,Y):
                y_pred=self.model(x)
                if y==1 and y_pred==0:
                    self.w= self.w + lr * x
                    self.b=self.b - lr * 1
                elif y==0 and  y_pred==1:
                    self.w=self.w - lr*x
                    self.b=self.b + lr*1

            wt_matrix.append(self.w)
            accuracy[i]= accuracy_score(self.predict(X),Y)
            if accuracy[i] > max_accuracy:
                max_accuracy=accuracy[i]
                checkpoint_b=self.w
                checkpoint_w=self.w

        self.w=checkpoint_w
        self.b=checkpoint_b

        print(max_accuracy)
        plt.plot(accuracy.values(),'.')
        plt.ylim([0,1])
        plt.show()

        return np.array(wt_matrix)

perceptron=Perceptron()
wt_matrix=perceptron.fit(X_train,Y_train,10000,0.5)



