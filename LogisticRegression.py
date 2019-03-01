#1. kutuphaneler
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4]
y=veriler.iloc[:,4:]


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(x_train)#öğren
X_test=sc.transform(x_test)#öğrendiğin yöntemi kullan

#Logistic Regression###########################################################

logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train) # X train ile y train sonucuna göre eğitiyoruz

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)


#Sınıflandırma Hata Oranı=1-acc(M) acc(M)=model için yüzde kaç doğru sınıflandırma olduğudur.
cm=confusion_matrix(y_test,y_pred) #tahmin edilecek değerler arasında oluşturulur

print(cm)#çıkan sonuç matrisinde çapraz olarak bakılır, 1. çapraz(soldan sağa) doğru tahmini, 2. çapraz(sağdan sola) yanlış tahmini gösterir.
#[[0 1][6 1]] 0 'dan tahmininde 1 tanesi doğru,1'den tahmininde ise 6 tane yanlış vardır



