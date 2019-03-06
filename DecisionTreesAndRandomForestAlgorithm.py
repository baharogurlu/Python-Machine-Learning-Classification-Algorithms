#Karar ağaçlarında bölgelerin hangi özellğe göre ayrılacağına entropi ile karar veririz.
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4]
y=veriler.iloc[:,4:]


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(x_train)#öğren
X_test=sc.transform(x_test)#öğrendiğin yöntemi kullan
#KARAR AĞAÇLARI ###########################################################################
dtc=DecisionTreeClassifier(criterion='entropy')
#criterion veya entropi ve gini kullanılmasını sağlıyor. Default olarak gini kullanır,entropiyi belirtirsek ancak o zaman kullanır.
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
print('KARAR AĞAÇLARI')
print(y_test)
print(y_pred)

cm=confusion_matrix(y_test,y_pred) #tahmin edilecek değerler arasında oluşturulur
print(cm)


#RANDOM FOREST RASSAL AĞAÇLAR ###########################################################################
rfc=RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
print('RANDOM FOREST ')
print(y_test)
print(y_pred)

cm=confusion_matrix(y_test,y_pred) #tahmin edilecek değerler arasında oluşturulur
print(cm)