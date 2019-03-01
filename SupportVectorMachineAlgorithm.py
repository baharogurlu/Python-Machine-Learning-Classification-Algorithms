
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4]
y=veriler.iloc[:,4:]


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(x_train)#öğren
X_test=sc.transform(x_test)#öğrendiğin yöntemi kullan

#KNN(En yakın Komşu Algoritması)###########################################################



svc_lin = SVC(kernel='rbf')#linear,poly vs örneklerde koyulabilir ve sonuçlarına bakılabilir.
svc_lin.fit(X_train,y_train)

y_pred=svc_lin.predict(X_test)

print(y_pred)
print(y_test)

cm=confusion_matrix(y_test,y_pred) #tahmin edilecek değerler arasında oluşturulur

print(cm)

#KERNEL TRICK######################################### Doğrusal Olmayan Sınıflandırma da ayrım yapmak için kullanılır
#Veri doğrusal düzlemden farklı bi boyuta 3.boyuta taşınarak sınıflandırılır.
