from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

#SKLEARN KNN SİTESİNDEN DETAYLI MESAFE HESAPLAMA TURLERİNE BAKILABİLİR.
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



knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski')#komşu sayısını artırdığımızda marjinal verilerde kendine komşu bulacağı için doğru tahmin yapıyoröuş gbi gözükür. buna dikkat etmek lazım.
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

print(y_pred)
print(y_test)

cm=confusion_matrix(y_test,y_pred) #tahmin edilecek değerler arasında oluşturulur

print(cm)