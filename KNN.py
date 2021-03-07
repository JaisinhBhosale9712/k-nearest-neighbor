import pandas
import numpy
import matplotlib
from matplotlib import pyplot as py
import sklearn
from sklearn import preprocessing,model_selection,svm
from sklearn.neighbors import KNeighborsClassifier
df=pandas.read_excel(r'.\cardata.xlsx')
buying=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['buying']))
print(df['buying'])
maint=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['maint']))
doors=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['doors']))
persons=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['persons']))
lug_boot=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['lug_boot']))
safety=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['safety']))
classs=sklearn.preprocessing.LabelEncoder().fit_transform(list(df['classs']))
X=list(zip(buying,maint,doors,persons,lug_boot,safety))
y=list(classs)
xtrain, xtest, ytrain, ytest=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
model=svm.SVC()
model.fit(xtrain,ytrain)
acc=model.score(xtest,ytest)
prediction=model.predict(xtest)
names=['acc','good','unacc','vgood']
#for i in range(len(xtest)):
 #   print("Prediction = ", names[prediction[i]],"X_Values = ",xtest[i], "Actualvalues = ",names[ytest[i]])
  #  print(model.kneighbors([xtest[i]],9))
py.scatter(df['buying'],df['classs'])
py.show()
