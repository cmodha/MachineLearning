import pandas as pd
import sklearn 
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from termcolor import colored, cprint

#All data sourced from fbref - https://fbref.com/en/comps/9/stats/Premier-League-Stats
#importing data from csv file and labelling
data = pd.read_csv("playerstats.csv", sep = ',')
data = data[["Rk","Player","Nation","Pos","Squad","Age","Born","MP","Starts","Min","90s","Gls","Ast","G-PK","PK","PKatt","CrdY","CrdR","Glsp90","Astp90","G+Ap90","G-PKp90","G+A-PKp90","xG","npxG","xA","npxG+xA","xGp90","xAp90","xG+xAp90","npxGp90","npxG+xAp90"]]
#dropping labels which are not needed
dropped_labels = ["Rk","Player","Nation","Born","MP","Starts","Min","90s","Gls","Ast","G-PK","xG","npxG","xA","npxG+xA"]
data.drop(dropped_labels,axis = 1, inplace = True)
#printing out the head of the data for debug purposes
print(data.head())

#initialising label encoder
le = preprocessing.LabelEncoder()

#turning all labels into ints(this is mostly only needed for the position as all other labels are already numerical)
pos = le.fit_transform(list(data["Pos"]))
Age = le.fit_transform(list(data["Age"]))
Squad = le.fit_transform(list(data["Squad"]))
PK = le.fit_transform(list(data["PK"]))
PKatt  = le.fit_transform(list(data["PKatt"]))
CrdY = le.fit_transform(list(data["CrdY"]))
CrdR = le.fit_transform(list(data["CrdR"]))
Glsp90 = le.fit_transform(list(data["Glsp90"]))
Astp90 = le.fit_transform(list(data["Astp90"]))
GplusAp90 = le.fit_transform(list(data["G+Ap90"]))
GminusPKp90 = le.fit_transform(list(data["G-PKp90"]))
GplusAminusPKp90 = le.fit_transform(list(data["G+A-PKp90"]))
xGp90 = le.fit_transform(list(data["xGp90"]))
xAp90 = le.fit_transform(list(data["xAp90"]))
xGplusxAp90 = le.fit_transform(list(data["xG+xAp90"]))
npxGp90 = le.fit_transform(list(data["npxGp90"]))
npxGplusxAp90 = le.fit_transform(list(data["npxG+xAp90"]))

#zipping the other variables into a single list
x = list(zip(PK,PKatt,CrdY,CrdR,Glsp90,Astp90,GplusAp90,GminusPKp90,GplusAminusPKp90,xGp90,xAp90,xGplusxAp90,npxGp90,npxGplusxAp90))
#creating a list of the encoded positions
y = list(pos)

xTrain,xTest,yTrain,yTest = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)
classes = ["DF","DFFW","DFMF","FW","FWDF","FWMF","GK","MF","MFDF","MFFW"]

# print(xTrain)
# print(yTrain)
#initialising SVM model
clf = svm.SVC(kernel = "poly",degree = 2,gamma = "scale", C=2)
clf.fit(xTrain,yTrain)
#initialising the KNeighbors model
# kNeighbors = KNeighborsClassifier(n_neighbors = 9)
# kNeighbors.fit(xTrain,yTrain)
#generating predictions and scoring metrics and displaying them to the console
yPrediction = clf.predict(xTest)
# yKPrediction = kNeighbors.predict(xTest)
# kAcc = metrics.accuracy_score(yTest,yKPrediction)
svmAcc = metrics.accuracy_score(yTest,yPrediction)
print(f"The accuracy of the SVM model is: {100*svmAcc}%")

#printing predictions and making it clear when the predictions and the test data do not match
for n in yPrediction:
    print(f"The prediction of the SVM model was: {classes[yPrediction[n]]}  The actual classification was: {classes[yTest[n]]}")
    if yPrediction[n] != yTest[n]:
        nonMatch = colored("NON MATCH!","red")
        print(nonMatch)
