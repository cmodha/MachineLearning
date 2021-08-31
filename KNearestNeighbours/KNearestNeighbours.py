import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import pprint
from termcolor import colored
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
#running the model multiple times for each odd number between 1 and 39 
#and saving the scores and the best model
allAcc = []
n = np.linspace(1,39,20).astype(int)
bestAcc = 0
for number in n:
    #splitting the data into test and train samples
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.2)
    #initialising the model as well as the number of neighbors
    model = KNeighborsClassifier(n_neighbors = number)
    #fitting the model
    model.fit(x_train,y_train)
    #scoring the model
    acc = model.score(x_test,y_test)
    #testing to see whether the accuracy is the best accuracy
    # and saving the model and number of neighbors if it is
    if acc > bestAcc:
        bestAcc = acc
        numOfNeighbors = number
        with open("carmodel.pickle","wb") as f:
            pickle.dump(model,f)
    #storing the accuracy of each model
    allAcc.append(acc*100)
#opening the best model using pickle    
pickeIn = open("carmodel.pickle","rb")
bestModel = pickle.load(pickeIn)
#printing the best accuracy alongside the number of neighbors for that model
print(f"Highest Accuracy is {bestAcc*100}% with {numOfNeighbors} neighbors")
#refitting the best model
bestModel.fit(x_train,y_train)
#predicting the the test data results
predicted = bestModel.predict(x_test)
#all the positions
names = ["DF","DFFW","DFMF","FW","FWDF","FWMF","GK","MF","MFDF","MFFW"]
#printing the predicted results alongisde the actual position
for x in range(len(predicted)):
    print(f"Predicted Position:    {names[predicted[x]]}    Actual Position:    {names[y_test[x]]}")
    if y_test[x] != predicted [x]:
        noMatch = colored("No Match!",'red')
        print(noMatch)
   
#showing the graph of accuracy against number of neighbors.
style.use("ggplot")
pyplot.scatter(n,allAcc)
pyplot.xlabel("Number of Neighbors")
pyplot.ylabel("Accuracy (%)")
pyplot.title("Graph of Accuracy against number of neighbors")
pyplot.show()