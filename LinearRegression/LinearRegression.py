import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read in the data to a pandas data frame
data = pd.read_csv("student-mat.csv",sep = ";")
#Split data into the columns that we are interested in
data = data[["G1","G2","G3","studytime","failures","absences","age"]]
#Declare which label we want to test
predict = "G3"
#create an array which is all the data minus the prediction column
x = np.array(data.drop([predict],1))
#y is the prediction column
y = np.array(data[predict])
best_acc = 0
#Randomly splitting the x and y data into test and train data sets
#with the size of the training set set to 0.1

#running the model 30 times and saving the model with the best score
for i in range(30):
    linear = linear_model.LinearRegression()
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print("Current Best Accuracy: ",best_acc,"Current Sim accuracy: ",acc)
    if acc > best_acc:
        best_acc = acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)
#loading the saved model
pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
acc = linear.score(x_test,y_test)
#printing the accuracy of the best data set
print("Percentage accuracy: %",100*acc)
#Declaring an array of prediction results
predictions = linear.predict(x_test)
#initialising variance
variance = 0
#for loop printing the predictions and the actual values
#and also calculating the variance
for x in range(len(predictions)):
    print(predictions[x],y_test[x])
    variance = variance +(abs(predictions[x]-y_test[x]))
print("Total Variance: ", variance)
#setting the plotting label
p = "G1"
#plotting the graph
style.use("fivethirtyeight")
pyplot.scatter(data[p],data[predict])
pyplot.xlabel(p.upper())
pyplot.ylabel("Final Grade")
pyplot.title("Graph to show Final Grade against First Sem Grade")
pyplot.show()
