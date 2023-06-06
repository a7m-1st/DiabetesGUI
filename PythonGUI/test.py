from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from random import randint
import pandas as pd
from sklearn.model_selection import train_test_split

models = []

#UPload to github, then get the raw url
diab_data = pd.read_csv('https://raw.githubusercontent.com/a7m-1st/UniJava/main/L1Q2/diabetes.csv')
print(diab_data)
X = diab_data.drop(columns='Outcome', axis = 1) #Take the X as the sample output
y = diab_data['Outcome'] #to Map feature X to house prices y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)  #splits the input & output into testing and training parts

ran = randint(1, 100)
ran2 = randint(1, 100)
print(ran, ran2)
model = DecisionTreeClassifier(random_state=ran2)


model.fit(X_train, y_train) #instead you train with 80%
prediction = model.predict(X_test) #You test with 25%
models.append(model)


#Accuracy
score = accuracy_score(y_test, prediction)
print('DecisionTreeClassifier Accuracy Score es',(score*100).round(2))
score = accuracy_score(y_train, model.predict(X_train))
print('Training DecisionTreeClassifier Accuracy es',(score*100).round(2))

# print(X_train)
# X_test = [[2, 33, 44, 55, 33, 55, 44, 55]]
# print('Training DecisionTreeClassifier Accuracy es', model.predict(X_test))
