import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#read dataset
titanic_df=pd.read_csv("Titanic-Dataset.csv")
print(titanic_df)
print(titanic_df.info())  #dataset information

rec=titanic_df.head(15)  #first 15 entries
print(rec)

print(titanic_df.describe())

print(titanic_df[['Pclass', 'Survived']])

#cleaning data
titanic_df.drop('Cabin', axis=1,inplace=True)
titanic_df.drop('Embarked', axis=1,inplace=True)
titanic_df.drop('PassengerId', axis=1,inplace=True)
print(titanic_df.info())

indexAge=titanic_df[titanic_df['Age'].isnull()].index

#titanic_df['Age']=titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df.drop(indexAge,inplace=True)
titanic_df.replace('male',0,inplace=True)
titanic_df.replace('female',1,inplace=True)
print(titanic_df.info())

#input-output separation
X=titanic_df[['Pclass','Sex','Age', 'SibSp','Parch','Fare']]
y=titanic_df['Survived']  #target column

#split data into training and testing
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=87)

print("Training samples: ", len(X_train))
print("Testing samples: ", len(X_test))

#Create a model
model=RandomForestClassifier()

#train the model
model.fit(X_train, y_train)

#make prediction
prediction=model.predict(X_test)

#calculate accuracy of model
accuracy=accuracy_score(y_test,prediction)

print("Predictions")
print(prediction)

print("Accuracy: ", accuracy)
print("Accuracy Percentage: ", accuracy*100)

