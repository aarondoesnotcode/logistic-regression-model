#target - 1 for heart disease, 0 for no heart disease
#classification problem - heart disease assessment, using features to predict whether a user has heart disease

#for saving model
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score

#reading the dataset
df = pd.read_csv("heart.csv")

# setting y variable to default - this is the label, 0 for no default, 1 for default
y = df['target']
#this is the features, what will be used to predict the y variable (label)
#axis = 1 -> refers to the columns 
X = df.drop('target', axis=1)

#splitting data into training & testing sets
#test_size -> refers to 25% of data going into test sets, 75% for training (this is default/standard)
#random_state = 42 -> controls randomness of data splitting (42 is occassionally used)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

#lbfgs - alg used to find best weights for model
classifier = LogisticRegression(solver='lbfgs',
                                max_iter=200,
                                random_state=42)


# standardised -> to make sure all numbers(features) in dataset are on same scale
# each feature(column) -> has mean = 0, s.d = 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #learns how to scale mean/s.d and applies to training data
#this uses the learned parameters (mean/s.d) from the previous line (fit_transform), but doesn't learn anything new -> because we're testing, we dont want the model to see anything about the test set  
#so it uses the mean/s.d from previous one, to new/unseen data
X_test = scaler.transform(X_test) 

#esentially creating an instance of the model
#then training the model on the training set
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

#saving the model and scaler after training
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

#making the prediction 
y_pred = model.predict(X_test)

#using this to find out the results, analyse them
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score", accuracy_score(y_test, y_pred))
print("ROC-AUC score", roc_auc_score(y_test,y_pred))
print("f1 score", f1_score(y_test,y_pred))

#shows no. of defaults and non defaults
print(df['Default'].value_counts())
