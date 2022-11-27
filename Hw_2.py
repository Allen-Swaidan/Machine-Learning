#%%
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import random
import math


#%%
data = pd.read_excel("heart.xlsx")

#dummy variable method
sex_dummy = pd.get_dummies(data.Sex)
chest_pain_dummy = pd.get_dummies(data.ChestPainType)
resting_ecg_dummy = pd.get_dummies(data.RestingECG)
exercise_angina_dummy = pd.get_dummies(data.ExerciseAngina)
st_slope_dummy = pd.get_dummies(data.ST_Slope)

#displaying dummmy variables in table
sex_dummy
chest_pain_dummy
resting_ecg_dummy
exercise_angina_dummy
st_slope_dummy

#merging dummy variables into the table
merged_table = pd.concat([data,sex_dummy, chest_pain_dummy, resting_ecg_dummy, exercise_angina_dummy, st_slope_dummy],axis='columns')
merged_table

data_table = merged_table.drop(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'], axis='columns')
data_table

# %%
#calculating the percentage of positve and negative values for heart disease 
def prior_prob(y_train, y):
    classes = sorted(list(y_train.unique()))             #sorting the array in ascending order and taking the unique values
    probability = []                                        #array to store the likelihoods of y=0 and y=1
    for i in classes:
        probability.append(len(y_train[y==i])/len(y_train)) #likelihood for negative and positive is length of zero's and one's seperately divided by the total length of the dataset 
    return probability                                      #returned likelihoods of negative and positive values of heart disease

#calculating the probability of x given y using the guassian distribution
def prob_x_given_y(x_train, feature_name, feature_value, y, label):
    rows = list(y==label)
    training_set = x_train[rows]
    mean = training_set[feature_name].mean()                #mean of training set
    sigma = training_set[feature_name].std()                #standard deviation of training set
    probability_x_given_y = 1/(np.sqrt(2*np.pi*sigma))*np.exp(-((feature_value-mean)**2)/(2*sigma**2)) #probability using guassian distribution
    return probability_x_given_y

def naive_bayes_func(x_train, x_test, y_train, y, y_test):
    features = list(x_train.columns)[:1]

    prior = prior_prob(y_train, y)
    prediction = []                             #empty array to store the precticted values

    for _, x_values in x_test.iterrows():       #looping over test set
        labels = sorted(list(y_train.unique()))
        likelihood = [1]*len(labels)

        for i in range(len(labels)):            
            for j in range(len(features)):
                likelihood[i] *= prob_x_given_y(x_train, features[j], x_values[j], y_train, labels[i])  #looping over the length of the labels and features to mulitiply all of the likelihoods

        posterior_prob = [1]*len(labels)
        for i in range(len(labels)):
            posterior_prob[i] = likelihood[i]*prior[i]      #multiplying all of the likelihoods with the prior probabilites

        prediction.append(np.argmax(posterior_prob))        #storing the predicted values in the array

    
    num_correct = sum(prediction==y_test)                   #number of correct predictions
    percent_correct = (num_correct/len(y_test)*100)         #percent of correct predictions
    return np.array(prediction), percent_correct

#euclidean_distance funtion
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2))**2)
    return distance

#could not get the knn classifer to work
def knn(k, x_train, x_test, y_train):
    predictions = []
    confidence = []
    for pred_row in x_test:
        euclidean_distances = []
        for x_row in x_train:
            distance = np.linalg.norm(x_row - pred_row, 2)
            euclidean_distances.append(distance)

        neighbors = y_train[np.argsort(euclidean_distances)[:k]]
        neighbors_bc = np.bincount(neighbors)
        prediction = np.argmax(neighbors_bc)
        predictions.append(prediction)
        confidence.append(neighbors[prediction]/len(neighbors))

    predictions = np.array(predictions)
    return predictions    

# %%
#Testing the model
#splitting training data for Naive Bayes Classifier
x = data_table[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','F','M','ASY','NAP','TA','LVH','Normal','ST','N','Y','Down','Flat','Up']]
y = data_table['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

prediction, percent_correct = naive_bayes_func(x_train, x_test, y_train, y, y_test)
#total,accuracy,precision,recall,f1_score = confusion_matrix(x_train, x_test, y_train, y, y_test)

print(f"Percent correct: {percent_correct}%")

print(f"Confusion matrix: {confusion_matrix(y_test, prediction)}")
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"precision: {precision_score(y_test, prediction)}")
print(f"recall: {recall_score(y_test, prediction)}")
print(f"f1 score: {f1_score(y_test, prediction)}")
# %%
#Splitting testing data for knn classifier, could not get this classifier to work
x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.8)
x_valid, x_test, y_valid, y_test = train_test_split(x_rem,y_rem, test_size=0.5) #splitting instances into 60:20:20
knn(5, x_train, x_test, y_train)
# %%
