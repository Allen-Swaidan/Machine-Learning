#Hw_1
'''
This algorithm predicted anywhere from a 54%-67% accuracy depending on the run. Every few runs there is an error where the number
of true positive and false positives equals zero which causes my prediction function to become undefined. I couldnt figure out why
this was happening so if you get that error, I suggest running the code again. 
'''
#%%
import math
from re import X
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import random

#%%
#Reading in file data
file = pd.read_excel("heart.xlsx")
file

#Converting catergorical features into dummy/indicator variables
sex_dummy = pd.get_dummies(file.Sex)
chest_pain_dummy = pd.get_dummies(file.ChestPainType)
resting_ecg_dummy = pd.get_dummies(file.RestingECG)
exercise_angina_dummy = pd.get_dummies(file.ExerciseAngina)
st_slope_dummy = pd.get_dummies(file.ST_Slope)

#displaying dummmy variables in table
sex_dummy
chest_pain_dummy
resting_ecg_dummy
exercise_angina_dummy
st_slope_dummy

#merging dummy variables into the table
merged_table = pd.concat([file,sex_dummy, chest_pain_dummy, resting_ecg_dummy, exercise_angina_dummy, st_slope_dummy],axis='columns')
merged_table

#dropping the text categorical features
final_table = merged_table.drop(['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'], axis='columns')
final_table

#%%
#Sigmoid func
def sigmoid(z):
    sigma = 1/(1+(math.e)**(-z))
    return sigma

#Regression function
def regression_func(w, b, x, y_train,learning_rate,iteration_number):
    for iteration in range(iteration_number):       #looping over a set amount of iterations
        deltas = np.zeros(18)                       #creating a vector of weights for each feature initialized to zero
        for rows,y in zip(x,y_train):               #looping over training set
            sum = 0
            for weight,xi in zip(w,rows):           #looping over each row with the assoicated weight to calculate the sum of each weight multiplied by element in that row
               sum = sum + weight*xi

            output = sigmoid(sum) + b               #using the result of sum to calculate the sigmoid function
            distance = y-output                     #difference between the actual value and predicted value
            for i,xi in enumerate(rows):            #calculating the delta formula
                deltas[i] = deltas[i] + learning_rate*xi*(distance)
            
        w = w + deltas                              #updating the weights
    return w,b

#this function uses the weights obtained from the regression function to make a prediction
def prediction(x, y, w, b):
    total = 0 
    tp = 0
    tn = 0 
    fp = 0 
    fn = 0
    for rows,yi in zip(x,y):
        sum = 0
        for weight,xi in zip(w,rows):
           sum = sum + weight*xi
           
        output = sigmoid(sum) + b

        #calculating TP,TN,FP,FN, the confusion matrix
        if(output > .5 and yi == 1):    #number of true positives
            total += 1
            tp += 1
        elif(output < .5 and yi == 0):  #number of true negatives
            total += 1
            tn += 1
        elif(output > .5 and yi == 0):  #number of false positives
            fp += 1
        elif(output < .5 and yi == 1):  #number of false negatives
            fn += 1

    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score =  2*((precision*recall)/(precision+recall))

    return total,accuracy,precision,recall,f1_score

    
#%%
#splitting training and test sets
x = final_table[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','F','M','ASY','NAP','TA','LVH','Normal','ST','N','Y','Down','Flat','Up']]
y = final_table['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2)

#%%
#Randomizing weights and initializing a learning rate and iteration number
w = np.random.randn(18)
b = np.random.randn()
learning_rate = 0.0001
iteration_number = 10000

trained_weights, trained_bias = regression_func(w, b, x_train.to_numpy(), y_train.to_numpy(),learning_rate, iteration_number)   #calling the algorithm for training
total,accuracy,precision,recall,f1_score = prediction(x_test.to_numpy(),y_test.to_numpy(),trained_weights,trained_bias)         #making a prediciton

#printing the confusion matrix, also printed the number of correct predictions represented as total
total_percent = (total/183)*100
print(f"Total Correct = {total}")
print(f"Total Wrong = {183-total}")
print(f"Total% = {(total/183)*100}")

print(f"Accuracy = {accuracy}")
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"f1_score = {f1_score}")

# %%
