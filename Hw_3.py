#%%
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

#%%
#loading data
train_set = pd.read_excel("mnist_test.xlsx")
test_set = pd.read_excel("mnist_train.xlsx")

#%%
#Splitting the training data set
df_x = train_set.iloc[:,1:]
df_y = train_set.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
#resplitting the data for a validation set
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size=0.2, random_state=4)

#splitting the testing data set
df_tx = test_set.iloc[:,1:]
df_ty = test_set.iloc[:,0]
tx_train, tx_test, ty_train, ty_test = train_test_split(df_tx, df_ty,test_size=0.01,random_state=4)

#%%
#logistic regression on original data set
reg = LogisticRegression() 
reg.fit(x_train,y_train)
print(f"Regression score = {reg.score(tx_test,ty_test)}")

#naive bayes classifier on original data set
nb = GaussianNB()                    
nb.fit(x_train, y_train)
print(f"Naive Bayes score = {nb.score(tx_test, ty_test)}")

#knn classifier on original data set
knn = KNeighborsClassifier()            
knn.fit(x_train, y_train)
print(f"Knn classifier score = {knn.score(tx_test, ty_test)}")

#%%
#scaling the data using MinMaxScaler
scalar = MinMaxScaler()                 
scaled_x_train = scalar.fit_transform(x_train)
scaled_x_test = scalar.fit_transform(x_test)
scaled_tx_train = scalar.fit_transform(tx_train)
scaled_tx_test = scalar.fit_transform(tx_test)

#performing SVD
svd = TruncatedSVD(n_components=200, n_iter=10, random_state=42) 
new_x_train = svd.fit_transform(scaled_x_train)
new_x_test = svd.fit_transform(scaled_x_test)
new_tx_train = svd.fit_transform(scaled_tx_train)
new_tx_test = svd.fit_transform(scaled_tx_test)

#%%
#plotting the number of principle components vs proportion of variance explained
plt.figure(num=0,dpi=120)
plt.plot(svd.explained_variance_ratio_)
plt.title("Proportion of Variance Explained Vs. Number of Principle Components")
plt.ylabel("Prop of Var")
plt.xlabel("Principle Components")

#%%
#logisting regresion on reduced dimension data set
reg.fit(new_x_train,y_train)
print(f"Regression score after SVD = {reg.score(new_tx_test,ty_test)}")

#naive bayes classifier on reduced dimension data set
nb.fit(new_x_train,y_train)
print(f"Naive Bayes score after SVD = {nb.score(new_tx_test,ty_test)}")

#knn classifer on reduced dimension data set
knn.fit(new_x_train,y_train)
print(f"KNN Classifier score after SVD = {knn.score(new_tx_test,ty_test)}")

# %%
'''
I am not sure why the accuracies are so low but naive bayes seemed to perform the best followed by KNN then linear regression.

'''