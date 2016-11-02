#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:31:25 2016

@author: bharath
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew, boxcox
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
#import statsmodels.formula.api as smf
train_csv=pd.read_csv('train.csv')
test_csv=pd.read_csv('test.csv')
#Remove id from both train and test dataframes
train_csv=train_csv.iloc[:,1:]
ID=test_csv['id']
test_csv=test_csv.iloc[:,1:]

#Get continuous independent variables

split=116
cont_data=train_csv.iloc[:,split:]
cont_col_names=cont_data.columns
print("Number of continuous features: "+str(cont_data.shape[1]))
n_cols=2
n_rows=7

        
fg,ax=plt.subplots(nrows=1,ncols=1,figsize=(12,8))
sns.violinplot(y='loss',data=train_csv,ax=ax)
trainsplit=train_csv.shape[0]
'''
#Calculate the skew of continuous variables
cont_data=cont_data.drop(labels=['loss'],axis=1)
skewed_features=cont_data.apply(lambda x: skew(x))
print("skew in numerical features")
print(skewed_features)
#Correct the skew everywhere
skewed_features=skewed_features[skewed_features>0.25]
skewed_features=skewed_features.index
print("Index of skewed features")
print(skewed_features)
loss=train_csv['loss']
train_csv=train_csv.drop(labels=['loss'],axis=1)


train_test_combined=pd.concat((train_csv,test_csv)).reset_index(drop=True)
for feature in skewed_features:
    train_test_combined[feature]=train_test_combined[feature]+1
    train_test_combined[feature],lam=boxcox(train_test_combined[feature])
    

#Get back the training and testing dataframes with their continuous data unskewed and scaled
train_csv=train_test_combined.iloc[:trainsplit,:]
test_csv=train_test_combined.iloc[trainsplit:,:]

#Draw them again 
for i in range(n_rows):
    fg,ax=plt.subplots(nrows=1,ncols=n_cols,figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cont_col_names[i*n_cols+j],data=train_csv,ax=ax[j])


train_csv['loss']=loss
'''
#Look at correlation between continuous varaibles
plt.imshow(cont_data.corr(),cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
tick_marks=[i for i in range(len(cont_data.columns))]
plt.xticks(tick_marks,cont_data.columns,rotation='vertical')
plt.yticks(tick_marks,cont_data.columns)
plt.show()
#Unskew the loss column
train_csv['loss']=np.log1p(train_csv['loss'])
fgi,axi=plt.subplots(nrows=1,ncols=1,figsize=(12,8))
sns.violinplot(y='loss',data=train_csv,ax=axi)
plt.show()
#For categorical attributes
cat_data=train_csv.iloc[:,:split]
cat_col_names=cat_data.columns
print("Number of categorical features: "+str(cat_data.shape[1]))
n_rows=29
n_cols=4

#for i in range(n_rows):
#    fg,ax=plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12,8))
#    for j in range(n_cols):
#        sns.countplot(x=cat_col_names[i*n_cols+j],data=cat_data,ax=ax[j])
        
#One-hot encoding of categorical features
Labels=[]
for i in range(split):
    train=train_csv[cat_col_names[i]].unique()
    test=test_csv[cat_col_names[i]].unique()
    Labels.append(list(set(train) | set(test)))

#To store encoded features
train_cat=[]
test_cat=[]
    
for i in range(split):
    #Prepare the current categorical feature for OneHotEncoder which takes integers as input categories
    label_encoder=LabelEncoder()
    label_encoder.fit(Labels[i])
    train_feature=label_encoder.transform(train_csv.iloc[:,i])
    test_feature=label_encoder.transform(test_csv.iloc[:,i])
    train_feature=train_feature.reshape(train_csv.shape[0],1)
    test_feature=test_feature.reshape(test_csv.shape[0],1)
    #Use OneHotEncoder and append the newly generated features
    onehot_encoder=OneHotEncoder(sparse=False, n_values=len(Labels[i]))
    train_feature=onehot_encoder.fit_transform(train_feature)
    test_feature=onehot_encoder.fit_transform(test_feature)
    train_cat.append(train_feature)
    test_cat.append(test_feature)
    
#Convert cat from a list of 1D arrays to a 2D array
encoded_train_cats=np.column_stack(train_cat)
encoded_test_cats=np.column_stack(test_cat)



#Create new training dataset with the newly generated features
train_csv_encoded=np.concatenate((encoded_train_cats,train_csv.iloc[:,split:].values),axis=1)
train_df_encoded=pd.DataFrame(train_csv_encoded)
#Remove the loss column from the final train dataframe
labels=train_df_encoded.iloc[:,train_df_encoded.shape[1]-1]
train_df_encoded=train_df_encoded.iloc[:,:train_df_encoded.shape[1]-1]
#Create new testing dataset with the newly generated features
test_csv_encoded=np.concatenate((encoded_test_cats,test_csv.iloc[:,split:].values),axis=1)
test_df_encoded=pd.DataFrame(test_csv_encoded)


#Scale the features in both training and testing dataframes
train_test=pd.concat((train_df_encoded,test_df_encoded)).reset_index(drop=True)
#Try scaling the features in both training and testing
#train_test.iloc[:,:]=preprocessing.Normalizer().fit_transform(train_test.iloc[:,:])
train_test.iloc[:,:]=preprocessing.MinMaxScaler().fit_transform(train_test.iloc[:,:])
#train_test.iloc[:,:]=preprocessing.MaxAbsScaler().fit_transform(train_test.iloc[:,:])
#train_test.iloc[:,:]=preprocessing.StandardScaler().fit_transform(train_test.iloc[:,:])

#Get back the training and testing dataframes with their continuous data unskewed and scaled
train_df_encoded=train_test.iloc[:trainsplit,:]
test_df_encoded=train_test.iloc[trainsplit:,:]



Xtrain,Xval,ytrain,yval=train_test_split(train_df_encoded,labels,test_size=0.1,random_state=5)


pca=PCA(n_components=300)
pca.fit(Xtrain)
newXtrain=pca.transform(Xtrain)
newXval=pca.transform(Xval)
#model=linear_model.LinearRegression()
#model=linear_model.Ridge(alpha=20,random_state=7)
#reg=linear_model.RidgeCV(alphas=[0.1,1.0,10.0,20.0])
#reg.fit(newXtrain,ytrain)
#print('The alpha: '+str(reg.alpha_))
#model=RandomForestRegressor(n_jobs=-1,random_state=7,n_estimators=50)
model=XGBRegressor(n_estimators=1000,learning_rate=0.3)
model.fit(newXtrain,ytrain)
score=mean_absolute_error(np.expm1(yval),np.expm1(model.predict(newXval)))
print('The MAE score: '+str(score))





newXtest=pca.transform(test_df_encoded)


#Predictions
predictions=np.expm1(model.predict(newXtest))

#Create submission file
with open('submissions.csv','w') as subfile:
    subfile.write('id,loss\n')
    for i,pred in enumerate(list(predictions)):
        subfile.write('%s,%s\n'%(ID[i],pred))

