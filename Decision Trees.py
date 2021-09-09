# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 21:31:36 2021

@author: Daniel Idan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def missing_categorical_features(x_train, y_train, x_test):
    
    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)
    y_pred = logmodel.predict(x_test)
    return y_pred

def missing_value_features(x_train, y_train, x_test):
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred
    
def nan_include_features(df):
    ''' Find Categories with mising values'''
    z = df.isna().sum(axis=0) != 0 
    z = np.trim_zeros(z.sort_values())
    return list(z.index)

def nan_includ_rows(df5, num_of_nan_in_raw=1):
    ''' DataFrame include only rows with nan value at a spesific count'''
    zz = pd.DataFrame(df5.isna().sum(axis=1))
    zz = zz[ zz <= num_of_nan_in_raw ].dropna()
    zz = df5.loc[zz.index]
    return zz

def enc_class_features(df9, Nan_class_features, nonan_class_feature):
    labele_encoded_features = nonan_class_feature + Nan_class_features
    labled_df = pd.DataFrame()
    enc_param_storage = {}
    for ftr in labele_encoded_features:
        enc = preprocessing.LabelEncoder()
        enc_param_storage[ftr] = [enc.get_params]
        print(ftr)
        if ftr == 'Dependents' : ### String Features with graduate importance:
            enc.fit(['0','1','2','3+'])

        else :
            enc.fit(df9[ftr].dropna())
            
        df1 = pd.DataFrame( enc.fit_transform(df9[ftr].dropna()) ,\
                        index =list(df9[ftr].dropna().index) , columns = [ftr])
            
        labled_df = pd.concat([labled_df, df1], axis=1)

            
        enc_param_storage[ftr] = [enc.get_params]
        
    return labled_df, enc_param_storage
            
# load the data
train_dataframe = pd.read_csv('train_ctrUa4K.csv', index_col='Loan_ID')
test_dataframe = pd.read_csv('test_lAUu6dG.csv', index_col='Loan_ID')
dataframe = pd.concat([train_dataframe, test_dataframe], axis=0)

# visualization
sns.heatmap(dataframe.isna(), yticklabels=False, cbar=False)

# Parse the features
nan_features = nan_include_features(dataframe)
nan_value_features = ['LoanAmount', 'Loan_Amount_Term']
NoNan_class_feature = ['Education', 'Loan_Status']
nan_class_features = ['Gender','Married','Dependents','Self_Employed', 'Credit_History']
final_y_feature = ['Loan_Status']

## Encoding categorical features:
### Encoding (binary / value importance) class Features  :
cat_inf = enc_class_features(dataframe, nan_class_features, NoNan_class_feature)
cat_b_df = pd.concat([cat_inf[0],dataframe[['LoanAmount', 'Loan_Amount_Term','Property_Area',\
                     'ApplicantIncome' , 'CoapplicantIncome']]], axis=1)
### Encoding get_dummies class Features  :
encoded_df = pd.get_dummies(cat_b_df, columns=['Property_Area'], drop_first=True)

Train_df = encoded_df.loc[train_dataframe.index]
Test_df = encoded_df.loc[test_dataframe.index]
Test_df['Loan_Status'] = 1
encoded_df = nan_includ_rows( pd.concat([Train_df,Test_df]) ,1)

# We will deal the missing values by using prediction models:
## Parse the data 
Nan_df = encoded_df[encoded_df.isna().any(axis=1)]
NoNan_df = encoded_df.dropna()

## Complete Nan Values:
Nan_fill_df = pd.DataFrame()
for category in nan_include_features(Nan_df):
    print(category)
    order_category = encoded_df[encoded_df[category].isna()]
    x_test = order_category.drop([category] , inplace=False, axis=1)
    x_train = NoNan_df.drop([category] , inplace=False, axis=1)
    y_train = NoNan_df[category]
    if category in nan_class_features:
        na_val = missing_categorical_features(x_train, y_train, x_test)
    if category in nan_value_features:
        na_val = missing_value_features(x_train, y_train, x_test)
    if category in final_y_feature:
        continue
    order_category[category] = pd.DataFrame(na_val,index=order_category.index)
    Nan_fill_df = pd.concat([Nan_fill_df,order_category])
    
## Concat all the data (first missing block & full detailed block)
full_train_df = pd.concat([NoNan_df, Nan_fill_df])
## Seperate to test(prediction)/train dataframes:
Train_df = full_train_df.loc[train_dataframe.index]
Test_df = full_train_df.loc[test_dataframe.index]

### Inside model test:
# Seperate X & Y
X = full_train_df.drop(final_y_feature, inplace=False, axis=1)
Y = full_train_df[final_y_feature[0]]

## Random forest model :
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

### Model test results:
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

###prediction:

