# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 01:30:22 2020

@author: User
"""
# import all libraries that are needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# code for a custom scaler class
class CustomScaler (BaseEstimator, TransformerMixin):
	def __init__ (self, columns, copy = True, with_mean = True, with_std = True):
		self.scaler = StandardScaler(copy, with_mean, with_std)
		self.columns = columns
		self.mean_ = None
		self.var_ = None

	def fit(self, X, y = None):
		self.scaler.fit(X[self.columns], y)
		self.mean_ = np.array(np.mean(X[self.columns]))
		self.var_ = np.array(np.var(X[self.columns]))
		return self

	def transform(self, X, y = None, copy = None):
		init_col_order = X.columns
		X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
		X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
		return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]

class absenteeism_module():
    def __init__(self, model_file, scaler_file):
        #reading the pickled file
        with open("model", "rb") as model_file, open("scaler", "rb") as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
    
    def load_and_clean_data(self, data_file):
        #reading the csv file
        df = pd.read_csv(data_file, delimiter = ",")
        self.df_with_predictions = df.copy()
        #droping the "ID" column
        df = df.drop(["ID"], axis = 1 )
        #note this step
        df["Absenteeism Time in Hours"] = "NaN"
        
        #working on values in "Reason of Absence"
        maxctr = len(df["Reason for Absence"])
        
        for i in range(maxctr):
            df.loc[df['Reason for Absence'] == 28, 'Reason_28'] = 1
            df.loc[df['Reason for Absence'] == 27, 'Reason_27'] = 1
            df.loc[df['Reason for Absence'] == 26, 'Reason_26'] = 1
            df.loc[df['Reason for Absence'] == 25, 'Reason_25'] = 1
            df.loc[df['Reason for Absence'] == 24, 'Reason_24'] = 1
            df.loc[df['Reason for Absence'] == 23, 'Reason_23'] = 1
            df.loc[df['Reason for Absence'] == 22, 'Reason_22'] = 1
            df.loc[df['Reason for Absence'] == 21, 'Reason_21'] = 1
            df.loc[df['Reason for Absence'] == 20, 'Reason_20'] = 1
            df.loc[df['Reason for Absence'] == 19, 'Reason_19'] = 1
            df.loc[df['Reason for Absence'] == 18, 'Reason_18'] = 1
            df.loc[df['Reason for Absence'] == 17, 'Reason_17'] = 1
            df.loc[df['Reason for Absence'] == 16, 'Reason_16'] = 1
            df.loc[df['Reason for Absence'] == 15, 'Reason_15'] = 1
            df.loc[df['Reason for Absence'] == 14, 'Reason_14'] = 1
            df.loc[df['Reason for Absence'] == 13, 'Reason_13'] = 1
            df.loc[df['Reason for Absence'] == 12, 'Reason_12'] = 1
            df.loc[df['Reason for Absence'] == 11, 'Reason_11'] = 1
            df.loc[df['Reason for Absence'] == 10, 'Reason_10'] = 1
            df.loc[df['Reason for Absence'] == 9, 'Reason_09'] = 1
            df.loc[df['Reason for Absence'] == 8, 'Reason_08'] = 1
            df.loc[df['Reason for Absence'] == 7, 'Reason_07'] = 1
            df.loc[df['Reason for Absence'] == 6, 'Reason_06'] = 1
            df.loc[df['Reason for Absence'] == 5, 'Reason_05'] = 1
            df.loc[df['Reason for Absence'] == 4, 'Reason_04'] = 1
            df.loc[df['Reason for Absence'] == 3, 'Reason_03'] = 1
            df.loc[df['Reason for Absence'] == 2, 'Reason_02'] = 1
            df.loc[df['Reason for Absence'] == 1, 'Reason_01'] = 1
        # creating groups of reasons
        reason_group1 = df[['Reason_14','Reason_13','Reason_12','Reason_11',
                            'Reason_10','Reason_09','Reason_08','Reason_07',
                            'Reason_08','Reason_07','Reason_06','Reason_05',
                            'Reason_04','Reason_03','Reason_02','Reason_01']].max(axis = 1)
        reason_group2 = df[['Reason_17','Reason_16','Reason_15']].max(axis = 1)
        reason_group3 = df[['Reason_21','Reason_20','Reason_19','Reason_18']].max(axis = 1)
        reason_group4 = df[['Reason_28','Reason_27','Reason_26','Reason_25',
                            'Reason_24','Reason_23','Reason_22']].max(axis = 1)
        columns = ['Reason for Absence','Date','Transportation Expense',
                 'Distance to Work','Age','Daily Work Load Average',
                 'Body Mass Index','Education','Children','Pets',
                 'Absenteeism Time in Hours']
        df = df[columns]
        df = pd.concat([df, reason_group1, reason_group2, reason_group3, reason_group4], axis = 1)
        #droping the "REason for Absence" column
        df = df.drop(["Reason for Absence"], axis = 1)
        
        # working with ["Date"]
        
        df["Date"] = pd.to_datetime(df["Date"], format = "%d/%m/%Y")
        #Extracting months
        list_months = []
        counter_max = df.shape[0]
        for i in range(counter_max):
            list_months.append(df["Date"][i].month)
        df["Month_Value"] = list_months
        #Extracting day of the week
        df["Weekday"] = df["Date"].apply(lambda x: x.weekday())
        #dropping the date
        df = df.drop(["Date"], axis = 1)
        
        #Mapping the education
        
        df["Education"] = df["Education"].map({1:0, 2:1, 3:1, 4:1})
        
        # droping the unnecessary columns which have lower weights
        
        df = df.drop(["Body Mass Index", "Education", "Distance to Work",
                      "Month_Value", "Daily Work Load Average"], axis = 1)
        
        # note that we are dropping the ["Absenteeism Time in Hours"]
        df = df.drop(["Absenteeism Time in Hours"], axis = 1)
        
        columns = ["Transportation Expense", "Age", "Children", "Pets",
                   "Reasons_Disease", "Reasons_Pregnancy", "Reasons_Extrenous", 
                   "Reasons_Soft", "Weekday"]
        df.columns = columns
        
        columns_reordered = ["Reasons_Disease", "Reasons_Pregnancy", "Reasons_Extrenous", 
                             "Reasons_Soft", "Weekday", "Transportation Expense",
                             "Age", "Children", "Pets"]
        df = df[columns_reordered]
        
        # setting NA values to 0
        
        df = df.fillna(value = 0)
        self.preprocessed_data = df.copy()
        
        #standardizing the data
        self.data = self.scaler.transform(df)
        
# a method that outputs the probability
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

# a method which will predict 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

# adding probabilities and predictions to the preprocessed data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data["Probability"] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data["Prrdiction"] = self.reg.predict(self.data)
            return self.preprocessed_data

