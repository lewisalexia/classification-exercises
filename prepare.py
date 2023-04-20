# IMPORTS
import pandas as pd 
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from pydataset import data

import env
import os

# import splitting functions
from sklearn.model_selection import train_test_split

# * from pydataset import data
# * mpg = data('mpg')
# * mpg.head()




def clean_iris(df):
    '''
    This function will clean the iris dataset
    '''
    df = df.drop(columns =['species_id', 'measurement_id'])

    df.rename({'species_name':'species'}, axis=1, inplace=True)

    dummy_df = pd.get_dummies(df.species)
    
    clean_iris_df = pd.concat([df, dummy_df], axis=1)
    
    return clean_iris_df





def clean_titanic(df):
    '''
    This function will clean the titanic dataset
    '''
    df = df.drop(columns =['embarked','class','age','deck'])

    df.embark_town = df.embark_town.fillna(value='Southampton')

    dummy_df = pd.get_dummies(df[['sex','embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df




def clean_telco(df):
    '''This function will clean the telco_churn data'''
    
    #convert total charges to float
    df.total_charges = pd.to_numeric(df['total_charges'], errors='coerce')

    # fill n/a values in total_charges
    df.total_charges.fillna(0)
    
    # dropping columns
    df = df.drop(columns=['contract_type_id','internet_service_type_id','payment_type_id'])

    # encoding
    df.gender = pd.get_dummies(df[['gender']], drop_first=True)

    df.partner = df.partner.replace('Yes',1).replace('No',0)

    df.dependents = df.dependents.replace('Yes',1).replace('No',0)

    df.phone_service = df.phone_service.replace('Yes',1).replace('No',0)

    df.online_security = df.online_security.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.online_backup = df.online_backup.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.device_protection = df.device_protection.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.tech_support = df.tech_support.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.streaming_tv = df.streaming_tv.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.streaming_movies = df.streaming_movies.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.paperless_billing = df.paperless_billing.replace('Yes',1).replace('No',0)

    df.churn = df.churn.replace('Yes',1).replace('No',0)

    df.multiple_lines = df.multiple_lines.replace('No phone service',0).replace('Yes',1).replace('No',0)

    # get dummies for categoricals
    dummy_df = pd.get_dummies(df[['contract_type','payment_type','internet_service_type']], dummy_na=False, drop_first=[True, True, True])

    # clean up and return final product
    df = pd.concat([df, dummy_df], axis=1)

    return df




# SPLITTING FUNCTION


def split_iris(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on selected column.
    return train, validate, test DataFrames.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train.species)
    return train, validate, test


def split_titanic(df):
    '''
    Takes in the titanic dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df.survived)
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train.survived)
    return train, validate, test




def split_telco(df):
    '''
    Takes in the telco dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df.churn)
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train.churn)
    return train, validate, test






def split_function(df, target_variable):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[target_variable])
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[target_variable])
    return train, validate, test
