# IMPORTS
import pandas as pd 
import numpy as np
from scipy import stats
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


# BASE FUNCTIONS

print(f'Load in successful, awaiting commands...')

def get_connection(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'



def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 





def get_titanic_data():
    url = env.get_connection('titanic_db')
    filename = 'titanic.csv'
    query = 'select * from passengers'

    df = check_file_exists(filename, query, url)

    return df 






def get_iris_data():
    url = env.get_connection('iris_db')
    query = '''
            select * from measurements
                join species
                    using (species_id)
            '''
    filename = 'iris.csv'
    df = check_file_exists(filename, query, url)
    return df






def get_telco_churn():
    url = env.get_connection('telco_churn')
    query = ''' select * from customers
	join contract_types
		using (contract_type_id)
	join internet_service_types
		using (internet_service_type_id)
	join payment_types
		using (payment_type_id)
        '''
    filename = 'telco_churn.csv'
    df = check_file_exists(filename, query, url)

    return df



# TOWER FUNCTIONS

# def get_titanic_data(SQL_query, directory, filename="titanic.csv"):
#     """
#     This function will:
#     - Check local directory for csv file
#         - return if exists
#     - IF csv does not exist
#         - create a df of the SQL_query
#         - write df to csv
#     - Return titanic df
#     """
#     if os.path.exists(directory + filename):
#         df = pd.read_csv(filename)
#         return df
#     else:
#         df = new_titanic_data(SQL_query)
#         df.to_csv(filename)
#         return df
    



# def get_iris_data(SQL_query, directory, filename="iris.csv"):
#     """
#     This function will:
#     - Check local directory for csv file
#         - return if exists
#     - IF csv does not exist
#         - create a df of the SQL_query
#         - write df to csv
#     - Return iris df
#     """
#     if os.path.exists(directory + filename):
#         df = pd.read_csv(filename)
#         return df
#     else:
#         df = new_iris_data(SQL_query)
#         df.to_csv(filename)
#         return df
    






# def get_telco_data(SQL_query, directory, filename="telco.csv"):
#     """
#     This function will:
#     - Check local directory for csv file
#         - return if exists
#     - IF csv does not exist
#         - create a df of the SQL_query
#         - write df to csv
#     - Return telco df
#     """
#     if os.path.exists(directory + filename):
#         df = pd.read_csv(filename)
#         return df
#     else:
#         df = new_telco_data(SQL_query)
#         df.to_csv(filename)
#         return df
    

# CLEANING AND SPLITTING FUNCTIONS IN PREPARE FILE

# STATS CONCLUSIONS FUNCTIONS IN STATS_CONCLUDE FILE


def chi2_test(table):
    α = 0.05
    chi2, pval, degf, expected = stats.chi2_contingency(table)
    print('Observed')
    print(table.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p-value = {pval:.4f}')
    print('----')
    if pval < α:
        print ('We reject the null hypothesis.')
    else:
        print ("We fail to reject the null hypothesis.")





def conclude_1samp_tt(group1, group_mean):
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if ((p < α) & (tstat > 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_1samp_gt(group1, group_mean):
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if ((p / 2) < α) and (tstat > 0):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')





def conclude_1samp_lt(group1, group_mean):
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f't-stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if ((p / 2) < α) and (tstat < 0):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_2samp_tt(sample1, sample2):
    α = 0.05
    stat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if p < α:
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')




def conclude_2samp_gt(sample1, sample2):
    α = 0.05
    stat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if (((p/2) < α) and (tstat > 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_2samp_lt(sample1, sample2):
    α = 0.05
    stat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(p)
    print('\n----')
    if (((p/2) < α) and (tstat < 0)):
        print("we can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')





def conclude_anova(theoretical_mean, group1, group2):
    α = 0.05
    tstat, pval = stats.f_oneway(theoretical_mean, group1, group2)
    print(f'stat')
    print(tstat)
    print(f'P-Value')
    print(pval)
    print('----')
    if pval < α:
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')






def conclude_pearsonr(floats1, floats2):
    α = 0.05
    r, p = stats.pearsonr(floats1, floats2)
    print('r =', r)
    print('p =', p)
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")




def conclude_spearmanr(floats1, floats2):
    α = 0.05
    r, p = stats.spearmanr(floats1, floats2)
    print('r =', r)
    print('p =', p)
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")