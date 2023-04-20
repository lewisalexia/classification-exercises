import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
import acquire as acq

import warnings
warnings.filterwarnings("ignore")

# I am trying to find the most significant categorical variables to send in



# def set_explore_columns(df):
#     '''This function is meant to split the df into categorical and numerical columns. The 
#     default split is more than 3 unique variables makes it a numerical column. Less than 3 indicates 
#     "buckets" which indicates categorical variables.

#     Use the "explore_categorical/numerical" function to further explore the variables.
#     '''
#     col_cat = []  #this is for my categorical varibles
#     col_num = [] #this is for my numeric varibles
#     for col in df.columns:
#         print(col) 
#         if len(df[col].unique()) < 3: # making anything with less than 3 unique variables a categorical
#             col_cat.append(col)
#         else:
#             col_num.append(col)



# # IN WORK! BREAKS AT TARGET....
# def explore_categoricals(df, target):
#     for col in col_cat:
#         print()
#         print(col.upper())
#         print(df[col].value_counts())
#         print(df[col].value_counts(normalize=True))
#         df[col].value_counts().plot.bar()
#         plt.show()
#         print()
#         print()
#         print(f'HYPOTHESIZE')
#         print(f"H_0: {col.lower().replace('_',' ')} does not affect {target}")
#         print(f"H_a: {col.lower().replace('_',' ')} affects {target}")
#         print()
#         print(f'VISUALIZE')
#         sns.barplot(x=df[col], y=df[{target}])
#         plt.title(f"{col.lower().replace('_',' ')} vs {target}")
#         plt.show()
#         print()
#         print('ANALYZE and SUMMARIZE')
#         observed = pd.crosstab(df[col], df[{target}])
#         acq.chi2_test(observed)
#         print()
#         print()


# # IN WORK!
# def explore_numericals(df)
#     for col in col_num:
#         sns.barplot(data=train, x='survived', y=col)
#         plt.title(f"Is survived independent of {col.lower().replace('_',' ')}?")
#         pop_mn = train[col].mean()
#         plt.axhline(pop_mn, label=(f"{col.lower().replace('_',' ')} mean"))
#         plt.legend()
#         plt.show()
#         print()




# BREAKS AT TARGET....
def explore(df, target):
    '''This function is meant to split the df into categorical and numerical columns. The 
    default split is more than 3 unique variables makes it a numerical column. Less than 3 indicates 
    "buckets" which indicates categorical variables.

    Use the "explore_categorical/numerical" function to further explore the variables.
    '''
    col_cat = []  #this is for my categorical varibles
    col_num = [] #this is for my numeric varibles
    for col in df.columns:
        print(col) 
        if len(df[col].unique()) < 3: # making anything with less than 3 unique variables a categorical
            col_cat.append(col)
        else:
            col_num.append(col)
    for col in df:
        if col in col_cat:
            print()
            print(col.upper())
            print(df[col].value_counts())
            print(df[col].value_counts(normalize=True)*100)
            print()
            print()
            print(f'HYPOTHESIZE')
            print(f"H_0: {col.lower().replace('_',' ')} does not affect {target}")
            print(f"H_a: {col.lower().replace('_',' ')} affects {target}")
            print()
            print(f'VISUALIZE')
            sns.barplot(x=df[col], y=df[target])
            plt.title(f"{col.lower().replace('_',' ')} vs {target}")
            plt.show()
            print()
            print('ANALYZE and SUMMARIZE')
            observed = pd.crosstab(df[col], df[target])
            acq.chi2_test(observed)
            print()
            print()
        else:
            sns.barplot(data=df, x=df[target], y=df[col])
            plt.title(f"Is survived independent of {col.lower().replace('_',' ')}?")
            pop_mn = df[col].mean()
            plt.axhline(pop_mn, label=(f"{col.lower().replace('_',' ')} mean"))
            plt.legend()
            plt.show()
            print()