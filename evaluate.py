# for loop to try MANY depths
# for x in range(1,20):
#     # print(x)
#     tree = DecisionTreeClassifier(max_depth=x)
#     tree.fit(X_train, y_train)
#     acc = tree.score(X_train, y_train)
#     print(f"For depth of {x:2}, the accuracy is {round(acc,2)}")

# to calculcate the best model
# make a list of lists, then turn to df, then turn to new column to give difference
# then it visulaizes the differences

def classifier_tree_eval(X_train, y_train, X_validate, y_validate):
    ''' This function is to calculate the best classifier decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    import warnings
    warnings.filterwarnings("ignore")
    for x in range(1,11):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        print(f"For depth of {x:2}, the accuracy is {round(train_acc,2)}")
        
        # evaludate on validate set
        validate_acc = tree.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # can plot to visulaize
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Decision Tree')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()

# select a model before the split of the two graphs. A large split indicates overfitting
# when selecing the depth to run with select the point where the difference between
# the train and validate set is the smallest before they seperate.



def random_forest_eval(X_train, y_train, X_validate, y_validate):
    ''' This function is to calculate the best random forest decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import warnings
    warnings.filterwarnings("ignore")

    for x in range(1,11):
        rf = RandomForestClassifier(random_state = 123,max_depth = x)
        rf.fit(X_train, y_train)
        train_acc = rf.score(X_train, y_train)
        print(f"For depth of {x:2}, the accuracy is {round(train_acc,2)}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc = rf.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
        # plot feature importance
    plt.figure(figsize=(12,12))
    plt.bar(X_train.columns, important_features)
    plt.title(f"Feature Importance")
    plt.xlabel(f"Features")
    plt.ylabel(f"Importance")
    plt.xticks(rotation = 60)
    plt.show()   

# increasing leaf samples by one and decreasing depth by 1
# for x in range(1,11):
#     print(x, 11-x)

# rf = RandomForestClassifier(random_state=123, min_samples_leaf=x, max_depth=11-x)
# rf.fit(X_train, y_train)
# train_acc = rf.score(X_train, y_train)


def knn_titanic_acq_prep_split_evaluate():
    import pandas as pd
    import numpy as np

    import acquire as acq
    import prepare as prep
    import stats_conclude as sc
    import evaluate as ev

    import matplotlib.pyplot as plt
    import seaborn as sns

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    """
    
    """

    # acquire
    df = acq.get_titanic_data()

    # prepare
    dft = prep.clean_titanic(df)
    dft.drop(columns=['passenger_id', 'sex', 'embark_town', 'embark_town_Queenstown','sibsp','parch', 'embark_town_Southampton'], inplace=True)
    
    # split
    train, validate, test = prep.split_titanic(dft)
    
    # assign variables
    target = 'survived'
    X_train = train.iloc[:,1:]
    X_validate = validate.iloc[:,1:]
    # X_test = test.iloc[:,1:]
    y_train = train[target]
    y_validate = validate[target]
    # y_test = test[target]
    print(f"The number of features sent in : {len(X_train.columns)}")

    # run for loop and plot
    metrics = []
    for k in range(1,21):
        
        # MAKE the thing
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # FIT the thing
        knn.fit(X_train, y_train)
        
        # USE the thing (calculate accuracy)
        train_score = knn.score(X_train, y_train)
        validate_score = knn.score(X_validate, y_validate)
        
        # append to df metrics
        metrics.append([k, train_score, validate_score])

        # turn to df
        metrics_df = pd.DataFrame(metrics, columns=['k', 'train score', 'validate score'])
      
        # make new column
        metrics_df['difference'] = metrics_df['train score'] - metrics_df['validate score']
    min_diff_idx = np.abs(metrics_df['difference']).argmin()
    n = metrics_df.loc[min_diff_idx, 'k']
    print(f"{n} is the number of neighbors that produces the best fit model")
    
    
    # plot the data
    metrics_df.set_index('k').plot(figsize = (14,12))
    plt.axvline(x=n, color='black', linestyle='--', linewidth=1, label='best fit neighbor size')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,21,1))
    plt.legend()
    plt.grid()
    
    
    
    
    
    
# This code calculates the value of k that results in the minimum absolute 
# difference between the train and validation accuracy. Here's a step-by-step 
# breakdown of what's happening:

# results['diff_score'] retrieves the column of the DataFrame that contains the 
# difference between the train and validation accuracy for each value of k.

# np.abs(results['diff_score']) takes the absolute value of each difference score, 
# since we're interested in the magnitude of the difference regardless of its sign.

# np.abs(results['diff_score']).argmin() finds the index of the minimum value in 
# the absolute difference score column. This corresponds to the value of k that 
# results in the smallest absolute difference between the train and validation accuracy.

# results.loc[min_diff_idx, 'k'] retrieves the value of k corresponding to the 
# minimum absolute difference score.

# results.loc[min_diff_idx, 'diff_score'] retrieves the minimum absolute difference 
# score itself.
