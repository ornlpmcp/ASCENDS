

```
import numpy as np
import ascends as asc
import keras
import ast
from sklearn.metrics import classification_report
import pickle
```


```
# 1. Regression API reference

# * NOTE: Ascends-toolkit was developed to be used via command-line interface or web-based interface; however, if needed,
# users may use ascends-toolkit's API. The following shows an example of performing a classification task using 
# the core ascends-toolkit APIs. 

# You need to download the example data file from https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv
# and save it under data folder

csv_file = 'data/iris.csv'
cols_to_remove = []
target_col = 'Name'
input_col = None

# Classifier will need a mapping between categorical values to numerical values
mapping = {'Name': {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}}

# Load data from csv file
# A standard csv file can be loaded and shuffled as follows

data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file, input_col, cols_to_remove, target_col, map_all = mapping, random_state = 0)
```


```
# check if data is loaded
data_df[:10]
```


```
# Generating a default model parameters
model_parameters = asc.default_model_parameters_classifier() 
```


```
model_type = 'RF'
scaler_option = 'StandardScaler' # scaler option can be 'False','StandardScaler','Normalizer','MinMaxScaler','RobustScaler'
num_of_folds = 5
model = asc.define_model_classifier(model_type, model_parameters, x_header_size = x_train.shape[1])   
```


```
# scikit-learn's classification report can be used to understand the accuracy of the trained model
predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=num_of_folds)
accuracy = asc.evaluate_classifier(predictions, actual_values)
print("")
print("* Classification Report")
print(classification_report(actual_values, predictions))
```

    
    * Classification Report
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00        50
             1.0       0.94      0.92      0.93        50
             2.0       0.92      0.94      0.93        50
    
       micro avg       0.95      0.95      0.95       150
       macro avg       0.95      0.95      0.95       150
    weighted avg       0.95      0.95      0.95       150
    



```
asc.train_and_save_classifier(model, "model.pkl", model_type
                            , input_cols=header_x, target_col=header_y
                            , x_train=x_train, y_train=y_train, scaler_option=scaler_option, path_to_save = '.', accuracy=accuracy)
```

    * Training initiated ..
    * Training done.
    * Trained model saved to file: model.pkl



```
# You can load the saved model by using pickle package
model_dict = pickle.load(open('model.pkl', 'rb'))

# Let's assume that we have a input as follows
x_to_predict = [[4.5, 2.4, 1.2, 4.2]]

# You can scale the data using the loaded scaler
scaler = model_dict['fitted_scaler_x']
x_to_predict = scaler.transform(x_to_predict)
print("Scaled x_to_predict = ", x_to_predict)

# Making prediction can be done as follows
predicted_y = model.predict(x_to_predict)

# Original prediction value will not be a class name, so you need to find out the class name by doing:
for key in mapping['Name'].keys():
    if mapping['Name'][key]==predicted_y[0]:
        print("* Your model thinks that it's a ", key)
```

    Scaled x_to_predict =  [[-1.62768837 -1.51337555 -1.45500383  3.94594202]]
    * Your model thinks that it's a  Iris-setosa



```
# 2. Regression API reference

# * NOTE: Ascends-toolkit was developed to be used via command-line interface or web-based interface; however, if needed,
# users may use ascends-toolkit's API. The following shows an example of performing a regression task using 
# the core ascends-toolkit APIs

csv_file = 'data/BostonHousing.csv'
cols_to_remove = []
target_col = 'medv'

# Load data from csv file
# A standard csv file can be loaded and shuffled as follows

data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file, None, cols_to_remove, target_col)
```


```
# Performing correlation analysis
# Correlation analysis can be performed as follows
# fs_dict will only contain the top-k features for each criteria (e.g., PCC)
# final_report will contain the full evaluation scores for each feature

k = 10
fs_dict, final_report = asc.correlation_analysis_all(data_df, target_col, top_k = k, file_to_save = None, save_chart = None)

print("Top-k features for each criteria")
print(fs_dict)
print("")
print("Full Correlation Analysis report")
print(final_report)

# To use top-k (k=10) features based on PCC (Pearson's correlation coefficient)

input_col = fs_dict['PCC']

# We need to load the file again
data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file, input_col, cols_to_remove, target_col)
```

    * correlation_analysis_all
    Top-k features for each criteria
    {'PCC': ['rm', 'zn', 'b', 'dis', 'chas', 'age', 'rad', 'crim', 'nox', 'tax'], 'PCC_SQRT': ['lstat', 'rm', 'ptratio', 'indus', 'tax', 'nox', 'crim', 'rad', 'age', 'zn'], 'MIC': ['lstat', 'rm', 'nox', 'age', 'indus', 'ptratio', 'crim', 'tax', 'dis', 'zn'], 'MAS': ['chas', 'b', 'age', 'zn', 'rad', 'dis', 'nox', 'ptratio', 'crim', 'tax'], 'MEV': ['lstat', 'rm', 'nox', 'age', 'indus', 'ptratio', 'crim', 'tax', 'dis', 'zn'], 'MCN': ['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio'], 'MCN_general': ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax'], 'GMIC': ['lstat', 'rm', 'nox', 'age', 'indus', 'ptratio', 'crim', 'tax', 'dis', 'rad'], 'TIC': ['lstat', 'rm', 'nox', 'indus', 'ptratio', 'age', 'crim', 'tax', 'dis', 'rad']}
    
    Full Correlation Analysis report
                  MIC       MAS       MEV       MCN  MCN_general      GMIC  \
    age      0.420689  0.099268  0.420689  5.321928          2.0  0.368688   
    b        0.272469  0.112505  0.272469  5.321928          2.0  0.207560   
    chas     0.133026  0.113481  0.133026  5.321928          2.0  0.079504   
    crim     0.358757  0.044427  0.358757  5.000000          2.0  0.326953   
    dis      0.315033  0.055968  0.315033  5.321928          2.0  0.282479   
    indus    0.414140  0.039397  0.414140  5.321928          2.0  0.350791   
    lstat    0.615427  0.034828  0.615427  5.321928          2.0  0.563114   
    nox      0.442723  0.047576  0.442723  5.321928          2.0  0.390978   
    ptratio  0.371581  0.045671  0.371581  5.321928          2.0  0.335871   
    rad      0.278780  0.060237  0.278780  5.321928          2.0  0.238301   
    rm       0.450967  0.038707  0.450967  5.321928          2.0  0.429243   
    tax      0.324490  0.041496  0.324490  5.321928          2.0  0.287834   
    zn       0.289734  0.098851  0.289734  5.321928          2.0  0.236065   
    
                   TIC  PCC_SQRT       PCC  
    age      21.160602  0.376955 -0.376955  
    b        11.615744  0.333461  0.333461  
    chas      4.074539  0.175260  0.175260  
    crim     21.033617  0.388305 -0.388305  
    dis      18.077608  0.249929  0.249929  
    indus    23.199544  0.483725 -0.483725  
    lstat    38.803088  0.737663 -0.737663  
    nox      25.119383  0.427321 -0.427321  
    ptratio  21.688436  0.507787 -0.507787  
    rad      14.967824  0.381626 -0.381626  
    rm       28.144318  0.695360  0.695360  
    tax      19.708624  0.468536 -0.468536  
    zn       12.802512  0.360445  0.360445  



```
# Generating a default model parameters
model_parameters = asc.default_model_parameters() 

# Model Training
model_type = 'RF' # model type can be 'LR','RF','NN','KR','BR','SVM'
scaler_option = 'StandardScaler' # scaler option can be 'False','StandardScaler','Normalizer','MinMaxScaler','RobustScaler'
num_of_folds = 5
model = asc.define_model_regression(model_type, model_parameters, x_header_size = x_train.shape[1])
predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=num_of_folds)
MAE, R2 = asc.evaluate(predictions, actual_values)

# Printing the performance of regression task
print("MAE = ", MAE,", R2 = ", R2)
```

    MAE =  2.793895239226903 , R2 =  0.6994175646013248



```
# tuning hyper parameters
tuned_parameters = asc.hyperparameter_tuning(model_type, x_train, y_train
                                             , num_of_folds, scaler_option
                                           , n_iter=1000, random_state=0, verbose=1)
```

    Fitting 5 folds for each of 1000 candidates, totalling 5000 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    6.1s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   18.6s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   41.6s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=-1)]: Done 1784 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 2434 tasks      | elapsed:  3.7min
    [Parallel(n_jobs=-1)]: Done 3184 tasks      | elapsed:  4.8min
    [Parallel(n_jobs=-1)]: Done 4034 tasks      | elapsed:  6.2min
    [Parallel(n_jobs=-1)]: Done 4984 tasks      | elapsed:  7.7min
    [Parallel(n_jobs=-1)]: Done 5000 out of 5000 | elapsed:  7.7min finished
    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```
# Model Training
model_type = 'RF' # model type can be 'LR','RF','NN','KR','BR','SVM'
scaler_option = 'StandardScaler' # scaler option can be 'False','StandardScaler','Normalizer','MinMaxScaler','RobustScaler'
num_of_folds = 5
model = asc.define_model_regression(model_type, model_parameters, x_header_size = x_train.shape[1])
predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=num_of_folds)
MAE, R2 = asc.evaluate(predictions, actual_values)

# Printing the performance of regression task
print("MAE = ", MAE,", R2 = ", R2)
```

    MAE =  2.8167252676854493 , R2 =  0.7039443187462722



```
# save prediction-actual result in a csv file
asc.save_test_data(predictions, actual_values, "result.csv")
```


```
# saving the trained model in a file
asc.train_and_save(model, "trained_model", model_type
                        , input_cols=header_x, target_col=header_y
                        , x_train=x_train, y_train=y_train, scaler_option=scaler_option, path_to_save = '.', MAE=MAE, R2=R2)
```

    * Training initiated ..
    * Training done.
    * Trained model saved to file: trained_model



```
# Model file loading and making a prediction can be done in the same way as the classification example
```
