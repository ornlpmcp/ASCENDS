

```python
import numpy as np
import ascends as asc
import keras
import ast
```


```python
# Ascends-toolkit was developed to be used via command-line interface or web-based interface; however, if needed,
# users may use ascends-toolkit's API. The following shows an example of performing a regression task using 
# the core ascends-toolkit APIs

csv_file = 'BostonHousing.csv'
cols_to_remove = []
target_col = 'medv'

# Load data from csv file
# A standard csv file can be loaded and shuffled as follows

data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file, None, cols_to_remove, target_col)
```


```python
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



```python
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

    MAE =  2.814968381964642 , R2 =  0.6958314131579451



```python
# tuning hyper parameters
tuned_parameters = asc.hyperparameter_tuning(model_type, x_train, y_train
                                             , num_of_folds, scaler_option
                                           , n_iter=1000, random_state=0, verbose=1)
```

    Fitting 5 folds for each of 1000 candidates, totalling 5000 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    5.5s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   16.5s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:   41.4s
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done 1784 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 2434 tasks      | elapsed:  3.8min
    [Parallel(n_jobs=-1)]: Done 3184 tasks      | elapsed:  5.0min
    [Parallel(n_jobs=-1)]: Done 4034 tasks      | elapsed:  6.7min
    [Parallel(n_jobs=-1)]: Done 4984 tasks      | elapsed:  8.4min
    [Parallel(n_jobs=-1)]: Done 5000 out of 5000 | elapsed:  8.4min finished
    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_search.py:841: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)



```python
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

    MAE =  2.793541478026997 , R2 =  0.702740528877592



```python
# save prediction-actual comparison chart
asc.save_comparison_chart(predictions, actual_values, "comparison_chart.png")
```


```python
# save prediction-actual result in a csv file
asc.save_test_data(predictions, actual_values, "result.csv")
```


```python
# saving the trained model in a file
asc.train_and_save(model, "trained_model", model_type
                        , input_cols=header_x, target_col=header_y
                        , x_train=x_train, y_train=y_train, scaler_option=scaler_option, path_to_save = '.', MAE=MAE, R2=R2)
```

    * Training initiated ..
    * Training done.
    * Trained model saved to file: ./trained_model.pkl

