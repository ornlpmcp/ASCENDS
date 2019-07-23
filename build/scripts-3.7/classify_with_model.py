#!python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
import numpy as np
import os
import sys
import configparser
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn import preprocessing
import argparse
import ast

def main(args):

    model_file = args.model_file
    input_file = args.input_file
    output_file = args.output_file
    mapping = args.mapping
    mapping = ast.literal_eval(mapping)

    path_to_model = model_file
    data = pd.read_csv(input_file)

    model_dict = pickle.load(open(path_to_model, 'rb'))
    model = model_dict['model']
    original_data = pd.DataFrame(data)
    input_cols = model_dict['input_cols']
    data = data[input_cols]
    
    try:
        scaler = model_dict['fitted_scaler_x']
        if scaler!='False' and scaler is not None:
            data = scaler.transform(data)
    except:
        pass
    
    try:
        for key in mapping.keys():
            data[key] = data[key].map(mapping[key])
    except:
        pass
    
    if model_dict['model_abbr']=='NET':
        outcome = model.predict_classes(data)
        outcome_prob = model.predict(data)
    else:
        outcome = model.predict(data)
        outcome_prob = model.predict_proba(data)
    
    result = pd.DataFrame(outcome,columns=[model_dict['target_col']])
    result_prob = pd.DataFrame(outcome_prob)
    
    final = original_data.join(result_prob).join(result)
    
    new_set_of_dicts = {}
    if mapping!={}:
        for key in mapping.keys():
            dict_item = mapping[key]
            new_dict = {}
            for key_ in dict_item.keys():
                new_dict[dict_item[key_]]=key_
            new_set_of_dicts[key] = new_dict
        mapping = new_set_of_dicts

        for key in mapping.keys():
            final[key] = final[key].map(mapping[key])

    final = final.rename(columns=mapping[model_dict['target_col']])
    

    final.to_csv(output_file, sep=',')
    print("* DONE")
    
if __name__=="__main__":

    print("\n * ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists ")
    print(" * Classification ML model predictor \n")

    parser = argparse.ArgumentParser()
    parser.add_argument( "model_file")
    parser.add_argument( "input_file")
    parser.add_argument( "output_file")
    parser.add_argument( "--mapping", help="Mapping string value to numbers", default='{}')

    args = parser.parse_args()
    main(args)
