#!/usr/bin/env python3 -W ignore::RuntimeWarning
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import argparse
import statistics
import sys
import pprint
import keras
import ascends as asc
import ast
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pathlib import PurePath
import numpy as np
import tensorflow as tf
import random as rn

# This is a tool for training machine learning models for a regression (value prediction) task

"""
Ensures neural net parameters are correct types.
"""
def clean_up_net_params(net_neuron_max, net_structure, net_l_2, net_learning_rate, net_epochs, net_dropout, net_layer_n, net_batch_size):
    
    if net_neuron_max=='-1':
        net_neuron_max = []
    else:
        try:
            net_neuron_max = [int(x) for x in net_neuron_max]
        except:
            net_neuron_max = []

    if net_structure=='Tune':
        net_structure = None
    else:
        try:
            net_structure = [int(x) for x in net_structure]
        except:
            net_structure = []

    if net_l_2=='Tune':
        net_l_2 = None
    else:
        try:
            net_l_2 = float(net_l_2)
        except:
            net_l_2 = None
    
    if net_learning_rate=='Tune':
        net_learning_rate = None
    else:
        try:
            net_learning_rate = float(net_learning_rate)
        except:
            net_learning_rate = None
    
    if net_epochs=='Tune':
        net_epochs = None
    else:
        try:
            net_epochs = int(net_epochs)
        except:
            net_epochs = None

    if net_dropout=='Tune':
        net_dropout = True
    else:
        try:
            net_dropout = float(net_dropout)
        except:
            net_dropout = True

    if net_layer_n=='Tune':
        net_layer_n = None
    else:
        try:
            net_layer_n = int(net_layer_n)
        except:
            net_layer_n = None
    
    if net_batch_size=='Tune':
        net_batch_size = None
    else:
        try:
            net_batch_size = int(net_batch_size)
        except:
            net_batch_size = None
   
    return net_neuron_max, net_structure, net_l_2, net_learning_rate, net_epochs, net_dropout, net_layer_n, net_batch_size




                  



def main(args):


    """
    Load data
    """
    try:
        
        print("\n [ Data Loading ]")

            
        save_metadata = asc.str2bool(args.save_metadata)
        train_type = args.train_type
        csv_file = PurePath(args.input_file)
        cols_to_remove = args.ignore_col
        target_col = args.target_col
        input_col = args.input_col
        model_type = args.model_type
        hyperparameter_file = asc.fix_value(args.hyperparameter_file,'PurePath')
        num_of_features = int(args.num_of_features)
        num_of_folds = int(args.num_of_folds)
        test = asc.str2bool(args.test)
        mapping = args.mapping
        project_file = PurePath(args.project_file)
        save_test_chart = asc.str2bool(args.save_test_chart)
        save_auto_tune = asc.str2bool(args.save_auto_tune)
        save_test_csv = asc.str2bool(args.save_test_csv)
        auto_tune = asc.str2bool(args.auto_tune)
        auto_tune_iter = int(args.auto_tune_iter)
        random_state = asc.fix_value(args.random_state,'int')
        feature_selection = args.feature_selection
        feature_selection_file = args.feature_selection_file
        scaler_option = args.scaler
        save_corr_chart = args.save_corr_chart
        only_pcc = args.only_pcc
        net_fast_tune = args.net_fast_tune
        save_corr_report = args.save_corr_report

        net_structure = args.net_structure
        net_layer_n = args.net_layer_n
        net_dropout = args.net_dropout
        net_l_2 = args.net_l_2
        net_learning_rate = args.net_learning_rate
        net_epochs = args.net_epochs
        net_batch_size = args.net_batch_size
        net_neuron_max = args.net_neuron_max
        net_batch_size_max = int(args.net_batch_size_max)
        net_layer_min = int(args.net_layer_min)
        net_layer_max = int(args.net_layer_max)
        net_dropout_max = float(args.net_dropout_max)
        net_default_neuron_max = int(args.net_default_neuron_max)
        net_checkpoint = args.net_checkpoint
        num_of_class = int(args.num_of_class)

        #os.environ['PYTHONHASHSEED'] = str(random_state)

        # Setting the seed for numpy-generated random numbers
        #np.random.seed(random_state)

        # Setting the seed for python random numbers
        #rn.seed(random_state)

        # Setting the graph-level random seed.
        tf.set_random_seed(random_state)
        
        print(" Loading data from :%s"%(csv_file))
        print(" Columns to ignore :%s"%(cols_to_remove))
                
        data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file, input_col, cols_to_remove, target_col, map_all = ast.literal_eval(mapping), random_state = random_state)

        print(" Input columns :%s"%(header_x))
        print(" Target column :%s"%(target_col))

        if not os.path.exists(project_file): os.makedirs(project_file)
        for folder in ["predictions","correlations","tests","parameters","graphs","models"]:
            if not os.path.exists(project_file / folder): os.makedirs(project_file / folder)
        input_name=csv_file.stem

    except Exception as e:
        print("* An error occurred while loading data from ", args.input_file)
        print(e)
        sys.exit()


    """
    Analyze correlation
    """
    if feature_selection is not None:
        session_number=asc.get_session(project_file)
        if save_corr_report is not None:
            if save_corr_report =='True':
                
                save_corr_report = project_file / "correlations" / ("session"+str(session_number) + train_type+"_"+csv_file.name +"_target="+target_col+".csv")
            else:
                save_corr_report = None
            
        if save_corr_chart is not None:
            if save_corr_chart=='True':
                save_corr_chart = project_file / "correlations" / ("session"+str(session_number) + train_type+"_"+csv_file.name +"_target="+target_col+".png")
            else:
                save_corr_chart = None

        fs_dict, final_report = asc.correlation_analysis_all(data_df, target_col, num_of_features, file_to_save = save_corr_report, save_chart = save_corr_chart, only_pcc = only_pcc, feature_selection_file = feature_selection_file)
        
        if (feature_selection!="PCC") and (feature_selection!="PCC_SQRT") and only_pcc=='True':
            print("!! Error: you need to use PCC or PCC_SQRT for feature selection when only_pcc is set to True.")
            sys.exit()

        input_col = fs_dict[feature_selection]

        print("\n [ Feature Selection ]")
        
        print(" Reloading the data using the selected features : ", input_col," by criteron ", feature_selection, "top_k=", num_of_features)
        
        data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file, input_col, cols_to_remove, target_col, map_all = ast.literal_eval(mapping), random_state = random_state)

        print(" Input columns :%s"%(header_x))
        print(" Target column :%s"%(target_col))     
        print(" Saving correlation report to " + str(project_file / "correlations" / ("session"+str(session_number) + train_type+"_"+csv_file.name +"_target="+target_col+".csv")))
        print(" Saving correlation chart to " + str(project_file / "correlations" / ("session"+str(session_number) + train_type+"_"+csv_file.name +"_target="+target_col+".png")))
                
    

    """
    Tune model
    """
    if auto_tune is True and model_type!='LR' and model_type!='LRC':

        print("\n [ Hyperparameter Tuning ]")
        print(" Training with %s ..."%asc.model_name(model_type))

        if model_type=='NET':
            
            if net_checkpoint=='True':
                checkpoint = csv_file
            else:
                checkpoint = None

            model_parameters = {}
            net_neuron_max, net_structure, net_l_2, net_learning_rate, net_epochs, net_dropout, net_layer_n, net_batch_size = \
            clean_up_net_params(net_neuron_max, net_structure, net_l_2, net_learning_rate, net_epochs, net_dropout, net_layer_n, net_batch_size)
            
            if net_fast_tune == 'True':
                fast_tune = True
            else:
                fast_tune = False

            if train_type=='r':
                model_parameters = asc.net_tuning(tries = auto_tune_iter, lr = net_learning_rate, x_train = x_train, y_train = y_train, layer = net_layer_n, \
                params=net_structure, epochs=net_epochs, batch_size=net_batch_size, dropout=net_dropout, l_2 = net_l_2, neuron_max=net_neuron_max, batch_size_max=net_batch_size_max, \
                layer_min = net_layer_min, layer_max=net_layer_max, dropout_max=net_dropout_max, default_neuron_max=net_default_neuron_max, checkpoint = checkpoint, num_of_folds=num_of_folds, fast_tune = fast_tune, random_state = random_state)
            else:
                model_parameters = asc.net_tuning_classifier(num_of_class = num_of_class, tries = auto_tune_iter, lr = net_learning_rate, x_train = x_train, y_train = y_train, layer = net_layer_n, \
                params=net_structure, epochs=net_epochs, batch_size=net_batch_size, dropout=net_dropout, l_2 = net_l_2, neuron_max=net_neuron_max, batch_size_max=net_batch_size_max, \
                layer_min = net_layer_min, layer_max=net_layer_max, dropout_max=net_dropout_max, default_neuron_max=net_default_neuron_max, checkpoint = checkpoint, num_of_folds=num_of_folds, fast_tune = fast_tune, random_state = random_state)
            

        else:
            print (" Auto hyperparameter tuning initiated. ")
            
            if hyperparameter_file is not None:
                print (" Warning: %s will be overrided and not be used."%(hyperparameter_file))
            if train_type=='r':
                model_parameters = asc.hyperparameter_tuning(model_type, x_train, y_train
                                            , num_of_folds, scaler_option
                                            , n_iter=auto_tune_iter, random_state=random_state, verbose=1)
            else:
                model_parameters = asc.hyperparameter_tuning_classifier(model_type, x_train, y_train
                                                    , num_of_folds, scaler_option
                                                    , n_iter=auto_tune_iter, random_state=random_state, verbose=1)
        
        if model_parameters == {}:
            print(" The tool couldn't find good parameters ")
            print (" Using default scikit-learn hyperparameters ")
            model_parameters = asc.default_model_parameters() 

    else:

        if hyperparameter_file is not None and model_type!='LRC':

            print (" Using hyperparameters from the file %s"%(hyperparameter_file))
            model_parameters = asc.load_model_parameter_from_file(hyperparameter_file)

        else:
            print (" Using default scikit-learn hyperparameters ")
            if train_type=='c': model_parameters = asc.default_model_parameters_classifier()            
            else: model_parameters = asc.default_model_parameters()    
            
            print (" Overriding parameters from command-line arguments ..")
            if net_structure !='Tune':
                print(" net_structure is set to ", net_structure)
                model_parameters['net_structure'] = net_structure
            if net_dropout !='Tune':
                print(" net_dropout is set to ", net_dropout)
                model_parameters['net_dropout'] = net_dropout
            if net_l_2 !='Tune':
                print(" net_l_2 is set to ", net_l_2)
                model_parameters['net_l_2'] = net_l_2
            if net_learning_rate !='Tune':
                print(" net_learning_rate is set to ", net_learning_rate)
                model_parameters['net_learning_rate'] = net_learning_rate
            if net_epochs !='Tune':
                print(" net_epochs is set to ", net_epochs)
                model_parameters['net_epochs'] = net_epochs
            if net_batch_size !='Tune':
                print(" net_batch_size is set to ", net_batch_size)
                model_parameters['net_batch_size'] = net_batch_size
            if net_layer_n !='Tune':
                print(" net_layer_n is set to ", net_layer_n)
                model_parameters['net_layer_n'] = net_layer_n

    
    if train_type=='r': model_parameters['scaler_option'] = scaler_option
    MAE = None
    R2 = None
     
    accuracy = None


    """
    Evaluate model
    """
    if test is True:

        try:
            
            print("\n [ Model Evaluation ]")
            
            if model_type!='NET':
                if train_type=='r':
                    model = asc.define_model_regression(model_type, model_parameters, x_header_size = x_train.shape[1], random_state = random_state)
                    predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=num_of_folds)
                    MAE, R2 = asc.evaluate(predictions, actual_values)
                else:
                    model = asc.define_model_classifier(model_type, model_parameters, x_header_size = x_train.shape[1], random_state = random_state)
                    predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=num_of_folds)
                    accuracy = asc.evaluate_classifier(predictions, actual_values)
                    print("")
                    print("* Classification Report")
                    print(classification_report(actual_values, predictions))
                    
                    print("* Confusion Matrix (See here: http://bit.ly/2WxfXTy)")
                    print(confusion_matrix(actual_values, predictions))
                    print("")
            else:
                
                lr = float(model_parameters['net_learning_rate'])
                layer = int(model_parameters['net_layer_n'])
                dropout = float(model_parameters['net_dropout'])
                l_2 = float(model_parameters['net_l_2'])
                epochs = int(model_parameters['net_epochs'])
                batch_size = int(model_parameters['net_batch_size'])
                if ((type(net_structure)==list)==False):
                    net_structure = [int(x) for x in model_parameters['net_structure'].split(" ")]
                else:
                    net_structure = [int(x) for x in net_structure]
                
                optimizer = keras.optimizers.Adam(lr=lr)
                
                if train_type=='r':
                    model = asc.net_define(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer, random_state = random_state)
                else:
                    model = asc.net_define_classifier(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer, num_of_class = num_of_class, random_state=random_state)
                
                predictions, actual_values = asc.cross_val_predict_net(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, num_of_folds = num_of_folds, fast_tune = net_fast_tune)
                if train_type=='r':
                    MAE, R2 = asc.evaluate(predictions, actual_values)
                    print("* (%s)\t MAE = %8.3f, R2 = %8.3f via %d-fold cross validation "%(model_type, MAE, R2, num_of_folds))
                else:
                    accuracy = asc.evaluate_classifier(predictions, actual_values)
            
                    
                
        except Exception as e:
            print("* An error occurred while performing ML evaluation")
            print(e)
            sys.exit()      

        project_name=project_file.stem
        project_path=project_file.parent

        

        if save_metadata is True:
            print(" Saving metadata to "+ str(project_file / "metadata") + ".csv")
            try:
                session_number=asc.save_metadata(vars(args),{'MAE':MAE, 'R2':R2,'Accuracy':accuracy}, project_file / "metadata.csv")
            except:
                print(" * Warning: couldn't generate metadata - please make sure the model is properly trained .. ")

        if save_test_chart is True and train_type=='r':
            print(" Saving test charts to : ", str(project_file / "graphs" / ("session" +str(session_number)+ "r_"+input_name+"_"+model_type+".png")))
            try:    
                asc.save_comparison_chart(predictions, actual_values, project_file / "graphs" / ("session" +str(session_number)+ "r_"+input_name+"_"+model_type+".png"))
            except:
                print(" * Warning: couldn't generate a chart - please make sure the model is properly trained .. ")
            
        if save_test_csv is True and train_type=='r':
            print(" Saving test csv to : ", str(project_file / "tests" / ("session"+str(session_number)+ "r_"+input_name+"_"+model_type+".csv")))
            try:   
                asc.save_test_data(predictions, actual_values, project_file / "tests" / ("session"+str(session_number)+ "r_"+input_name+"_"+model_type+".csv"))
            except:
                print(" * Warning: couldn't generate a csv - please make sure the model is properly trained .. ")

        if save_auto_tune is True:
            print(" Saving hyperparameters to file: ", str(project_file / "parameters" / ("session"+str(session_number)+train_type+"_"+input_name+"_"+model_type+".tuned.prop")))
            asc.save_parameters(model_parameters, project_file / "parameters" / ("session"+str(session_number)+train_type+"_"+input_name+"_"+model_type+".tuned.prop"))

    

    """
    Save model
    """
    try:

        print("\n [ Model Save ]")
        
        if model_type!='NET':
            if train_type=='r':
                model = asc.define_model_regression(model_type, model_parameters, x_header_size = x_train.shape[1], random_state = random_state)
                asc.train_and_save(model, project_file / "models" / ("session"+str(session_number)+train_type+"_"+input_name+"_"+model_type+".pkl"), model_type
                            , input_cols=header_x, target_col=header_y
                            , x_train=x_train, y_train=y_train, scaler_option=scaler_option, path_to_save = '.', MAE=MAE, R2=R2)
            else:
                model = asc.define_model_classifier(model_type, model_parameters, x_header_size = x_train.shape[1], random_state = random_state)
                asc.train_and_save_classifier(model, project_file / "models" / ("session"+str(session_number)+train_type+"_"+input_name+"_"+model_type+".pkl"), model_type
                            , input_cols=header_x, target_col=header_y
                            , x_train=x_train, y_train=y_train, scaler_option=scaler_option, path_to_save = '.', accuracy=accuracy)
        else:
            
            lr = float(model_parameters['net_learning_rate'])
            layer = int(model_parameters['net_layer_n'])
            dropout = float(model_parameters['net_dropout'])
            l_2 = float(model_parameters['net_l_2'])
            epochs = int(model_parameters['net_epochs'])
            batch_size = int(model_parameters['net_batch_size'])
            if ((type(net_structure)==list)==False):
                    net_structure = [int(x) for x in model_parameters['net_structure'].split(" ")]
            else:
                net_structure = [int(x) for x in net_structure]

            optimizer = keras.optimizers.Adam(lr=lr)

            if train_type=='c':
                model = asc.net_define_classifier(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer, num_of_class = num_of_class)
                asc.train_and_save_net_classifier(model, project_file / "models" / ("session"+str(session_number)+train_type+"_"+input_name+"_"+model_type+".pkl"), input_cols=header_x, target_col=header_y, x_train=x_train, y_train=y_train, scaler_option=scaler_option, accuracy=accuracy, path_to_save = '.', num_of_folds=num_of_folds, epochs=epochs, batch_size=batch_size, num_of_class = num_of_class)
            else: 
                model = asc.net_define(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer)
                asc.train_and_save_net(model, project_file / "models" / ("session"+str(session_number)+train_type+"_"+input_name+"_"+model_type+".pkl"), input_cols=header_x, target_col=header_y, x_train=x_train, y_train=y_train, scaler_option=scaler_option, MAE=MAE, R2=R2, path_to_save = '.', num_of_folds=num_of_folds, epochs=epochs, batch_size=batch_size)
    
    except Exception as e:
        print("* An error occurred while training and saving .. ")
        print(e)
        sys.exit()  
    
    if test==True:
        if train_type=='r': print("\n MAE: %s R2: %s" % (MAE,R2))
        else: print("\n Accuracy: %s"%accuracy)
    
if __name__=="__main__":
    
    print("\n * ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists ")
    print(" * ML model trainer \n")
    print(" programmed by Matt Sangkeun Lee (lees4@ornl.gov) ")

    parser = argparse.ArgumentParser()
    parser.add_argument("train_type", help="Choose training type: 'c' for classification or 'r' for regression.",choices=['c','r'])
    parser.add_argument( "input_file", help="A csv file to train ML model")
    parser.add_argument( "project_file", help="project file to write")
    parser.add_argument( "--scaler", help="Perform hyperparameter tuning", choices=['False','StandardScaler','MinMaxScaler','RobustScaler','Normalizer'], default='StandardScaler')
    parser.add_argument( "--test", help="Perform cross validation and evaluate the expected performance of the model", choices=['True','False'], default='True')
    parser.add_argument( "--auto_tune", help="Perform hyperparameter tuning", choices=['True','False'], default='False')
    parser.add_argument( "--save_test_chart", choices=['True','False'], default='True')
    parser.add_argument( "--save_test_csv", choices=['True','False'], default='False')
    parser.add_argument( "--save_metadata", choices=['True','False'], default='True')
    parser.add_argument( "--mapping", help="Mapping string value to numbers", default='{}')
    parser.add_argument( "--save_auto_tune", choices=['True','False'], default='True')
    parser.add_argument( "--auto_tune_iter", default='1000')
    parser.add_argument( "--input_col", help="Input columns for training", nargs='+')
    parser.add_argument( "--num_of_class", help="number of class for classification", default=2)
    parser.add_argument( "--ignore_col", help="Columns to ignore for training", nargs='+')
    parser.add_argument( "--random_state", help="Random seed to shuffle dataset for random values", default='None')
    parser.add_argument( "target_col", help="A column to predict"),
    parser.add_argument( "--model_type", choices=['NET','RF','SVM','NN','RG','LRC','BR','KR','LR'], default='RF', help="LRC (Logistic Regression), RF (Random Forest), SVM (Support Vector Machine), RG (Ridge), or NN (k-Nearest Neighbor), RF is selected by default")
    parser.add_argument( "--num_of_features", help="Number of total features (automatic feature selection)", default=10)
    parser.add_argument( "--num_of_folds", help="Number of folds for cross validation", default=5)
    parser.add_argument( "--hyperparameter_file", help="Specify a hyperparameter file in case you want to \
    use specific hyper parameters")
    parser.add_argument("--feature_selection", default=None, choices=['PCC','PCC_SQRT','MIC','MAS','MEV','MCN','MCN_general','GMIC','TIC'])
    parser.add_argument("--feature_selection_file", default=None)
    parser.add_argument("--save_corr_report", default='False', choices=['True','False'])
    parser.add_argument("--save_corr_chart", default='False', choices=['True','False'])
    parser.add_argument("--only_pcc", default='False', choices=['True','False'])
    parser.add_argument("--net_fast_tune", default='False', choices=['True','False'])
    
    # neural net parameters
    parser.add_argument("--net_layer_n", default='Tune', help='Number of layers for neural network for hyperparameter tuning')
    parser.add_argument("--net_structure", default='Tune', nargs='+', help='If set to Tune, then the tool tries to tune when hyperparameter tuning is on . Specify specific structure of neural network if you want (e.g., 16 64 16)')
    parser.add_argument("--net_dropout", default='Tune', help='If set to Tune, then the tool tries to tune when hyperparameter tuning is on ')
    parser.add_argument("--net_l_2", default='Tune', help='If set to Tune, then the tool tries to tune when hyperparameter tuning is on ')
    parser.add_argument("--net_learning_rate", default='Tune', help='then the tool tries to tune when hyperparameter tuning is on ')
    parser.add_argument("--net_epochs", default='Tune', help='If set to Tune, then the tool tries to tune when hyperparameter tuning is on ')
    parser.add_argument("--net_batch_size", default='Tune', help='If set to Tune, then the tool tries to tune when hyperparameter tuning is on ')
    parser.add_argument("--net_neuron_max", default=-1, nargs='+', help='specify max neurons for each layer for tuning (e.g., 64 32 128), if -1, default_neuron_max value will be used for all layers')
    parser.add_argument("--net_batch_size_max", default=5, type=int)
    parser.add_argument("--net_layer_min", default=3, type=int)
    parser.add_argument("--net_layer_max", default=5, type=int)
    parser.add_argument("--net_dropout_max", default=0.2, type=float)
    parser.add_argument("--net_default_neuron_max", default=32, type=int)
    parser.add_argument("--net_checkpoint", default='True', choices=['True','False'])

    args = parser.parse_args()
    main(args)
