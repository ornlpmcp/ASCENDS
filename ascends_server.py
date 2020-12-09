#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
import warnings
import tornado.escape
import tornado.ioloop
import tornado.web
from  tornado.escape import json_decode
from  tornado.escape import json_encode
from tornado.concurrent import Future
from tornado import gen
from tornado.options import define, options, parse_command_line
import traceback
import os
import json
import csv
import sys
import ascends as asc
import pandas as pd
import glob
import pickle
import keras
from pathlib import PurePath

__UPLOADS__ = PurePath("static/uploads/")

define("port", default=7777, help="run on the given port", type=int)
define("debug", default=False, help="run in debug mode")

# -- Helper functions -- #

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

def index_cols(header, rows):
    cols = {}
    for i in range(0, len(header)):
        attr_name = header[i]
        for row in rows:
            try:
                cols[attr_name].append(row[i])
            except:
                cols[attr_name] = [row[i]]
    
    if_number = {}

    for key in cols.keys():
        for val in cols[key]:
            try:
                float(val)
            except:
                if_number[key]=False
                break
            if_number[key]=True
    
    return cols, if_number

# -- Handler functions -- #

class MainHandler(tornado.web.RequestHandler):

    def get(self):
        path_to_data =  self.get_argument("path_to_data", default=None, strip=False)
        path_to_data = path_to_data.replace("\\","/")
        target_col =  self.get_argument("target_col", default=None, strip=False)
        input_cols =  self.get_argument("input_cols", default=None, strip=False)
        
        json_data = {}
        json_data['path_to_data'] = path_to_data
        json_data['target_col'] = target_col
        json_data['input_cols'] = input_cols

        self.render("index.html", title="Profile", data=json.dumps(json_data))

class OpenFileHandler(tornado.web.RequestHandler):
    
    def post(self):
        
        response_to_send = {}
        need_to_upload = True
        
        try:
            fileinfo = self.request.files['input-csv'][0]
            fname = fileinfo['filename']
            extn = os.path.splitext(fname)[1]
        
        except:
            need_to_upload = False
            json_obj = json_decode(self.request.body)
            path_to_data = json_obj['path_to_data'].split(".")
            extn = "."+path_to_data[-1]

        if extn==".csv":
            
            if need_to_upload==True:
                cname = "opened" + extn
                fh = open(__UPLOADS__ / cname, 'wb')
                fh.write(fileinfo['body'])
                fh.close()
                file_path = __UPLOADS__ / cname
            else:
                json_obj = json_decode(self.request.body)
                file_path = json_obj["path_to_data"]

            try:

                header = []
                rows = []
                cols = {}

                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    r_idx = 0
                    for row in reader:
                        if r_idx==0:
                            for i in range(0,len(row)):
                                header.append(row[i])
                        else:
                            if row!=[]: rows.append(row)
                        r_idx+=1
            
                response_to_send['msg'] = 'success'
                response_to_send['header'] = header
                response_to_send['rows'] = rows
                response_to_send['path_to_data'] = str(file_path)

                cols, if_number = index_cols(header, rows)
                response_to_send['if_number'] = if_number

            except Exception as e:
                response_to_send['msg'] = 'fail_open_csv'
                print(e)

        else:
            response_to_send['msg'] = 'error_no_csv'

        self.write(json.dumps(response_to_send))

class FeatureAnalysisHandler(tornado.web.RequestHandler):
    
    def get(self):
        path_to_data =  self.get_argument("path_to_data", default=None, strip=False)
        target_col =  self.get_argument("target_col", default=None, strip=False)
        input_cols =  self.get_argument("input_cols", default=None, strip=False)
        
        json_data = {}
        json_data['path_to_data'] = path_to_data
        json_data['target_col'] = target_col
        json_data['input_cols'] = input_cols

        self.render("index.html", title="Profile", data=json.dumps(json_data))

    def post(self):
        
        print("* Feature Analysis Started ..")
        json_obj = json_decode(self.request.body)
        target_col = json_obj["target_col"]
        input_cols = json_obj["input_cols"]
        file_path = json_obj["path_to_data"]
        
        try:
            input_cols.remove(target_col)
        except:
            # remove target column from input column list
            pass

        data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file = file_path, input_col=input_cols, cols_to_remove=[], target_col=target_col, random_state=0)
        fs_dict, final_report = asc.correlation_analysis_all(data_df, target_col, top_k=99999, file_to_save = None, save_chart = None)
        rows = [['Feature','MIC','MAS','MEV','MCN','MCN_general','GMIC','TIC','PCC_SQRT','PCC']]
        for index, row in final_report.iterrows():
            rows.append([index, row['MIC'], row['MAS'], row['MEV'], row['MCN'], row['MCN_general'], row['GMIC'], row['TIC'], row['PCC_SQRT'], row['PCC']])

        response_to_send = {}
        response_to_send['rows'] = rows
        self.write(json.dumps(response_to_send))

class MLAnalysisHandler(tornado.web.RequestHandler):
    
    def get(self):
        path_to_data =  self.get_argument("path_to_data", default=None, strip=False)
        target_col =  self.get_argument("target_col", default=None, strip=False)
        input_cols =  self.get_argument("input_cols", default=None, strip=False)
        
        json_data = {}
        json_data['path_to_data'] = path_to_data
        json_data['target_col'] = target_col
        json_data['input_cols'] = input_cols

        self.render("ml.html", title="Profile", data=json.dumps(json_data))

class GetModelFileListHandler(tornado.web.RequestHandler):
    
    def post(self):
        model_file_list = glob.glob(str(PurePath("static/learned_models/*.pkl")))
        response_to_send = {}
        response_to_send['model_files'] = model_file_list

        self.write(json.dumps(response_to_send))

class GetPresetFileListHandler(tornado.web.RequestHandler):
    
    def post(self):
        preset_file_list = glob.glob(str(PurePath("static/config/*.*")))
        response_to_send = {}
        response_to_send['preset_files'] = preset_file_list

        self.write(json.dumps(response_to_send))

class ExecuteMLTuningHandler(tornado.web.RequestHandler):
    
    def post(self):
        json_obj = json_decode(self.request.body)
        target_col = json_obj["target_col"]
        input_cols = json_obj["input_cols"]
        num_of_folds = int(json_obj["num_fold"])
        preset = json_obj["preset"]
        scaler_option = json_obj["scaler"]
        
        file_path = json_obj["path_to_data"]
        model_type = json_obj["model_abbr"]
        auto_tune_iter = 1000
        random_state = None
        
        data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file = file_path, input_col=input_cols, cols_to_remove=[], target_col=target_col, random_state=None)
        
        if model_type=='NET':
            
            net_neuron_max, net_structure, net_l_2, net_learning_rate, net_epochs, net_dropout, net_layer_n, net_batch_size = \
            clean_up_net_params(-1,'Tune','Tune','Tune','Tune','Tune','Tune','Tune')
            net_batch_size_max = 5
            net_layer_min = 3
            net_layer_max = 5
            net_dropout_max = 0.2
            net_default_neuron_max = 32
            checkpoint = None
            model_parameters = asc.net_tuning(tries = auto_tune_iter, lr = net_learning_rate, x_train = x_train, y_train = y_train, layer = net_layer_n, \
            params=net_structure, epochs=net_epochs, batch_size=net_batch_size, dropout=net_dropout, l_2 = net_l_2, neuron_max=net_neuron_max, batch_size_max=net_batch_size_max, \
            layer_min = net_layer_min, layer_max=net_layer_max, dropout_max=net_dropout_max, default_neuron_max=net_default_neuron_max, checkpoint = checkpoint, num_of_folds=num_of_folds)

            if model_parameters == {}:
                print(" The tool couldn't find good parameters ")
                print (" Using default scikit-learn hyperparameters ")
                model_parameters = asc.default_model_parameters() 

        else:
            print (" Auto hyperparameter tuning initiated. ")
            model_parameters = asc.hyperparameter_tuning(model_type, x_train, y_train
                                                , num_of_folds, scaler_option
                                                , n_iter=auto_tune_iter, random_state=random_state, verbose=1)

        csv_file = PurePath('static/config/') / PurePath(file_path).name
        print(" Saving tuned hyperparameters to file: ", str(csv_file)+",WEB,Model="+model_type+",Scaler="+scaler_option+".tuned.prop")
        asc.save_parameters(model_parameters, str(csv_file)+",Model="+model_type+",Scaler="+scaler_option+".tuned.prop")

        response_to_send = {'output':str(csv_file)+",Model="+model_type+",Scaler="+scaler_option+".tuned.prop"}
        
        self.write(json.dumps(response_to_send))

class ExecuteMLAnalysisHandler(tornado.web.RequestHandler):
    
    def post(self):
        json_obj = json_decode(self.request.body)
        target_col = json_obj["target_col"]
        input_cols = json_obj["input_cols"]
        num_fold = json_obj["num_fold"]
        preset = json_obj["preset"]
        scaler_option = json_obj["scaler"]
        file_path = json_obj["path_to_data"]
        model_abbr = json_obj["model_abbr"]
        

        data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file = file_path, input_col=input_cols, cols_to_remove=[], target_col=target_col, random_state=None)
    
        if(preset=='default'):
            model_parameters = asc.default_model_parameters()
            #scaler_option = model_parameters['scaler_option']
        else:
            model_parameters = asc.load_model_parameter_from_file(preset)
            #scaler_option = model_parameters['scaler_option']
        if scaler_option=="AutoLoad":
            scaler_option = model_parameters['scaler_option'] 

        try:
            
            if model_abbr=='NET':
                lr = float(model_parameters['net_learning_rate'])
                layer = int(model_parameters['net_layer_n'])
                dropout = float(model_parameters['net_dropout'])
                l_2 = float(model_parameters['net_l_2'])
                epochs = int(model_parameters['net_epochs'])
                batch_size = int(model_parameters['net_batch_size'])
                net_structure = [int(x) for x in model_parameters['net_structure'].split(" ")]

                optimizer = keras.optimizers.Adam(lr=lr)
                model = asc.net_define(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer)
                predictions, actual_values = asc.cross_val_predict_net(model, epochs=epochs, batch_size=batch_size, x_train = x_train, y_train = y_train, verbose = 0, scaler_option = scaler_option, force_to_proceed=True)
                MAE, R2 = asc.evaluate(predictions, actual_values)
                
            else:
                model = asc.define_model_regression(model_abbr, model_parameters, x_header_size = x_train.shape[1])
                predictions, actual_values = asc.train_and_predict(model, x_train, y_train, scaler_option=scaler_option, num_of_folds=int(num_fold))
                MAE, R2 = asc.evaluate(predictions, actual_values)

        except Exception as e:
            MAE = -1
            R2 = -1

        if MAE!=-1:          
            asc.save_comparison_chart(predictions, actual_values, PurePath("static/output/ml/ml_result.png"))
        response_to_send = {}
        response_to_send["MAE"]=float(MAE)
        response_to_send["R2"]=float(R2)
        response_to_send["input_cols"]=input_cols
        response_to_send["target_col"]=target_col
        response_to_send["model_abbr"]=model_abbr
        response_to_send["num_fold"]=num_fold
        response_to_send["scaler"]=scaler_option
        print(response_to_send)
        
        self.write(json.dumps(response_to_send))

class SaveModelHandler(tornado.web.RequestHandler):
    
    def post(self):
        
        json_obj = json_decode(self.request.body)
        target_col = json_obj["target_col"]
        input_cols = json_obj["input_cols"]
        num_fold = json_obj["num_fold"]
        tag = json_obj["tag"]
        MAE = json_obj["MAE"]
        R2 = json_obj["R2"]
        preset = json_obj["preset"]
        scaler_option = json_obj["scaler"]
        
        file_path = json_obj["path_to_data"]
        model_abbr = json_obj["model_abbr"]

        if(preset=='default'):
            model_parameters = asc.default_model_parameters()
        else:
            model_parameters = asc.load_model_parameter_from_file(preset)

        data_df, x_train, y_train, header_x, header_y = asc.data_load_shuffle(csv_file = file_path, input_col=input_cols, cols_to_remove=[], target_col=target_col, random_state=0)
            
        if model_abbr!='NET':
            model = asc.define_model_regression(model_type=model_abbr, model_parameters = model_parameters, x_header_size = x_train.shape[1])
            asc.train_and_save(model, PurePath('static/learned_models/'+tag+'.pkl'), model_abbr
                            , input_cols=header_x, target_col=header_y
                            , x_train=x_train, y_train=y_train, scaler_option=scaler_option, path_to_save = '.', MAE=MAE, R2=R2)
        else:
            
            lr = float(model_parameters['net_learning_rate'])
            layer = int(model_parameters['net_layer_n'])
            dropout = float(model_parameters['net_dropout'])
            l_2 = float(model_parameters['net_l_2'])
            epochs = int(model_parameters['net_epochs'])
            batch_size = int(model_parameters['net_batch_size'])
            net_structure = [int(x) for x in model_parameters['net_structure'].split(" ")]

            optimizer = keras.optimizers.Adam(lr=lr)
            model = asc.net_define(params=net_structure, layer_n = layer, input_size = x_train.shape[1], dropout=dropout, l_2=l_2, optimizer=optimizer)
            asc.train_and_save_net(model, PurePath('static/learned_models/'+tag+'.pkl'), input_cols=header_x, target_col=header_y, x_train=x_train, y_train=y_train, scaler_option=scaler_option, MAE=MAE, R2=R2, path_to_save = '.', num_of_folds=5, epochs=epochs, batch_size=batch_size)

        model_files =  glob.glob(str(PurePath("static/learned_models/*.pkl")))

        response_to_send = {}
        response_to_send['model_files'] = model_files
        self.write(json.dumps(response_to_send))

class GetModelInfoHandler(tornado.web.RequestHandler):
    def post(self):
        
        json_obj = json_decode(self.request.body)
        model_file = json_obj["model_file"]
        model_dict = pickle.load(open(model_file.strip(), 'rb'))
        response_to_send = {}
        print(model_dict)
        response_to_send['input_cols'] = list(model_dict['input_cols'])
        response_to_send['target_col'] = model_dict['target_col']
        response_to_send['model_abbr'] = model_dict['model_abbr']
        response_to_send['MAE'] = float(model_dict['MAE'])
        response_to_send['R2'] = float(model_dict['R2'])

        self.write(json.dumps(response_to_send))

class PredictPageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("predict.html")

class DeleteModelHandeler(tornado.web.RequestHandler):
    
    def post(self):
        json_obj = json_decode(self.request.body)
        model_file = json_obj["model_to_delete"]
        os.remove(model_file.strip())
        response_to_send = {}
        
        self.write(json.dumps(response_to_send))

class GetPredictedTarget(tornado.web.RequestHandler):
    
    def post(self):
        json_obj = json_decode(self.request.body)
        current_model =  json_obj['current_model']
        target_col = json_obj['target_col']
        input_cols = json_obj['input_cols']
        
        #table header
        header_str = json_obj['header_str']
        
        col_index_to_consider = []
        for i in range(0, len(input_cols)):
            if input_cols[i] in header_str:
                for j in range(0,len(header_str)):
                    if header_str[j]==input_cols[i]:
                        col_index_to_consider.append(j)
        #print input_cols
        rows = json_obj['rows']
        predictions = []
        model_dict = pickle.load(open(current_model.strip(), 'rb'))
        model = model_dict['model']
        scaler = model_dict['fitted_scaler_x']

        new_rows = []
        for row in rows:
            new_row =[]
            for i in range(0,len(col_index_to_consider)):
                new_row.append(row[col_index_to_consider[i]])
            
            pred_input = pd.DataFrame([new_row],columns=input_cols)
            if scaler!="None" and scaler is not None:
                pred_input_scaled = scaler.transform(pred_input)
            else:
                pred_input_scaled = pred_input
        
            if model_dict['model_abbr']!='NET':
                prediction_result = model.predict(pred_input_scaled)[0]
            else:
                prediction_result = float(model.predict(pred_input_scaled)[0][0])
                
            predictions.append(prediction_result)
            new_row = row+[prediction_result]
            new_rows.append(new_row)

        response_to_send = {}
        response_to_send['new_rows']=new_rows
        header_list = header_str+['(Predicted) '+target_col]
        response_to_send['header']=header_list
        
        print(response_to_send)
        self.write(json.dumps(response_to_send))

def main():
    
    print("\n * ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists ")
    print(" * Web Server ver 0.1 \n")
    print(" programmed by Matt Sangkeun Lee (lees4@ornl.gov) ")
    print(" please go to : http://localhost:7777/")

    parse_command_line()
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/open_file", OpenFileHandler),
            (r"/feature_analysis", FeatureAnalysisHandler),
            (r"/ml_analysis", MLAnalysisHandler),
            (r"/get_model_file_list",GetModelFileListHandler),
            (r"/get_preset_file_list",GetPresetFileListHandler),
            (r"/execute_ml_analysis", ExecuteMLAnalysisHandler),
            (r"/execute_ml_tuning", ExecuteMLTuningHandler),
            (r"/save_model", SaveModelHandler),
            (r"/get_model_info", GetModelInfoHandler),
            (r"/predict_page", PredictPageHandler),
            (r"/delete_model",DeleteModelHandeler),
            (r"/get_predicted_target",GetPredictedTarget)
            ],
        
        cookie_secret="cookingpapamattlee",
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        xsrf_cookies=False,
        debug=options.debug,

        )
    app.listen(options.port)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()

