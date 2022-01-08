#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 07:03:23 2021

@author: jinshengdan
"""

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from sklearn.linear_model import Ridge, Lasso, LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import uuid 
import os


app = Flask(__name__)

# @app.route("/") # Start here
@app.route("/",methods=['GET','POST']) # We need to change the first line to include GET and POST methods

def hello_world():
    request_type_str = request.method
    if request_type_str=='GET':
        return render_template("index.html",href="static/baseimage.svg")
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "app/static/"+random_string +".svg"

        # Load and Create Dataframe
        #import pandas as pd
        filename = 'app/ObesityDataSet_raw_and_data_sinthetic.csv'
        df = pd.read_csv(filename, header=0)
        df = pd.DataFrame(df)
        #boston = load_boston()
        #df = pd.DataFrame(boston.data, columns = boston.feature_names)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        df["family_history_with_overweight"] = le.fit_transform(df["family_history_with_overweight"])
        df["FAVC"] = le.fit_transform(df["FAVC"])
        df["CAEC"] = le.fit_transform(df["CAEC"])
        df["SMOKE"] = le.fit_transform(df["SMOKE"])
        df["SCC"] = le.fit_transform(df["SCC"])
        df["CALC"] = le.fit_transform(df["CALC"])
        df["MTRANS"] = le.fit_transform(df["MTRANS"])
        df["NObeyesdad"] = le.fit_transform(df["NObeyesdad"])
        
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],
                                        df.iloc[:,-1],
                                        test_size=0.2,
                                        random_state=820)
        
        
        #df['PRICE'] = boston.target

        # Split the data frame
        #X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25)
        
        # Load the model with details
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
          print(f)
        np_arr = floatsome_to_np_array(text).reshape(1, -1)
        pkl_filename="app/TrainedModel/AssignmentPickle.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        plot_graphs(model=pickle_model,new_input_arr=np_arr,output_file= path)
        return render_template("index.html",href=path[4:])


def plot_graphs(model,new_input_arr, output_file):
    
    import matplotlib
    matplotlib.use('Agg')
    
    filename = 'app/ObesityDataSet_raw_and_data_sinthetic.csv'
    df = pd.read_csv(filename, header=0)
    df = pd.DataFrame(df)
    #boston = load_boston()
    #df = pd.DataFrame(boston.data, columns = boston.feature_names)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df["family_history_with_overweight"] = le.fit_transform(df["family_history_with_overweight"])
    df["FAVC"] = le.fit_transform(df["FAVC"])
    df["CAEC"] = le.fit_transform(df["CAEC"])
    df["SMOKE"] = le.fit_transform(df["SMOKE"])
    df["SCC"] = le.fit_transform(df["SCC"])
    df["CALC"] = le.fit_transform(df["CALC"])
    df["MTRANS"] = le.fit_transform(df["MTRANS"])
    df["NObeyesdad"] = le.fit_transform(df["NObeyesdad"])
    
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],
                                    df.iloc[:,-1],
                                    test_size=0.2,
                                    random_state=820)
    
    
    # get the classes
    n_class = len(set(y_test))
    #print('We have %d classes in the data'%(n_class))
    target_names = list(df.NObeyesdad.unique())
    feature_names=list(X_train.columns.values)
    colors = ['green','orange','brown','bisque','tomato','blue','purple']
    symbols = ['o', '^', '*','v','1','p','D']
    plt.figure(figsize = (10,8))
    for i, c, s in (zip(range(n_class), colors, symbols)):
        ix = df.iloc[:,-1] == i
        plt.scatter(df.iloc[:, 2][ix],df.iloc[:, 3][ix], \
                    color = c, marker = s, s = 60, \
                    label = target_names[i])
    
    plt.legend(loc = 2, scatterpoints = 1)
    plt.xlabel('Feature 1 - ' + feature_names[2])
    plt.ylabel('Feature 2 - ' + feature_names[3])
    #outputfile = 'baseimage.svg'
    #plt.savefig(outputfile,width=1200)
    #plt.show()
    
    
    colors = ['green','orange','brown','bisque','tomato','blue','purple']
    symbols = ['o', '^', '*','v','1','p','D']
    
    fig = plt.subplot(1,1,1
    # subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
    )


    new_preds = model.predict(new_input_arr)
    # print(new_preds)
    Height_input = np.array(new_input_arr[0][2])
    # print(RM_input)
    Weight_input =np.array(new_input_arr[0][3])
    # print(LSTAT_input)
    
    if new_preds[0]==1:
        cols = colors[0]
        syms = symbols[0]
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)

    
    if new_preds[0]==5:
        cols = colors[1]
        syms = symbols[1]
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)
        
        
    if new_preds[0]==6:
        cols = colors[2]
        syms = symbols[2]
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)
        
    if new_preds[0]==2:
        cols = colors[3]
        syms = symbols[3]
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)
        
    if new_preds[0]==0:
        cols = colors[4]
        syms = symbols[4] 
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)
        
    if new_preds[0]==3:
        cols = colors[5]
        syms = symbols[5]
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)
        
    if new_preds[0]==4:
        cols = colors[6]
        syms = symbols[6]
        fig.scatter(Height_input, Weight_input,color='black',marker=syms,s=900)
        fig.figure.savefig(output_file)
    


    # Update xaxis properties  
    #fig.update_xaxes(title_text="Height", row=1, col=1)
    # Update yaxis properties
    #fig.update_yaxes(title_text="Weight", row=1, col=1)
    # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    # Update title and height
    #fig.update_layout(height=600, width=1400, title_text="Obsity Level")
    #fig.write_image(output_file,width=1200,engine="kaleido")
    #fig.show()


def floatsome_to_np_array(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)




#if __name__ == "__main__":
#    app.run(debug=True)


### 0,23.0,1.65,54.0,1,0,3.0,1.0,2,1,2.0,0,2.0,1.0,1,3
### 1,20.0,1.75,68.2,1,0,2.0,1.0,1,0,2.0,1,2.0,1.0,2,1
### 1,61.0,1.98,173.0,1,1,2.0,3.0,2,1,2.0,1,2.0,1.0,3,2
### 1,55.0,1.75,160.0,0,0,2.0,4.0,0,0,2.0,2,2.0,0.0,0,4
### 1,50.0,1.88,100.0,1,0,2.0,3.0,1,0,2.0,1,2.0,0.0,1,3


