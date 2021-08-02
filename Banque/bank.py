from flask import Flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pydantic import BaseModel
import pandas as pd



app = Flask(__name__)


list_quant= ['Customer_Age',
             'Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']

list_cat = ['Marital_Status','Gender','Education_Level','Income_Category','Card_Category']


class Item(BaseModel):
    
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon : int 
    Total_Revolving_Bal:int 
    Total_Amt_Chng_Q4_Q1:int
    Total_Trans_Ct:int
    Total_Ct_Chng_Q4_Q1:int


 # Charger Encoder
enc = pickle.load(open('encodage.sav', 'rb'))
    
# Charger Normalisation
robust = pickle.load(open('normalisation.sav', 'rb'))
    
# Selection de Variable
selection = pickle.load(open('Select_Chi2.sav', 'rb'))

#Charger model
model = pickle.load(open('Model.sav', 'rb'))

@app.route('/')

def home():
    return render_template


@app.route('/predict', methods=['POST'])
def predict(client):
    
    client = {

    'Total_Relationship_Count': client.Total_Relationship_Count,
    'Months_Inactive_12_mon': client.Months_Inactive_12_mon,
    'Contacts_Count_12_mon' : client.Contacts_Count_12_mon, 
    'Total_Revolving_Bal': client.Total_Revolving_Bal,
    'Total_Amt_Chng_Q4_Q1':client.Total_Amt_Chng_Q4_Q1,
    'Total_Trans_Ct':client.Total_Trans_Ct,
    'Total_Ct_Chng_Q4_Q1':client.Total_Ct_Chng_Q4_Q1
    
    
    # données
    d = pd.DataFrame([list(client.values())], columns=list(client.keys()))
    
    # Charger Encoder
    enc = pickle.load(open('encodage.sav', 'rb'))
    
    # Charger Normalisation
    robust = pickle.load(open('normalisation.sav', 'rb'))
    
    # Selection de Variable
    selection = pickle.load(open('Select_Chi2.sav', 'rb'))
    
    #Charger model
    model = pickle.load(open('Model.sav', 'rb'))
    
    #Encodage OneHot
    
    enc_cat = enc.transform(d[list_cat])
    
    d_cat = pd.DataFrame(enc_cat, columns = enc.get_feature_names())
    
    # Normalisation
    d_numeric = d[list_quant].copy()
    
    d_numeric[list_quant] = pd.DataFrame(robust.transform(d[list_quant]), columns=list_quant)
    
    #Concatener donnée finale
    d_final = pd.merge(d_numeric, d_cat, right_index=True, left_index=True)
    
    # Selection de variable Anova/Khi 2
#     var_select = selection.transform(d_final)
    
#     d_ = pd.DataFrame(selection.inverse_transform(var_select), index=d_final.index, columns=d_final.columns)
    
#     list_variable = d_.columns[d_.var() != 0]
    
    var_import = pd.DataFrame(selection.get_support(), d_final.columns).T

    list_variable = []
    for col in var_import:
        if var_import.loc[0, col]:
            list_variable.append(col)
    
    # prediction
    result = model.predict(d_final[list_variable])
    # return the result
    if result == 1:
        output = "Est susceptible de se desabonner"
    else:
        output = "Ne va pas se désabonner"
    return render_template('index.html', prediction_text='Le client {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)