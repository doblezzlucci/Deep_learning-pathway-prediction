
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pyarrow import parquet


def preprocess_data(expr, label):
    # extract caseID ,cancer type from the labels
    label["caseID"] = label["cases"].str.split('|').str[1]
    expr["caseID"].str.strip()
    lab=label[["caseID","cancer_type"]]
    print('show data labels and shape ',lab.head(),lab.shape)
    print('show expression data and shape',expr.head(), expr.shape) 
    # merge expression data with labels on caseID
    df=pd.merge(expr,lab, on='caseID', how='inner')
    
    # separate features and labels
    labels=df['cancer_type']
    features=df.drop(columns=['cancer_type','caseID'])
    unique_labels = labels.unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = labels.map(label_mapping)
    encoded_labels.to_csv(os.path.join(outputdir,'encoded_labels.csv'), index=False)
    #minMaxscaler 
    scaler = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
 
  
    # drop columns with average less than threshold <0.001
    for col in scaled_features.columns:
       average=scaled_features[col].astype(float).sum()/len(scaled_features)
       if average < 0.001:
         scaled_features=scaled_features.drop(columns=[col])
         print(f'drop column {col} with average {average}')
  #combine scaled features with encoded labels
    comb=pd.concat([scaled_features, encoded_labels], axis=1)
  # train test split 70 30 stratified of data based on encoded labels
    comb_x_tr,comb_x_rst,comb_y_tr,comb_y_rst=train_test_split(comb.iloc[:,:-1],comb.iloc[:,-1] ,test_size=0.3, random_state=42, stratify=encoded_labels)
   # train test split 50 50 of rest data for finetune and test
    comb_x_val,comb_x_te,comb_y_val,comb_y_te=train_test_split(comb_x_rst,comb_y_rst ,test_size=0.5, random_state=42, stratify=comb_y_rst)
  
    return comb_x_tr,comb_y_tr,comb_x_val,comb_y_val,comb_x_te,comb_y_te

def save_processed_data(outputdir,comb_x_tr,comb_y_tr,comb_x_val,comb_y_val,comb_x_te,comb_y_te):
     # save processed data to parquet files
    comb_x_tr.to_csv(os.path.join(outputdir,'comb_x_tr.csv'), index=False)
    comb_y_tr.to_csv(os.path.join(outputdir,'comb_y_tr.csv'), index=False)
    comb_x_val.to_csv(os.path.join(outputdir,'comb_x_val.csv'), index=False)
    comb_y_val.to_csv(os.path.join(outputdir,'comb_y_val.csv'), index=False)
    comb_x_te.to_csv(os.path.join(outputdir,'comb_x_te.csv'), index=False)
    comb_y_te.to_csv(os.path.join(outputdir,'comb_y_te.csv'), index=False)
    print('Processed data saved to', outputdir,'as (comb_x_tr.csv , comb_y_tr.csv) for pretraining, (comb_x_val.csv, comb_y_val.csv) for test  (comb_x_te.csv, comb_y_te.csv) for finetuning')