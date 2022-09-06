import pandas as pd
import os
import numpy as np

#function to find Missing data from dataset
def findMissing(col_name, df):
    print(df.loc[df[col_name]=='?'])
        
class BreastCancer:
    
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\breast-cancer-wisconsin.csv')
        columns = [    
            "Id",
            "Clump Thickness",
            "Uniformity of Cell Size" ,
            "Uniformity of Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "Class"
            ]
    
        for col_names in df.columns:
            findMissing(col_names, df)
        df.columns = columns
        self.df = df