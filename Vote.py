import pandas as pd
import os
import numpy as np

def findMissing(col_name, df):
    df[col_name] = df[col_name].replace(['?'],['n']) #maybe place random no or yes
    

class Vote:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\house-votes-84.csv')
        columns = [   
            'Class Name',
            'handicapped-infants',
            'water-project-cost-sharing',
            'adoption-of-the-budget-resolution',
            'physician-fee-freeze',
            'el-salvador-aid',
            'religious-groups-in-schools',
            'anti-satellite-test-ban',
            'aid-to-nicaraguan-contras',
            'mx-missile',
            'immigration',
            'synfuels-corporation-cutback',
            'education-spending',
            'superfund-right-to-sue',
            'crime',
            'duty-free-exports',
            'export-administration-act-south-africa'
        ]
        df.columns = columns
        for col_names in columns:
            findMissing(col_names, df)
        self.df = df
        
        p_cancer = df.groupby('Class Name').size().div(len(df)) #count()['Age']/len(data)
        
        likelihood = {}
        
        for col_name in columns:
            if (col_name != 'Class Name'):
                likelihood[col_name] = df.groupby(['Class Name', col_name]).size().div(len(df)).div(p_cancer) 
                
                
        
        