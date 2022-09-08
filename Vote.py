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