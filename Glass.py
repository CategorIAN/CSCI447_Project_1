from NaiveBayes import NaiveBayes as NB  
import pandas as pd
import numpy as np

def bin(df, col_name, n):
    if col_name != 'Class':
            df[col_name] = pd.qcut(df[col_name].rank(method = 'first'), q=n, labels=np.arange(n) + 1)

class Glass (NB):
    def __init__(self):
        features = [ #Class at end
            "Id number: 1 to 214",
            "RI: refractive index",
            "Na: Sodium",
            "Mg: Magnesium",
            "Al: Aluminum",
            "Si: Silicon",
            "K: Potassium",
            "Ca: Calcium",
            "Ba: Barium",
            "Fe: Iron"
            ]
        glass = NB(file = 'glass.csv', features = features, name = 'Glass', classLoc = 'end')
        
        for col_names in glass.df: #get rid of continuous values
            bin(glass.df, col_names, 5)

        glass.test()



