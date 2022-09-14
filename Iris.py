from NaiveBayes import NaiveBayes as NB  
import pandas as pd 
import numpy as np

def bin(df, col_name, n):
    if col_name != 'Class':
        df[col_name] = pd.qcut(df[col_name].rank(method = 'first'), q=n, labels=np.arange(n) + 1)

class Iris (NB):
    def __init__(self):
        features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        iris = NB(file = 'iris.csv', features = features, name = "Iris",  classLoc= 'end')
        
        for col_names in iris.df: #get rid of continuous values
            bin(iris.df, col_names, 5)
        
        iris.test()




















