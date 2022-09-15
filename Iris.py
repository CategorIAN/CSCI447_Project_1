from NaiveBayes import NaiveBayes as NB  
import pandas as pd 
import numpy as np

class Iris (NB):
    def __init__(self):
        features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        super().__init__(file = 'iris.csv', features = features, name = "Iris",  classLoc= 'end')




















