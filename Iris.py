import pandas as pd
import os
import numpy as np

class Iris:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\iris.csv')
        columns = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)', 'Class']
        df.columns = columns
        self.df = df

