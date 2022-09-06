import pandas as pd
import os
import numpy as np

class Glass:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\glass.csv')
        columns = [
            "Id number: 1 to 214",
            "RI: refractive index",
            "Na: Sodium",
            "Mg: Magnesium",
            "Al: Aluminum",
            "Si: Silicon",
            "K: Potassium",
            "Ca: Calcium",
            "Ba: Barium",
            "Fe: Iron",
            "Type of glass"
            ]
        df.columns = columns
        self.df = df

