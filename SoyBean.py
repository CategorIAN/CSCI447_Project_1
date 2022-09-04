import pandas as pd
import os
import numpy as np

class SoyBean:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\soybean-small.csv')
        columns = ['Date', 'Plant-Stand', 'Precip', 'Temp', 'Hail', 'Crop-Hist', 'Area-Damaged', 'Severity',
                   'Seed-TMT', 'Germination', 'Plant-Growth', 'Leaves', 'Leafspots-Halo', 'Leafspots-Marg',
                   'Leafspot-Size', 'Leaf-Shread', 'Leaf-Malf', 'Leaf-Mild', 'Stem', 'Lodging', 'Stem-Cankers',
                   'Canker-Lesion', 'Fruiting-Bodies', 'External Decay', 'Mycelium', 'Int-Discolor', 'Sclerotia',
                   'Fruit-Pods', 'Fruit Spots', 'Seed', 'Mold-Growth', 'Seed-Discolor', 'Seed-Size', 'Shriveling',
                   'Roots', 'Class']
        df.columns = columns
        self.df = df