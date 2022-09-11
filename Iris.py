import pandas as pd
import os
import math
import numpy as np
import random


class Iris:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\iris.csv')
        self.features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        df.columns = self.features + ['Class']
        self.df = df
        self.k_fold_partition(10)
        self.training_test_sets(0)
        self.seed = random.random()

    def getNoise(self):
        d = self.df.to_dict()
        random.seed(self.seed)
        noise_features = random.sample(self.features, k = math.ceil(len(self.features) * .1))
        for feature in noise_features:
            random.shuffle(d[feature])
        self.noisy_df = pd.DataFrame(d)

    def k_fold_partition(self, k):
        n = self.df.shape[0]
        p = []
        q = n // k
        r = n % k
        j = 0
        for i in range(r):
                p.append(list(range(j, j + q + 1)))
                j += q + 1
        for i in range(r, k):
                p.append(list(range(j, j + q)))
                j += q
        self.p = p

    def training_test_sets(self, j):
        train = []
        for i in range(len(self.p)):
            if j != i: train += self.p[i]
            else: test = self.p[i]
        self.train = self.df.filter(items = train, axis = 0)
        self.test = self.df.filter(items = test, axis = 0)

    def getQ(self):
        df = pd.DataFrame(self.train.groupby(by = ['Class'])['Class'].agg('count')).rename(columns =
                                                                                           {'Class': 'Count'})
        df['Q'] = df['Count'].apply(lambda x: x / self.train.shape[0])
        return df

    def getF(self, j, Qtrain = None):
        if Qtrain is None: Qtrain = self.getQ()
        df = self.train.groupby(by = ['Class', self.features[j]]).agg(Count = pd.NamedAgg(column = 'Class',
                                    aggfunc = 'count'))
        y = []
        for ((cl, _), count) in df['Count'].to_dict().items():
            y.append((count + 1)/(Qtrain.at[cl, 'Count'] + len(self.features)))
        df['F'] = y
        return df

    def getFs(self, Qtrain = None):
        Fs = []
        for j in range(len(self.features)):
            Fs.append(self.getF(j, Qtrain))
        return Fs

    def value(self, i):
        return self.df.iloc[i, len(self.features) * [True] + [False]]











