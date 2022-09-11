import pandas as pd
import os
import math
import numpy as np

class Iris:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\iris.csv')
        self.features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        df.columns = self.features + ['Class']
        self.df = df
        self.k_fold_partition(10)
        self.training_test_sets(0)


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

    def Q(self):
        return self.train.groupby(by = ['Class'])['Class'].count().apply(lambda x: x / self.train.shape[0])

    def F(self, j):
        return self.train.groupby(by = ['Class', self.features[j]]).agg(Count = pd.NamedAgg(column = 'Class',
                                                                                               aggfunc = 'count'))





