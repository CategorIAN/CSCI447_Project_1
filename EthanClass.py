import pandas as pd
import os
import math
import random

class EthanClass:



    def __init__(self, file, features, name, classLoc):
        
        df = pd .read_csv(os.getcwd() + r'\Raw Data' + '\\' + file)
        self.features = features
        df.columns
        if(classLoc == 'beginning'):
            df.columns = ['Class'] + self.features
        elif(classLoc == 'end'):
            df.columns = self.features + ['Class']
        else:
            print('Not sure where to place Class column')
        self.df = df
        self.seed = random.random()
        self.name = name

    def __str__(self):
        return self.name
    
    def getNoise(self):
        df = self.df.to_dict()
        random.seed(self.seed)
        noise_features = random.sample(self.features, k=math.ceil(len(self.features) * .1))
        for feature in noise_features:
            random.shuffle(df[feature])
        return pd.DataFrame(df)

    def partition(self, k):
        n = self.df.shape[0]
        (q, r) = (n // k, n % k)
        (p, j) = ([], 0)
        for i in range(r):
                p.append(list(range(j, j + q + 1)))
                j += q + 1
        for i in range(r, k):
                p.append(list(range(j, j + q)))
                j += q
        return p

    def training_test_sets(self, j, partition = None):
        if partition is None: partition = self.partition(10)
        train = []
        for i in range(len(partition)):
            if j != i: train += partition[i]
            else: test = partition[i]
        self.train_set = self.df.filter(items = train, axis = 0)
        self.test_set = self.df.filter(items = test, axis = 0)

    def getQ(self):
        df = pd.DataFrame(self.train_set.groupby(by = ['Class'])['Class'].agg('count')).rename(columns =
                                                                                               {'Class': 'Count'})
        df['Q'] = df['Count'].apply(lambda x: x / self.train_set.shape[0])
        return df

    def getF(self, j, Qtrain = None):
        if Qtrain is None: Qtrain = self.getQ()
        df = pd.DataFrame(self.train_set.groupby(by = ['Class', self.features[j]])['Class'].agg('count')).rename(
                                                                                        columns = {'Class' : 'Count'})
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
        return self.df.iloc[i, (len(self.features)) * [True] + [False]]

    def C(self, cl, x, Qtrain = None, Ftrains = None):
        if Qtrain is None: Qtrain = self.getQ()
        if Ftrains is None: Ftrains = self.getFs(Qtrain)
        result = Qtrain.at[cl, 'Q']
        for j in range(len(self.features)):
            F = Ftrains[j]
            if (cl, x[j]) in F.index:
                result = result * F.at[(cl, x[j]), 'F']
            else: return 0
        return result

    def predicted_class(self, x, Qtrain = None, Ftrains = None):
        if Qtrain is None: Qtrain = self.getQ()
        if Ftrains is None: Ftrains = self.getFs(Qtrain)
        (argmax, max_C) = (None, 0)
        for cl in Qtrain.index:
            y = self.C(cl, x, Qtrain, Ftrains)
            if y > max_C:
                argmax = cl
                max_C = y
        return argmax

    def test(self):
        p = self.partition(10)
        pred_df = pd.DataFrame(self.df.to_dict())
        pred_df_noise = self.getNoise()
        dfs = [(pred_df, "{}_Pred.csv".format(str(self))), (pred_df_noise, "{}_Pred_Noise.csv".format(str(self)))]
        for (data, file_name) in dfs:
            for j in range(len(p)):
                self.training_test_sets(j, p)
                Qtrain = self.getQ()
                Ftrains = self.getFs(Qtrain)
                predicted_classes = []
                for i in range(len(range(self.df.shape[0]))):
                    predicted_classes.append(self.predicted_class(self.value(i), Qtrain, Ftrains))
                data["Pred_{}".format(j)] = predicted_classes
            data.to_csv(file_name)
    


    # def getClassProb(self):
    #     p_class = self.df.groupby('Class').size().div(len(self.df))
    #     return p_class

    # def getFeatureProb(self, j, Qtrain = None):
    #     if Qtrain is None: Qtrain = self.getClassProb() # create class probabilities if not already created
    #     if (self.features[j] == 'Class'):
    #         return
    #     df = self.df.groupby(['Class', self.features[j]]).size().div(len(self.df)).div(Qtrain) 
    #     return df

    # def getFeatureProbSet(self, Qtrain = None):
    #     Fs = []
    #     for j in range(len(self.features)):
    #         Fs.append(self.getFeatureProb(j, Qtrain))
        
    #     return Fs

    