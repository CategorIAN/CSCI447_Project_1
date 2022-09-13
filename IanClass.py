import pandas as pd
import os
import math
import random

# This class encapsulates the data to learn from as well as methods used to learn based on a representation bias
# 1. getNoise: returns a copy of the data that has added noise
# 2. partition(k): returns a k-fold partition of the indices of the data
# 3. training_test_sets(j, partition) sets the training set and test set from an index of a partition
# 4. getQ(): returns the Q dataframe indexed by the classes of the data
# 5. getF(j, Qtrain): returns the F dataframe for the jth feature using a Q dataframe
# 6. getFs(Qtrain): returns a list of F dataframes for each feature of the data
# 7. value(i): returns the ith value of the data, which includes the features but excludes the class
# 8. C(cl, x, Qtrain, Ftrains): returns real value for a given class cl and value x using trained Q and Fs
# 9. predicted_class(x, Qtrain, Ftrains): returns predicted class for a given value using trained Q and Fs
# 10. test(): creates two csv files that show predicted classes for each value for each train/test partition
class IanClass:

    def __init__(self, file, features, name, discretize = False):
        df = pd.read_csv(os.getcwd() + r'\Raw Data' + '\\' + file)  #creates a dataframe from the raw data
        self.features = features                                    #defines the features of the data
        df.columns = self.features + ['Class']                      #names the columns from the features and the class
        self.df = df
        if discretize: self.discrete()                              #defines the dataframe for the class
        self.seed = random.random()                                 #creates a random seed
        self.name = name                                            #gives the object a name

    def __str__(self):
        return self.name

    def discrete(self):
        new_df = pd.DataFrame({})
        new_features = []
        for f in self.features:
            c = self.df[f]
            new_df[f + "_Int"] = c.apply(lambda x: math.floor(x))
            new_features.append(f + "_Int")
            new_df[f + "_Dec"] = c.apply(lambda x: int(10 * (x - math.floor(x))))
            new_features.append(f + "_Dec")
        new_df['Class'] = self.df['Class']
        self.df = new_df
        self.features = new_features

    def getNoise(self):
        d = self.df.to_dict()
        random.seed(self.seed)
        noise_features = random.sample(self.features, k=math.ceil(len(self.features) * .1))
        for feature in noise_features:
            random.shuffle(d[feature])
        return pd.DataFrame(d)

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
        return self.df.iloc[i, len(self.features) * [True] + [False]]

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

    def zero_one_loss(self, predicted, actual):
        return int(predicted == actual)

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
                zero_one_losses = []
                for i in range(len(range(self.df.shape[0]))):
                    predicted = self.predicted_class(self.value(i), Qtrain, Ftrains)
                    actual = self.df.at[i, 'Class']
                    predicted_classes.append(predicted)
                    zero_one_losses.append(self.zero_one_loss(predicted, actual))
                data["Pred_{}".format(j)] = predicted_classes
                data["Pred_{}_Zero_One_Loss".format(j)] = zero_one_losses
            data.to_csv(file_name)