from asyncio.windows_events import NULL
import pandas as pd
import os
import math
import random
from ConfusionMatrix import ConfusionMatrix
import numpy as np

class NaiveBayes:

    def __init__(self, file, features, name, classLoc, replaceValue = None): 
        
        df = pd .read_csv(os.getcwd() + r'\Raw Data' + '\\' + file)
        self.df = df #dataframe
        self.features = features 
        self.name = name
        self.addColumnNames(classLoc) #add column names to correct spot
        self.classes = list(set(self.df['Class']))
        if replaceValue: self.findMissing(replaceValue) #replace missing values
        self.df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_w_colnames.csv".format(str(self))) #create csv of file
        self.seed = random.random()


    #print out the name of the file
    def __str__(self):
        return self.name 

    #create the column names 
    def addColumnNames(self, classLoc):
        if (classLoc == 'beginning'):  # if the class column is at the beginning
            self.df.columns = ['Class'] + self.features
            # shift the class column to the last column
            last_column = self.df.pop('Class')
            self.df.insert(len(self.df.columns), 'Class', last_column)
        elif (classLoc == 'end'):  # if the class column is at the end -> continue as normal
            self.df.columns = self.features + ['Class']
        else:
            print('Not sure where to place Class column')

    # function to find Missing data from dataset
    def findMissing(self, replaceValue):
        for col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].replace(['?'], [replaceValue])

    #bin the data into a certain amount of equal sections
    def bin(self, df, n):
        binned_df = pd.DataFrame(df.to_dict())
        try:
            binned_df[self.features] = binned_df[self.features].apply(pd.to_numeric, axis=1)
            for col_name in self.features:  # get rid of continuous values
                binned_df[col_name] = pd.qcut(df[col_name].rank(method='first'), q=n, labels=np.arange(n) + 1)
        except:
            pass
        return binned_df


    #create Noise in the dataset
    def getNoise(self):
        df = self.df.to_dict()
        random.seed(self.seed)
        noise_features = random.sample(self.features, k=math.ceil(len(self.features) * .1))
        for feature in noise_features:
            random.shuffle(df[feature])
        return pd.DataFrame(df)

    #partition the dataset into k partitions
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

    #separate into training and test sets
    def training_test_sets(self, j, df, partition = None):
        if partition is None: partition = self.partition(10)
        train = []
        for i in range(len(partition)):
            if j != i: train += partition[i]
            else: test = partition[i]
        self.train_set = df.filter(items = train, axis = 0)
        self.test_set = df.filter(items = test, axis = 0)

    #probability of the class
    def getQ(self):
        df = pd.DataFrame(self.train_set.groupby(by = ['Class'])['Class'].agg('count')).rename(columns =
                                                                                               {'Class': 'Count'})
        df['Q'] = df['Count'].apply(lambda x: x / self.train_set.shape[0])
        return df

    #probability of a sigle feature 
    def getF(self, j, m, p, Qtrain = None): 
        if Qtrain is None: Qtrain = self.getQ()
        df = pd.DataFrame(self.train_set.groupby(by = ['Class', self.features[j]])['Class'].agg('count')).rename(
                                                                                        columns = {'Class' : 'Count'})
        y = []
        for ((cl, _), count) in df['Count'].to_dict().items():
            y.append((count + 1 + m*p)/(Qtrain.at[cl, 'Count'] + len(self.features) + m)) 
        df['F'] = y
        return df

    #probability of a set of features
    def getFs(self, m, p, Qtrain = None):
        Fs = []
        for j in range(len(self.features)):
            Fs.append(self.getF(j, m, p, Qtrain))
        return Fs

    #return value of a ceratin feature
    def value(self, df, i):
        return df.loc[i, self.features]

    #Calculate the probabilities of the class based on the features
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

    #predict the class value
    def predicted_class(self, x, Qtrain = None, Ftrains = None):
        if Qtrain is None: Qtrain = self.getQ()  #create class if not there
        if Ftrains is None: Ftrains = self.getFs(Qtrain)  #create feature set if not there
        (argmax, max_C) = (None, 0) 
        for cl in Qtrain.index:
            y = self.C(cl, x, Qtrain, Ftrains)
            if y > max_C:
                argmax = cl
                max_C = y
        return argmax
    
    #detirmine if the predicted class and actual class are the same
    def zero_one_loss(self, predicted, actual): 
        return int(predicted == actual)


    #test a certain set of data, tuning it a certain amount of times, with a starting bin_number and starting m_val
    def test(self, tuning, bin_number, m_val):
        p = self.partition(10)
        pred_df = pd.DataFrame(self.df.to_dict())
        pred_df_noise = self.getNoise()
        evaluation_df = pd.DataFrame(columns=['Noise?', 'Bin_Number', 'Test_Set', 'M_Value', 'Prob_Value', 'Zero_One_Loss_Avg', 'P_Macro'])
        dfs = [(pred_df, "{}_Pred".format(str(self)), False),
               (pred_df_noise, "{}_Pred_Noise".format(str(self)), True)]
        
        #go through normal df and df with noise
        for (data, file_name, noise) in dfs:

            #go through each bin number
            for b in range(bin_number, bin_number + tuning): 
                
                binned_df = self.bin(df = data, n = b) #create new dataframe for each bin number
                
                for tun in range(m_val, tuning + m_val):
                    
                    m = tun+1 #value to be tuned for above 1
                    prob = 1/(100*(tun+1)) #another probability that will 
                    
                
                    #partition 10 times
                    for j in range(len(p)):
                        self.training_test_sets(j, binned_df, p) #create partitions
                        Qtrain = self.getQ()
                        Ftrains = self.getFs(m, prob, Qtrain)
                        predicted_classes = []
                        zero_one_sum = 0
                        CM = ConfusionMatrix(self.classes) #Create Confusion Matrix for loss function
                        
                        #go through each row
                        for i in range(binned_df.shape[0]): 
                            #if there is a partition
                            if i in p[j]:
                                predicted = self.predicted_class(self.value(binned_df, i), Qtrain, Ftrains) #predicted class value associated with the row
                                actual = binned_df.at[i, 'Class'] #the class value associated with the row
                                predicted_classes.append(predicted) 
                                zero_one_sum += self.zero_one_loss(predicted, actual)
                                CM.addOne(predicted, actual)
                            else:
                                predicted_classes.append(None)
                        binned_df["Pred_{}".format(b)] = predicted_classes #populate the bin df with values
                        zero_one_avg = zero_one_sum / len(p[j]) #calculate zero/one sum
                        evaluation_df.loc[len(evaluation_df)] = [noise, b, j, m, prob, zero_one_avg, CM.pmacro()]
                    
                binned_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + file_name
                                 + '_B{}'.format(b) + '.csv')
                
                
        evaluation_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Eval.csv".format(str(self)))
        analysis_df = evaluation_df.groupby(by = ['Bin_Number', 'M_Value', 'Prob_Value'])[['Zero_One_Loss_Avg', 'P_Macro']].agg('mean').rename(
            columns = {'P_Macro': 'P_Macro_Avg'})
        analysis_df["Average"] = .5 * (analysis_df['Zero_One_Loss_Avg'] + analysis_df['P_Macro_Avg'])
        analysis_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Analysis.csv".format(str(self)))
        analysis_df.reset_index(inplace=True)
        analysis_df.insert(0, 'Data', analysis_df.shape[0] * [str(self)])
        self.analysis_df = analysis_df
    

    