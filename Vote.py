import pandas as pd
import os
from math import sqrt
from math import exp
from math import pi

def findMissing(col_name, df):
    df[col_name] = df[col_name].replace(['?'],['n']) #maybe place random no or yes
    
def replaceValues(col_name, df):
    df[col_name] = df[col_name].replace(['y'], [1])
    df[col_name] = df[col_name].replace(['n'], [0])
   
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0   
    
def separate_by_class(df):
    return df.groupby("Class Name")
    
    # separated = dict()
    # for i in range(len(df)):
    #     vector = df.iloc[i]
    #     class_value = vector[0]
    #     if (class_value not in separated):
    #         separated[class_value] = list()
    #     separated[class_value].append(vector)
    # return separated

def mean(numbers):
    if isinstance(numbers[0], str):
        return 1
    else:  
        return sum(numbers)/float(len(numbers))

def stdev(numbers):
    if isinstance(numbers[0], str):
        return 1
    else:
        avg = mean(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return sqrt(variance)

def summarize_dataset(df):
    summaries = [(mean(df[column].values), stdev(df[column].values), len(df[column].values)) for column in df.columns]
    del(summaries[1])
    return summaries

def summarize_by_class(df):
    separated = separate_by_class(df)
    summaries = dict()
    for classes in separated.groups:
        summaries[classes] = summarize_dataset(separated.get_group(classes))
        
    #     print(type(rows))
    #     summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2*pi) * stdev)) * exponent

def calcuate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def predict(summaries, row):
    probabilities = calcuate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label
      
    

class Vote:
    def __init__(self):
        df = pd.read_csv(os.getcwd() + r'\Raw Data\house-votes-84.csv')
        columns = [   
            'Class Name',
            'handicapped-infants',
            'water-project-cost-sharing',
            'adoption-of-the-budget-resolution',
            'physician-fee-freeze',
            'el-salvador-aid',
            'religious-groups-in-schools',
            'anti-satellite-test-ban',
            'aid-to-nicaraguan-contras',
            'mx-missile',
            'immigration',
            'synfuels-corporation-cutback',
            'education-spending',
            'superfund-right-to-sue',
            'crime',
            'duty-free-exports',
            'export-administration-act-south-africa'
        ]
        df.columns = columns
        for col_names in columns:
            findMissing(col_names, df)
        for col_names in columns:
            replaceValues(col_names, df)
        self.df = df
        
        row = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
        
        model = summarize_by_class(df)
        label = predict(model, row)
        print('Data = %s, Predicted: %s' % (row, label))
        
    
        

        
    # def str_column_to_float(df, column):
    #     for row in df:
    #         row[column] = float(row[column].strip())
            
    # def str_column_to_int(df,column):
    #     class_values = [row[column] for row in df]
    #     unique = set(class_values)
    #     lookup = dict
        
    # p_cancer = df.groupby('Class Name').size().div(len(df)) #count()['Age']/len(data)
    
    # likelihood = {}
    
    # for col_name in columns:
    #     if (col_name != 'Class Name'):
    #         likelihood[col_name] = df.groupby(['Class Name', col_name]).size().div(len(df)).div(p_cancer) 
    # print(likelihood)
                
                
        
        