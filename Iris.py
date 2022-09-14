from NaiveBayes import NaiveBayes as NB   

def bin(df, col_name, values):
    for val in values:
        if(isinstance(val, float)):
            df[col_name] = df[col_name].replace([val], round(val))

class Iris (NB):
    def __init__(self):
        features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        iris = NB(file = 'iris.csv', features = features, name = "Iris",  classLoc= 'end')
        
        for col_names in iris.df: #get rid of continuous values
            bin(iris.df, col_names,iris.df[col_names].values)
        
        iris.test()




















