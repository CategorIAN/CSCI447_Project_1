from NaiveBayes import NaiveBayes as NB   

class Iris (NB):
    def __init__(self):
        features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        iris = NB(file = 'iris.csv', features = features, name = "Iris",  classLoc= 'end')
        
        iris.test()




















