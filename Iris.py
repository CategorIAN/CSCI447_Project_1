from IanClass import IanClass

class Iris (IanClass):
    def __init__(self):
        features = ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)',
                         'Petal Width (cm)']
        IanClass.__init__(self, file = 'iris.csv', features = features, name = "Iris")




















