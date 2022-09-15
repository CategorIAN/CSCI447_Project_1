from NaiveBayes import NaiveBayes as NB
        
class BreastCancer (NB):
    
    def __init__(self):
        features = [   #column names class at end
            'Id',
            'Clump Thickness',
            'Uniformity of Cell Size' ,
            'Uniformity of Cell Shape',
            'Marginal Adhesion',
            'Single Epithelial Cell Size',
            'Bare Nuclei',
            'Bland Chromatin',
            'Normal Nucleoli',
            'Mitoses'
            ]

        super().__init__(file = 'breast-cancer-wisconsin.csv', features = features,
                       name = "Cancer", classLoc = 'end', replaceValue = '3')


    
        




