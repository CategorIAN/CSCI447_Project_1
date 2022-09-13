from NaiveBayes import NaiveBayes as NB   

#function to find Missing data from dataset
def findMissing(col_name, df):
    df[col_name] = df[col_name].replace(['?'],['3']) 
        
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

        cancer = NB(file = 'breast-cancer-wisconsin.csv', features = features, name = "Cancer", classLoc = 'end')
        for col_names in cancer.df.columns: #replace missing values in df
            findMissing(col_names, cancer.df)


        cancer.test()
    
        




