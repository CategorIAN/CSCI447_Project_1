from EthanClass import EthanClass as EC

#function to find Missing data from dataset
def findMissing(col_name, df):
    
    #df.drop((df.loc[df[col_name]=='?']).index, inplace = True) #drop the row that contains the missing value
    
    df[col_name] = df[col_name].replace(['?'],['3']) #maybe place random no or yes
        
class BreastCancer:
    
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

        cancer = EC(file = 'breast-cancer-wisconsin.csv', features = features, name = "Cancer", classLoc = 'end')
        for col_names in cancer.df.columns: #replace missing values in df
            findMissing(col_names, cancer.df)


        cancer.test()
    
        




        # df.columns = columns
        # for col_names in columns:
        #     findMissing(col_names, df)
        # self.df = df
        
        # p_cancer = df.groupby('Class').size().div(len(df)) #count()['Age']/len(data)
        
        # likelihood = {}
        # likelihood['Clump Thickness'] = df.groupby(['Class', 'Clump Thickness']).size().div(len(df)).div(p_cancer)
        # likelihood['Uniformity of Cell Size'] = df.groupby(['Class', 'Uniformity of Cell Size']).size().div(len(df)).div(p_cancer)
        # likelihood['Uniformity of Cell Shape'] = df.groupby(['Class', 'Uniformity of Cell Shape']).size().div(len(df)).div(p_cancer)
        # likelihood['Marginal Adhesion'] = df.groupby(['Class', 'Marginal Adhesion']).size().div(len(df)).div(p_cancer)
        
        # print(likelihood)
