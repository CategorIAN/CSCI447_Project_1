from NaiveBayes import NaiveBayes as NB   

def findMissing(col_name, df):  #finds the missing ? and replaces with value
    df[col_name] = df[col_name].replace(['?'],['n'])

class Vote (NB):
    def __init__(self):
        features = [   #Class at beginning
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

        vote = NB(file = 'house-votes-84.csv', features = features, name = "Vote", classLoc = 'beginning')
        for col_names in vote.df.columns: #replace missing values in df
            findMissing(col_names, vote.df)

        vote.test()
                
                
        
        