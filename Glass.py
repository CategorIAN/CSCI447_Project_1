from NaiveBayes import NaiveBayes as NB  

def bin(df, col_name, values):
    for val in values:
        if(isinstance(val, float)):
            df[col_name] = df[col_name].replace([val], round(val))

class Glass (NB):
    def __init__(self):
        features = [ #Class at end
            "Id number: 1 to 214",
            "RI: refractive index",
            "Na: Sodium",
            "Mg: Magnesium",
            "Al: Aluminum",
            "Si: Silicon",
            "K: Potassium",
            "Ca: Calcium",
            "Ba: Barium",
            "Fe: Iron"
            ]
        glass = NB(file = 'glass.csv', features = features, name = 'Glass', classLoc = 'end')
        
        for col_names in glass.df: #get rid of continuous values
            bin(glass.df, col_names,glass.df[col_names].values)

        glass.test()



