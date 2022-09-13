from NaiveBayes import NaiveBayes as NB  

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

        glass.test()



