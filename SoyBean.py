from NaiveBayes import NaiveBayes as NB   

class SoyBean (NB):
    def __init__(self):
        features = ['Date', 'Plant-Stand', 'Precip', 'Temp', 'Hail', 'Crop-Hist', 'Area-Damaged', 'Severity',
                   'Seed-TMT', 'Germination', 'Plant-Growth', 'Leaves', 'Leafspots-Halo', 'Leafspots-Marg',
                   'Leafspot-Size', 'Leaf-Shread', 'Leaf-Malf', 'Leaf-Mild', 'Stem', 'Lodging', 'Stem-Cankers',
                   'Canker-Lesion', 'Fruiting-Bodies', 'External Decay', 'Mycelium', 'Int-Discolor', 'Sclerotia',
                   'Fruit-Pods', 'Fruit Spots', 'Seed', 'Mold-Growth', 'Seed-Discolor', 'Seed-Size', 'Shriveling',
                   'Roots']
        soybean = NB(file = 'soybean-small.csv', features = features, name = "SoyBean", classLoc='end')
        
        soybean.test()

