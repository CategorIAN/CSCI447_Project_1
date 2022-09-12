from IanClass import IanClass

class SoyBean (IanClass):
    def __init__(self):
        features = ['Date', 'Plant-Stand', 'Precip', 'Temp', 'Hail', 'Crop-Hist', 'Area-Damaged', 'Severity',
                   'Seed-TMT', 'Germination', 'Plant-Growth', 'Leaves', 'Leafspots-Halo', 'Leafspots-Marg',
                   'Leafspot-Size', 'Leaf-Shread', 'Leaf-Malf', 'Leaf-Mild', 'Stem', 'Lodging', 'Stem-Cankers',
                   'Canker-Lesion', 'Fruiting-Bodies', 'External Decay', 'Mycelium', 'Int-Discolor', 'Sclerotia',
                   'Fruit-Pods', 'Fruit Spots', 'Seed', 'Mold-Growth', 'Seed-Discolor', 'Seed-Size', 'Shriveling',
                   'Roots']
        IanClass.__init__(self, file = 'soybean-small.csv', features = features, name = "SoyBean")

