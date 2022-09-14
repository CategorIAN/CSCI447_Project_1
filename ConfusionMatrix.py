import pandas as pd

class ConfusionMatrix:
    def __init__(self, classes):
        self.classes = classes
        df = {}
        for cl in classes:
            df[cl] = len(classes) * [0]
        df = pd.DataFrame(df)
        df.index = classes
        self.df = df

    def __str__(self):
        return str(self.df)

    def addOne(self, predicted, actual):
        if predicted in self.classes and actual in self.classes:
            self.df.at[predicted, actual] += 1

    def truepositive(self, cl):
        return self.df.at[cl, cl]

    def truenegative(self, cl):
        count = 0
        for (m, n) in list(zip(self.classes)):
            if m != cl and n != cl:
                count += self.df.at[m, n]
        return count

    def falsepositive(self, cl):
        return self.df.loc[cl, :].sum() - self.truepositive(cl)

    def falsenegative(self, cl):
        return self.df.loc[:, cl].sum() - self.truepositive(cl)

    def pmicro(self):
        numerator = 0
        denominator = 0
        for cl in self.classes:
            numerator += self.truepositive(cl)
            denominator += (self.truepositive(cl) + self.falsepositive(cl))
        return numerator / denominator





