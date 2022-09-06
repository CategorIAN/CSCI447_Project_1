import pandas as pd
from Iris import Iris
from SoyBean import SoyBean
from BreastCancer import BreastCancer
from Glass import Glass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    I = Iris()
    I.df.to_csv("iris_output.csv")
    S = SoyBean()
    S.df.to_csv("soybean-small_output.csv")
    B = BreastCancer()
    B.df.to_csv('breast-cancer-wisconsin_output.csv')
    G = Glass()
    G.df.to_csv('glass_output.csv')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
