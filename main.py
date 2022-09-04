import pandas as pd
from Iris import Iris
from SoyBean import SoyBean


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    I = Iris()
    I.df.to_csv("iris_output.csv")
    S = SoyBean()
    S.df.to_csv("soybean-small_output.csv")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
