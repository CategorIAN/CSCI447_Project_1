import pandas as pd
from Iris import Iris
from SoyBean import SoyBean


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    I = Iris()
    print(I.Q())
    I.Q().to_csv("Q.csv")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
