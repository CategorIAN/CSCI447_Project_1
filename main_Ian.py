from Iris import Iris
from SoyBean import SoyBean
from BreastCancer import BreastCancer
from Glass import Glass
import pandas as pd


if __name__ == '__main__':
    I = Iris()
    for i in range(20):
        print(I.value(i))
