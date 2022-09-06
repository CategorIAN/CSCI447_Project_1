from Iris import Iris
from SoyBean import SoyBean
from BreastCancer import BreastCancer
from Glass import Glass


if __name__ == '__main__':
    I = Iris()
    print(I.Q())
    I.Q().to_csv("Q.csv")