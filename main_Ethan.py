from Iris import Iris
from SoyBean import SoyBean
from BreastCancer import BreastCancer
from Glass import Glass
from Vote import Vote

if __name__ == '__main__':
    #B = BreastCancer()
    #B.df.to_csv('breast-cancer-wisconsin_output.csv')
    G = Glass()
    G.df.to_csv('glass_output.csv')
    V = Vote()
    V.df.to_csv('house-votes-84_output.csv')