from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd
from os.path import exists


def show_bins(data, bin_numbers):
    pd.set_option('max_columns', None)
    print("Original Data:")
    print(data.df.head())
    print("=================================")
    for bin_number in bin_numbers:
        binned_df = data.bin(data.df, bin_number)
        print("Binned Data with {} Bin(s):".format(bin_number))
        print(binned_df.head())
        print("---------------------------------")