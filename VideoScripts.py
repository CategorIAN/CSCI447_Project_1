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

def show_trained_model(data, bin_number, m_value, partition_index):
    pd.set_option('max_columns', None)
    binned_df = data.bin(data.df, bin_number)
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    Qtrain = data.getQ()
    print("Q Frame (Class Parameter Values):")
    print(Qtrain)
    print("=======================================")
    Ftrains = data.getFs(m_value, 1/(100*(m_value)), Qtrain)
    for i in range(len(Ftrains)):
        print("F Frame {}".format(i))
        print(Ftrains[i])
        print("------------------------------")

def show_model_count(data, bin_number, m_value, partition_index):
    pd.set_option('max_columns', None)
    binned_df = data.bin(data.df, bin_number)
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    Qtrain = data.getQ()
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    for cl in data.classes:
        print("There are {} values for class {}".format(Qtrain.at[cl, 'Count'], cl))
        print("----------------------------------------------------------------------")
        for i in range(len(data.features)):
            for bin in range(1, bin_number + 1):
                print("There are {} values in class {} with {} "
                      "in bin {}".format(Ftrains[i].at[(cl, data.features[i]), 'Count'], cl, data.features[i], bin))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("==================================================")
