from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd
from os.path import exists
from ConfusionMatrix import ConfusionMatrix


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
    binned_df = data.bin(data.df, bin_number)
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    Qtrain = data.getQ()
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    for cl in data.classes:
        print("There are {} values for class {}".format(Qtrain.at[cl, 'Count'], cl))
        print("----------------------------------------------------------------------")
        for i in range(len(data.features)):
            print("Feature: {}".format(data.features[i]))
            print("+++++++++++++")
            for bin in range(1, bin_number + 1):
                print("There are {} values in class {} with {} "
                      "in bin {}".format(Ftrains[i].at[(cl, bin), 'Count'], cl, data.features[i], bin))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("==================================================")

def preds_and_evals(data, bin_number, m_value, partition_index):
    pd.set_option('max_columns', None)
    binned_df = data.bin(data.df, bin_number)
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    Qtrain = data.getQ()
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    predicted_classes = []
    zero_one_sum = 0
    CM = ConfusionMatrix(data.classes)
    result_data = pd.DataFrame(data.test_set.to_dict())
    p = data.partition(10)
    for i in p[partition_index]:
        predicted = data.predicted_class(data.value(binned_df, i), Qtrain,
                                         Ftrains)  # predicted class value associated with the row
        actual = binned_df.at[i, 'Class']  # the class value associated with the row
        predicted_classes.append(predicted)
        zero_one_sum += data.zero_one_loss(predicted, actual)
        CM.addOne(predicted, actual)
    result_data["Predicted_Class"] = predicted_classes
    print("Test Set with Predictions:")
    print(result_data)
    zero_one_avg = zero_one_sum / len(p[partition_index])
    print("Zero One Loss Average Value: {}".format(zero_one_avg))
    print("P Macro of Confusion Matrix: {}".format(CM.pmacro()))
    print("----------------------------------")


    print("Now with Noisy Data:")
    noisy_data = data.getNoise()
    binned_df = data.bin(noisy_data, bin_number)
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    Qtrain = data.getQ()
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    predicted_classes = []
    zero_one_sum = 0
    CM = ConfusionMatrix(data.classes)
    result_data = pd.DataFrame(data.test_set.to_dict())
    p = data.partition(10)
    for i in p[partition_index]:
        predicted = data.predicted_class(data.value(binned_df, i), Qtrain,
                                         Ftrains)  # predicted class value associated with the row
        actual = binned_df.at[i, 'Class']  # the class value associated with the row
        predicted_classes.append(predicted)
        zero_one_sum += data.zero_one_loss(predicted, actual)
        CM.addOne(predicted, actual)
    result_data["Predicted_Class"] = predicted_classes
    print("Test Set with Predictions:")
    print(result_data)
    zero_one_avg = zero_one_sum / len(p[partition_index])
    print("Zero One Loss Average Value: {}".format(zero_one_avg))
    print("P Macro of Confusion Matrix: {}".format(CM.pmacro()))


