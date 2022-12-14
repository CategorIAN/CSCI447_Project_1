from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd
from os.path import exists
from ConfusionMatrix import ConfusionMatrix


def show_bins(data, bin_numbers):
    print("Original Data:")
    #printing original dataset
    print(data.df.head())
    print("=================================")
    #for each of the bins prints out the corresponding dataset
    for bin_number in bin_numbers:
        binned_df = data.bin(data.df, bin_number)
        print("Binned Data with {} Bin(s):".format(bin_number))
        print(binned_df.head())
        print("---------------------------------")


#training data
def show_trained_model(data, bin_number, m_value, partition_index):
    #create bin df
    binned_df = data.bin(data.df, bin_number)
    #training each partition of the binned data
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    #class probability
    Qtrain = data.getQ()
    print("Q Frame (Class Parameter Values):")
    print(Qtrain)
    print("=======================================")
    #for each feature we print feature probability based on bin number and class
    Ftrains = data.getFs(m_value, 1/(100*(m_value)), Qtrain)
    for i in range(len(Ftrains)):
        print("F Frame {}".format(i))
        print(Ftrains[i])
        print("------------------------------")

def show_model_count(data, bin_number, m_value, partition_index):
    #bin dataframe
    binned_df = data.bin(data.df, bin_number)
    #training each partition of the binned data
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    #class probability
    Qtrain = data.getQ()
    #feature probabilities for each feature 
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    for cl in data.classes:
        #print out class count
        print("There are {} values for class {}".format(Qtrain.at[cl, 'Count'], cl))
        print("----------------------------------------------------------------------")
        for i in range(len(data.features)):
            #print out the count of the feature in a certain class with a specfic bin number
            print("Feature: {}".format(data.features[i]))
            print("+++++++++++++")
            for bin in range(1, bin_number + 1):
                print("There are {} values in class {} with {} "
                      "in bin {}".format(Ftrains[i].at[(cl, bin), 'Count'], cl, data.features[i], bin))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("==================================================")

def preds_and_evals(data, bin_number, m_value, partition_index):
    #bin dataframe
    binned_df = data.bin(data.df, bin_number)
    #training each partition of the binned data
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    #class probability
    Qtrain = data.getQ()
    #feature probabilities for each feature 
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    predicted_classes = []
    zero_one_sum = 0
    CM = ConfusionMatrix(data.classes) #confusion matric for p macro
    result_data = pd.DataFrame(data.test_set.to_dict())
    p = data.partition(10)
    #print only one fold of the partitions
    for i in p[partition_index]:
        predicted = data.predicted_class(data.value(binned_df, i), Qtrain,
                                         Ftrains)  # predicted class value associated with the row
        actual = binned_df.at[i, 'Class']  # the class value associated with the row
        predicted_classes.append(predicted)
        zero_one_sum += data.zero_one_loss(predicted, actual) 
        CM.addOne(predicted, actual)
    result_data["Predicted_Class"] = predicted_classes #create predicted classes column
    print("Test Set with Predictions:")
    print(result_data) #print out df
    zero_one_avg = zero_one_sum / len(p[partition_index]) 
    print("Zero One Loss Average Value: {}".format(zero_one_avg))
    print("P Macro of Confusion Matrix: {}".format(CM.pmacro()))
    print("----------------------------------")

    #same thing for the noisy dataset
    print("Now with Noisy Data:")
    noisy_data = data.getNoise()
    binned_df = data.bin(noisy_data, bin_number)
    #training each partition of the binned data
    data.training_test_sets(partition_index, binned_df, data.partition(10))
    #class probability
    Qtrain = data.getQ()
    #feature probabilities for each feature 
    Ftrains = data.getFs(m_value, 1 / (100 * (m_value)), Qtrain)
    predicted_classes = []
    zero_one_sum = 0
    CM = ConfusionMatrix(data.classes) #confusion matric for p macro
    result_data = pd.DataFrame(data.test_set.to_dict())
    p = data.partition(10)
    #print only one fold of the partitions
    for i in p[partition_index]:
        predicted = data.predicted_class(data.value(binned_df, i), Qtrain,
                                         Ftrains)  # predicted class value associated with the row
        actual = binned_df.at[i, 'Class']  # the class value associated with the row
        predicted_classes.append(predicted)
        zero_one_sum += data.zero_one_loss(predicted, actual) 
        CM.addOne(predicted, actual)
    result_data["Predicted_Class"] = predicted_classes #create predicted classes column
    print("Test Set with Predictions:")
    print(result_data) #print out df
    zero_one_avg = zero_one_sum / len(p[partition_index]) 
    print("Zero One Loss Average Value: {}".format(zero_one_avg))
    print("P Macro of Confusion Matrix: {}".format(CM.pmacro()))
    print("----------------------------------")


