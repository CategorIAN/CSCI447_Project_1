from EthanClass import EthanClass as EC

# def accuracy(actual, predicted):
#     correct = 0
#     for i in range(len(actual)):
#         if actual[i] == predicted[i]:
#             correct += 1
#     return correct / float(len(actual)) * 100.0   
    
# def separate_by_class(df):
#     return df.groupby(class_name)

# def mean(numbers):
#     if isinstance(numbers[0], str):
#         return 1
#     else:  
#         return sum(numbers)/float(len(numbers))

# def stdev(numbers):
#     if isinstance(numbers[0], str):
#         return 1
#     else:
#         avg = mean(numbers)
#         variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
#         return sqrt(variance)

# def summarize_dataset(df):
#     summaries = [(mean(df[column].values), stdev(df[column].values), len(df[column].values)) for column in df.columns]
#     del(summaries[-1]) #delete the class values from the summaries
#     return summaries

# def summarize_by_class(df):
#     separated = separate_by_class(df)
#     summaries = dict()
#     for classes in separated.groups:
#         summaries[classes] = summarize_dataset(separated.get_group(classes))
        
#     #     print(type(rows))
#     #     summaries[class_value] = summarize_dataset(rows)
#     return summaries

# def calculate_probability(x, mean, stdev):
    
#     if stdev == 0.0 or mean == 0.0:
#         # print(stdev)
#         return 1
#     else:
#         exponent = exp(-((x-mean)**2 / (2 * stdev**2)))  
#         if exponent == 0.0:
#             # print(1 / (sqrt(2*pi) * stdev))
#             return (1 / (sqrt(2*pi) * stdev))
#         else: 
#             # psrint((1 / (sqrt(2*pi) * stdev)) * exponent)   
#             return (1 / (sqrt(2*pi) * stdev)) * exponent
         

# def calcuate_class_probabilities(summaries, row):
#     total_rows = sum([summaries[label][0][2] for label in summaries])
#     probabilities = dict()
#     for class_value, class_summaries in summaries.items():
#         probabilities[class_value] = summaries[class_value][0][2]/float(total_rows) * 10**200
#         for i in range(len(class_summaries)-1):
#             mean, stdev, _ = class_summaries[i]
#             probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
#     return probabilities

# def predict(summaries, row):
#     probabilities = calcuate_class_probabilities(summaries, row)
#     print(probabilities)
#     best_label, best_prob = None, -1
#     for class_value, probability in probabilities.items():
#         if best_label is None or probability > best_prob:
#             best_prob = probability
#             best_label = class_value
#     return best_label
   

class Glass (EC):
    def __init__(self):
        features = [ #Class at end
            "Id number: 1 to 214",
            "RI: refractive index",
            "Na: Sodium",
            "Mg: Magnesium",
            "Al: Aluminum",
            "Si: Silicon",
            "K: Potassium",
            "Ca: Calcium",
            "Ba: Barium",
            "Fe: Iron"
            ]
        glass = EC(file = 'glass.csv', features = features, name = 'Glass', classLoc = 'end')

        glass.test()



