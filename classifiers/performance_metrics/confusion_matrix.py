from sklearn import datasets
import numpy as np
import pandas
from classifiers.knn import knn

########################################
# Load and organize data into different arrays
########################################
iris_dataset = datasets.load_wine()  # load
data = iris_dataset['data']  # data values
target = iris_dataset['target']  # its targets
target_names = iris_dataset['target_names']
# split data index
split_80 = int(len(data)*0.8)  # *0.8 = get 80%
# 80% for train
train_data = data[0:split_80]
train_target = target[0:split_80]
# 20% for test
test_data = data[split_80:]
test_target = target[split_80:]


########################################
# Create a confusion Matrix
########################################
classes_count = len(target_names)
# rows: actual class, cols: predicted class
confusion_matrix = np.zeros((classes_count, classes_count), dtype=int)
k = 3  # k nearest neighbors#

# for each example in test data
for i in range(0, len(test_data)):
    predicted_class = knn.classify(test_data[i], train_data, train_target, k)
    # confusion_matrix[actual][predicted]
    actual = test_target[i]
    predicted = predicted_class
    confusion_matrix[actual][predicted] += 1  # sum 1

# Print Confusion Matrix
print("--- Confusion Matrix:")
print(pandas.DataFrame(confusion_matrix, target_names, target_names))