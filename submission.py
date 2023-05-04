import copy
import math
from typing import Dict, List, Tuple

num_C = 7  # Represents the total number of classes

class Solution:

  def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
    """Calculate the prior probabilities of each class
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
    Returns:
      A list of length n_classes where n_classes is the number of classes in the dataset
    """

    n = len(X_train)
    n_classes = 7

    class_counts = [0] * num_C
    for i in range(n):
        class_counts[Y_train[i]-1] += 1

    priors = [(count+0.1) / (n + 0.1*n_classes) for count in class_counts]
    return priors

  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
    """Calculate the classification labels for each test datapoint
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
      X_test: Row i represents the i-th testing datapoint
    Returns:
      A list of length M where M is the number of datapoints in the test set
    """

    feature_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']
    # Calculate the prior probabilities of each class
    priors = self.prior(X_train, Y_train)
    
    n_classes = len(priors)
    n_features = len(X_train[0])
    n_train = len(X_train)
    likelihoods = [{} for _ in range(n_features)] * n_classes
    for c in range(n_classes):
      print(f"CLASS {c}")
      # Collect all training datapoints belonging to class c
      X_train_c = [X_train[i] for i in range(n_train) if Y_train[i] == c+1]
      # Count the number of times each feature value occurs for each class

      for i in range(n_features):

        # possible_values = [0,1]
        possible_values = [0,1]
        if feature_names[i] == 'legs': possible_values = [0,2,4,5,6,8]

        feature_values = [x[i] for x in X_train_c]
        value_counts = {}

        for value in possible_values:
            value_counts[value] = 0

        for value in feature_values:
          value = int(value)
          if value in value_counts:
            value_counts[value] += 1
          else:
            value_counts[value] = 1
            
        # Calculate the likelihood for each feature value
        for value in value_counts:
          value = int(value)
          likelihoods[c][value,i] = (value_counts[value]+0.1) /  (len(X_train_c)+0.1*(len(set([x[i] for x in X_train])))) #n_features) #
          print(f"feature {i} value {value} likelihood = {likelihoods[c][value,i]}")
    # Classify the test data
    n_test = len(X_test)
    y_pred = []
    for j in range(n_test):
      # Calculate the posterior probability of each class
      posteriors = [priors[c] for c in range(n_classes)]
      for i in range(n_features):
        value = int(X_test[j][i])
        for c in range(n_classes):
          if (value,i) in likelihoods[c]:
            posteriors[c] *= likelihoods[c][value,i]
          else:
            # Laplacian
            posteriors[c] *= 0.1 / (len(X_train_c)+0.1*(len(set([x[i] for x in X_train])))) #(len(X_train_c)+0.1*n_features) #
      # Classify the test datapoint by choosing the class with the highest posterior probability
      print(f"posteriors {posteriors}")
      y_pred.append(posteriors.index(max(posteriors))+1)
    return y_pred
