import math
from typing import List

class Node:

  """
  This class, Node, represents a single node in a decision tree. It is designed to store information about the tree
  structure and the specific split criteria at each node. It is important to note that this class should NOT be
  modified as it is part of the assignment and will be used by the autograder.

  The attributes of the Node class are:
  - split_dim: The dimension/feature along which the node splits the data (-1 by default, indicating uninitialized)
  - split_point: The value used for splitting the data at this node (-1 by default, indicating uninitialized)
  - label: The class label assigned to this node, which is the majority label of the data at this node. If there is a tie,
    the numerically smaller label is assigned (-1 by default, indicating uninitialized)
  - left: The left child node of this node (None by default). Either None or a Node object.
  - right: The right child node of this node (None by default) Either None or a Node object.
  """

  def __init__(self):
    self.split_dim = -1
    self.split_point = -1
    self.label = -1
    self.left = None
    self.right = None
    self.depth = None

  def _is_leaf(self):
      if self.right == None and self.left == None:
          return True
      else: return False
  
  def to_str(self):
      if self._is_leaf():
          print_str = f"Depth {self.depth} Leaf Node {self.label} with split_dim = {self.split_dim} and split_point = {self.split_point}" 
      else:
          print_str = f"Depth {self.depth} Node {self.label} with split_dim = {self.split_dim} and split_point = {self.split_point}" 
      return print_str
  
  def for_output(self):
      return f"split_dim: {self.split_dim}, split_point: {round(self.split_point, 4)}, label: {self.label}"
  
  def pre_order(self, node):
      
      print(node.for_output())
      if node.left != None: self.pre_order(node.left)
      if node.right != None: self.pre_order(node.right)

  def in_order(self, node):
      
      if node.left != None: self.pre_order(node.left)
      print(node.for_output())
      if node.right != None: self.pre_order(node.right)

      
      


class Solution:

  def entropy(self, p_pos: float) -> float:
        """
        Calculate entropy given the proportion of positive examples
        """
        if p_pos == 0 or p_pos == 1:
            return 0
        else:
            return -p_pos * math.log2(p_pos) 
        
  def info_gain(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> float:
      info_D = 1

      n = len(data)
      left_count = 0
      left_label_counts = {}
      right_count = 0
      right_label_counts = {}

      label_counts = {}
      for i in range(n):
          if label[i] in label_counts:
              label_counts[label[i]] += 1
          else:
              label_counts[label[i]] = 1

      # label_p = {key: value/n for key, value in label_counts.items()}
      infos = [self.entropy(label_counts[label] / n) for label in label_counts]
      print(f"infos: {infos}")
      info_D = sum([self.entropy(label_counts[label] / n) for label in label_counts])
      info_A = self.split_info(data, label, split_dim, split_point)
      return info_D - info_A

  def split_info(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> float:
      """
      Compute the information needed to classify a dataset if it's split
      with the given splitting dimension and splitting point, i.e. Info_A in the slides.

      Parameters:
      data (List[List]): A nested list representing the dataset.
      label (List): A list containing the class labels for each data point.
      split_dim (int): The dimension/attribute index to split the data on.
      split_point (float): The value at which the data should be split along the given dimension.

      Returns:
      float: The calculated Info_A value for the given split. Do NOT round this value
      """

      # print(f"Called split_info on data set of size {len(data)} with {len(data[0])} attributes and {len(set(label))} classifications")

      n = len(data)
      left_count = 0
      left_label_counts = {}
      right_count = 0
      right_label_counts = {}

      for i in range(n):
          if data[i][split_dim] <= split_point:
              left_count += 1
              if label[i] in left_label_counts:
                  left_label_counts[label[i]] += 1
              else:
                  left_label_counts[label[i]] = 1
          else:
              right_count += 1
              if label[i] in right_label_counts:
                  right_label_counts[label[i]] += 1
              else:
                  right_label_counts[label[i]] = 1

      # print(f"split_point = {split_point}")
      # print(f"split_dim = {split_dim}")
      # print(f"left_label_counts = {left_label_counts}")
      # print(f"right_label_counts = {right_label_counts}\n")

      p_left = left_count / n
      p_right = right_count / n

      if left_count == 0 or right_count == 0:
          return float(0)

      left_entropy = sum([self.entropy(left_label_counts[label] / left_count) for label in left_label_counts])
      # print(f"left entropy = {left_entropy}\n")
      right_entropy = sum([self.entropy(right_label_counts[label] / right_count) for label in right_label_counts])
      # print(f"right entropy = {right_entropy}\n")

      return p_left * left_entropy + p_right * right_entropy

  def split_node(self, node, data, labels, depth):
        """
        Recursively split nodes to create the decision tree.

        Parameters:
        node (Node): The node to be split.
        data (List[List[float]]): A nested list of floating point numbers representing the training data.
        labels (List[int]): A list of integers representing the class labels for each data point in the training set.
        depth (int): The current depth of the tree.

        This method splits the data at the current node based on the feature with the highest information gain.
        It then creates child nodes for the two resulting subsets of data, and recursively calls itself on each child node
        until a stopping condition is met.
        """
        node.depth = depth

        counts = [0] * len(range(0, max(labels)))
        for label in labels:
            counts[label-1] += 1
    
        node.label = max(range(len(counts)), key=counts.__getitem__) + 1

        # If stopping condition met, set the node's label and return
        if len(data) == 0 or depth >= self.max_depth:
            return
      
        print(f"\nCalled split_node at depth = {depth} on data set of size {len(data)} with {len(data[0])} attributes and {len(set(labels))} classifications\n")
        
        # Find the best feature to split the data
        best_feature, best_split_point = None, None
        best_info_gain = 0
        features = range(len(data[0]))
        for feature in features:
            print(f"Feature {feature}")
            
            a = sorted([data[i][feature] for i in range(len(data))])
            print(a)
            split_points = set()
            for i in range(len(a)-1):
                midpoint = (a[i] + a[i+1]) / 2.0
                split_points.add(midpoint)
            split_points = sorted(split_points)
            print(f"Split points {split_points}")
            for split_point in split_points:
                info_gain = self.info_gain(data, labels, feature, split_point)
                # print(f"f = {feature} sp = {split_point} I = {info_gain}\n")
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_split_point = split_point
        print(f"WINNER: f = {best_feature} sp = {best_split_point} I = {best_info_gain}")

        if best_info_gain == 0 or best_info_gain == 1:
            return

        # Split the data based on the best feature and split point
        left_data, left_labels, right_data, right_labels = [], [], [], []
        for i in range(len(data)):
            if data[i][best_feature] <= best_split_point:
                left_data.append(data[i])
                left_labels.append(labels[i])
            else:
                right_data.append(data[i])
                right_labels.append(labels[i])

        # Create child nodes and recursively split them
        node.split_dim = best_feature
        node.split_point = best_split_point
        node.left = Node()
        node.right = Node()
        print(f"parent node: {node.to_str()}")

        # if len(left_data) > 0: 
        self.split_node(node.left, left_data, left_labels, depth + 1)
        print(f"left node: {node.left.to_str()}")

        # if len(right_data) > 0: 
        self.split_node(node.right, right_data, right_labels, depth + 1)
        print(f"right node: {node.right.to_str()}")

  def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:

    self.root = Node()
    self.max_depth = 2

    """
    Fit the decision tree model using the provided training data and labels.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.

    This method initializes the decision tree model by creating the root node. It then builds the decision tree starting 
    from the root node
    
    It is important to note that for tree structure evaluation, the autograder for this assignment
    first calls this method. It then performs tree traversals starting from the root node in order to check whether 
    the tree structure is correct. 
    
    So it is very important to ensure that self.root is assigned correctly to the root node
    
    It is best to use a different method (such as in the example above) to build the decision tree.
    """

    # Call the recursive method to build the decision tree
    self.split_node(self.root, train_data, train_label, depth=0)

  def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
    """
    Classify the test data using a decision tree model built from the provided training data and labels.
    This method first fits the decision tree model using the provided training data and labels by calling the
    'fit()' method.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.
    test_data (List[List[float]]): A nested list of floating point numbers representing the test data.

    Returns:
    List[int]: A list of integer predictions, which are the label predictions for the test data after fitting
               the train data and labels to a decision tree.
    """
    # Fit the decision tree model using the provided training data and labels
    self.fit(train_data, train_label)

    # Initialize an empty list to store the predicted labels for the test data
    predicted_labels = []

    # Traverse the decision tree and predict the label for each test data point
    for point in test_data:
        current_node = self.root
        while not current_node._is_leaf:
            # Determine which child node to traverse based on the split condition
            if point[current_node.split_dim] <= current_node.split_point:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        # Add the predicted label to the list of predicted labels
        predicted_labels.append(current_node.label)

    return predicted_labels
