import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    label = data[:, data.shape[1] - 1]
    values, count = np.unique(label, return_counts=True)

    gini = 1 - np.sum(
        ((count[i] / np.sum(count)) ** 2 for i in range(len(values)))
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    label = data[:, data.shape[1] - 1]
    values, count = np.unique(label, return_counts=True)
    entropy = np.sum(
        [-1 * (count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(len(values))]
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


def impurity_reduce(child_one, child_two, parent_impurity, impurity):
    len_one = child_one.shape[0]
    len_two = child_two.shape[0]
    child_impurity = (impurity(child_one) * len_one / (len_one + len_two)) + \
                     (impurity(child_two) * len_two / (len_one + len_two))
    return parent_impurity - child_impurity


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.children = []
        self.parent = None
        self.data = None
        self.majority = -1.0

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

    # def set_parent(self, parent):
    #     self.parent = parent

    # To be used with leaves
    # def set_data(self, data):
    #     self.data = data

    def choose_side(self, single_row):
        """"
             takes a row and classify it according to the current node's feature and value
             @:param singleRow - a row from the data
             @:return true if the row value for the feature is larger/equal to the node value
        """
        row_value = single_row[self.feature]
        return row_value >= self.value

    def is_leaf(self):
        return self.feature is None

    def set_leaf(self, data):
        self.feature = None
        self.value = 0
        self.data = data

    def split_data(self, data):
        """"
            takes the data and split it according to the node's feature and value
            @:param data - the data set
            @:return two lists of rows, originated from the original data
        """
        child_one = []
        child_two = []
        for row in data:
            if self.choose_side(row):
                child_one.append(row)
            else:
                child_two.append(row)
        return np.array(child_one), np.array(child_two)

    def chi_square(self, data):

        sum = 0
        prob_y1 = np.sum(data[:, -1]) / data.shape[0]
        prob_y0 = 1 - prob_y1

        datasets = np.array(self.split_data(data))

        for dataset in datasets:
            d_f = dataset.shape[0]
            n_f = np.sum(dataset[:, -1])
            p_f = d_f - n_f
            e0 = d_f * prob_y0
            e1 = d_f * prob_y1
            current_val_to_sum = (np.square(p_f - e0) / e0) + (np.square(n_f - e1) / e1)
            sum += current_val_to_sum

        return sum


def best_split(data, impurity):
    """"
        finds the best feature and threshold to split the data, according to the impurity function
        @:param data - the data set
        @:param impurity - either calc_gini or calc_entropy
        @:return best feature and threshold, and the impurity gain it provides
    """
    best_gain = 0
    best_feature = None
    best_threshold = 0
    current_impurity = impurity(data)
    num_features = data.shape[1] - 2

    # Iterating over all features and thresholds to find the best split
    for feature in range(num_features):
        thresholds = []
        # capturing unique values from feature and calculating the thresholds as average of each consecutive pair
        unique_values = np.unique(data[:, feature])
        for i in range(len(unique_values) - 1):
            thresholds.append((unique_values[i] + unique_values[i + 1]) / 2.0)
        for value in thresholds:
            # Reaching this part means we now posses a (feature,threshold) pair
            # For each pair - creating a node and calculating the impurity gain
            current_node = DecisionNode(feature, value)
            child_one, child_two = current_node.split_data(data)
            if len(child_two) == 0 or len(child_one) == 0:
                continue
            current_gain = impurity_reduce(child_one, child_two, current_impurity, impurity)
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold, best_gain


def build_tree(data, impurity, p_value=1):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    # best_feature, best_value, gain = best_split(data, impurity)
    #
    # # if gain is 0 than we are in a leaf, so setting feature to None and value to 0
    # if gain == 0:
    if np.unique(data[:, -1]).size == 1:
        root = DecisionNode(None, 0)
        root.data = data
        root.majority = 1.0 if np.sum(data[:, -1]) > (data.shape[0] / 2) else 0
        return root

    best_feature, best_value, gain = best_split(data, impurity)
    root = DecisionNode(best_feature, best_value)

    if p_value == 1 or root.chi_square(data) >= chi_table[p_value]:
        left, right = root.split_data(data)
        root.add_child(build_tree(left, impurity))
        root.add_child(build_tree(right, impurity))
    else:
        root = DecisionNode(None, 0)
        root.data = data
        root.majority = root.majority = 1.0 if np.sum(data[:, -1]) > (data.shape[0] / 2) else 0
        return root

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    while not node.is_leaf():
        if node.choose_side(instance):
            node = node.children[0]
        else:
            node = node.children[1]
    pred = node.majority
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    counter = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for row in dataset:
        pred = predict(node, row)
        label = row[-1]
        if pred == label:
            counter += 1
    accuracy = counter / dataset.shape[0] * 100
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy


def print_tree(node, count=0):
    """
    prints the tree according to the example in the notebook

	Input:
	- node: a node in the decision tree

	This function has no return value
	"""

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################    
    str = ""
    space = '  ' * count
    if node.is_leaf():
        label = node.data[:, -1]
        length = len(node.data[:, -1])
        ones = np.count_nonzero(node.data[:, -1])
        count = ones if node.majority > 0 else length - ones
        str = "leaf: [{%.1f, %d}]" % (node.majority, count)
        print("%s%s" % (space, str))
    else:
        str = "[X%d <= %.5f]" % (node.feature, node.value)
        print("%s%s" % (space, str))
        count += 1
        if node.children[1] is not None:
            print_tree(node.children[0], count)
        if node.children[1] is not None:
            print_tree(node.children[1], count)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
