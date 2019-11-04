import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    gini = 0.0
    uni = np.array(np.unique(data[:, -1], return_counts=True)[1])       # sums all unique values in the last column
    uni = uni / data[:, -1].shape
    uni = np.square(uni)
    gini = np.sum(uni)

    return 1 - gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    uni = np.array(np.unique(data[:, -1], return_counts=True)[1])       # sums all unique values in the last column
    uni = uni / data[:, -1].shape
    logUni = np.log2(uni)
    uni = np.dot(uni, logUni)
    return uni.sum(axis=0) * (-1)


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value, majority):
        self.feature = feature              # column index of criteria being tested
        self.value = value                  # value necessary to get a true result
        self.children = []                  # the node will have two children
        index = np.argmax(majority[1])      # index is the index of the majority
        self.label = majority[0][index]     # the label of this node, or the majority in it
        self.label_count = majority[1][index]
        self.parent = None                  # a pointer the the nodes parent

    # adds child to a node
    def add_child(self, node):
        self.children.append(node)


# returns an array of all the leaves in the tree
def find_leaves(node, leaves):
    if len(node.children) == 0:
        leaves.append(node)
        return
    else:
        find_leaves(node.children[0], leaves)
        find_leaves(node.children[1], leaves)


# returns the number of nodes in the tree
def count_nodes(node):
    if len(node.children) == 0:
        return 1
    return 1 + count_nodes(node.children[0]) + count_nodes(node.children[1])


# count the number of internal nodes in the tree
def count_internal_nodes(node):
    leaves = []
    find_leaves(node, leaves)
    num_of_leaves = len(leaves)
    num_of_nodes = count_nodes(node)
    return num_of_nodes - num_of_leaves


# calculates the chi square of a node
def chi_square(pf, nf, e_0, e_1):
    a = np.square(pf - e_0) / e_0
    b = np.square(nf - e_1) / e_1
    return a + b


# builds a pre pruned tree based on the given chi square value
def pre_pruning(data, impurity, chi_value):
    if chi_value != 1:
        return build_tree(data, impurity, chi_table[chi_value])
    else:
        return build_tree(data, impurity, 1)


# finds the best feature and best value to split the data
def find_feature_and_value(data, impurity):

    max_feature_column = None  # the column of the best feature
    max_val_in_feature = None  # the value, threshold, that
    min_feature_impurity = 2

    # # # # # iterate on all features to find the feature and value # # # # #

    for col in range(0, data.shape[1] - 1):
        # find all possible thresholds from the current feature
        unique_column = np.unique(data[:, [col]])
        unique_column_2 = np.roll(unique_column, len(unique_column) - 1)
        thresholds = (unique_column + unique_column_2) / 2  # an array of all the thresholds
        thresholds = thresholds[:-1]  # last value of the array is junk

        data_for_impurity = data[:, [col, -1]]  # keeps only the two relevant columns for this process

        # split the data for two nodes by the thresholds
        for val in thresholds:
            node_right = data_for_impurity[data_for_impurity[:, 0] > val]
            node_left = data_for_impurity[data_for_impurity[:, 0] <= val]

            # calculate impurity for the two new nodes
            impurity_left = (node_left.shape[0] / data.shape[0]) * impurity(node_left)
            impurity_right = (node_right.shape[0] / data.shape[0]) * impurity(node_right)
            impurity_total = impurity_right + impurity_left

            # if impurity is lower the current min update the minFeatureColumn
            if impurity_total < min_feature_impurity:
                min_feature_impurity = impurity_total
                max_feature_column = col
                max_val_in_feature = val

    return max_feature_column, max_val_in_feature


# decides whether the node should split or not
def should_split(data, max_feature_column, max_val_in_feature, chi_value):
    # create two children to the node
    node_left = data[data[:, max_feature_column] < max_val_in_feature]
    node_right = data[data[:, max_feature_column] >= max_val_in_feature]

    # calculate the chi square value of the split
    num_of_instances = data.shape[0]
    num_of_zero_instances = (data[:, -1] == 0).sum()
    num_of_one_instances = (data[:, -1] == 1).sum()

    # calculating the values for left node
    df = node_left.shape[0]
    pf = (node_left[:, -1] == 0).sum()
    nf = (node_left[:, -1] == 1).sum()

    e_0 = df * num_of_zero_instances / num_of_instances
    e_1 = df * num_of_one_instances / num_of_instances

    chi_left_node = chi_square(pf, nf, e_0, e_1)

    # calculating the values for right node
    df = node_right.shape[0]
    pf = (node_right[:, -1] == 0).sum()
    nf = (node_right[:, -1] == 1).sum()

    e_0 = df * num_of_zero_instances / num_of_instances
    e_1 = df * num_of_one_instances / num_of_instances

    chi_right_node = chi_square(pf, nf, e_0, e_1)

    # final chi square value of this node
    chi_square_value = chi_left_node + chi_right_node

    # else stop the building and make this node a leaf
    return chi_square_value <= chi_value


def build_tree(data, impurity, chi_value, ):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """

    # # # # # stopping condition for the recursive function # # # # #

    if impurity(data) == 0:
        majority = np.array(np.unique(data[:, -1], return_counts=True))    # finds the majority in this node
        # index = np.argmax(majority[1])                                     # index is the index of the majority
        # majority = majority[0][index]                                      # majority gets the majority value

        root = DecisionNode(None, None, majority)                          # when impurity is 0 we return a leaf
        return root

    max_feature_column, max_val_in_feature = find_feature_and_value(data, impurity)

    # choose the best feature to divide the data and the majority at this point
    majority = np.array(np.unique(data[:, -1], return_counts=True))           # finds the majority in this node
    # index = np.argmax(majority[1])                                            # index is the index of the majority
    # majority = majority[0][index]                                             # majority gets the majority value

    # create two children to the node
    node_left = data[data[:, max_feature_column] < max_val_in_feature]
    node_right = data[data[:, max_feature_column] >= max_val_in_feature]

    # create the root or node with the feature and value that we found above
    root = DecisionNode(max_feature_column, max_val_in_feature, majority)

    # check if we should split the node or return this node as a leaf
    if should_split(data, max_feature_column, max_val_in_feature, chi_value):
        return root

    # if it is sufficient continue building the tree

    # # # # # building the tree recursively  # # # # #

    # add the children as a node to the tree and continue recursively
    node_left = build_tree(node_left, impurity, chi_value)
    node_right = build_tree(node_right, impurity, chi_value)

    # add parent field to each leaf
    node_left.parent = root
    node_right.parent = root

    root.add_child(node_left)
    root.add_child(node_right)

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

    # if we are in a leaf return the label of this leaf or majority
    if len(node.children) == 0:
        return node.label
    elif len(node.children) == 1 or instance[node.feature] < node.value:
        x = predict(node.children[0], instance)
    else:
        x = predict(node.children[1], instance)

    return x


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
    predicted_correctly = 0
    total_predictions = 0
    # run on all the rows, instances, in the given dataset
    for row in dataset:
        if predict(node, row) == row[-1]:
            predicted_correctly = predicted_correctly + 1
        total_predictions = total_predictions + 1
    accuracy = predicted_correctly / total_predictions

    return accuracy * 100


# post prune the tree all the way until you are left with only the root
def post_pruning(node, dataset):
    number_of_nodes = [count_internal_nodes(node)]
    tree_accuracy = [calc_accuracy(node, dataset)]

    # run on a loop until we are left only with the root
    while count_nodes(node) > 2:
        temp = find_best_parent(node, dataset)
        number_of_nodes.append(temp[1])
        tree_accuracy.append(temp[0])

    return number_of_nodes, tree_accuracy


# finds the best parent in the tree, which means that if we delete its children we will
# increase or reduce the accuracy the most or least
# returns None if we ran on the root else the best parent
def find_best_parent(node, dataset):

    if len(node.children) == 0:
        return None

    leaves = []
    find_leaves(node, leaves)                     # a list with all the leaves in the tree
    best_value = None

    # run on all the leaves in the tree
    for leaf in leaves:
        parent = leaf.parent
        temp_children = parent.children
        parent.children = []                                # remove children
        accuracy = calc_accuracy(node, dataset)             # calc accuracy
        parent.children = temp_children                     # return the children
        if best_value is None or best_value < accuracy:
            best_value = accuracy
            best_parent = parent

    # remove the leafs that we found will increase or reduce the accuracy the most
    best_parent.children = []

    return best_value, count_internal_nodes(node)


# prints the tree
def print_tree(node, count):
    """
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    """

    spaces = '  ' * count
    if len(node.children) == 0:
        string = "leaf: [{%.1f, %d}]" % (node.label, node.label_count)
        print("%s%s" % (spaces, string))
        return
    else:
        string = "[X%d <= %.3f]" % (node.feature, node.value)
        print("%s%s" % (spaces, string))
        count += 1
        if node.children[0] is not None:
            print_tree(node.children[0], count)
        if node.children[1] is not None:
            print_tree(node.children[1], count)
