import numpy as np

from src.algorithms.DecisionTreeNode import DecisionTreeNode


class DecisionTreeRegression:

    def __init__(self, min_samples_split, max_tree_depth):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_tree_depth = max_tree_depth

    def build_tree(self, np_array, current_depth=0):
        X, Y = np_array[:, :-1], np_array[:, -1]
        m, n = X.shape
        if current_depth <= self.max_tree_depth and m >= self.min_samples_split:
            best_split = self.get_best_split(np_array)
            if best_split["variance_reduction"] > 0:
                left_node = self.build_tree(best_split["left_dataset"], current_depth+1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth+1)

                return DecisionTreeNode(best_split["feature_index"], best_split["threshold"],
                                        left_node, right_node, best_split["variation_reduction"])

        return DecisionTreeNode(value=self.calculate_leaf_value(Y))

    def get_best_split(self, np_array):
        X, Y = np_array[:, :-1], np_array[:, -1]
        m, n = X.shape
        max_variance_reduction = -float('inf')
        best_split = {}
        for feature_index in range(0, n):
            unique_feature_values = np.unique(X[:, feature_index])
            for threshold in unique_feature_values:
                left_dataset, right_dataset = self.split(np_array, feature_index, threshold)
                variance_reduction = \
                    self.calculate_variance(Y, left_dataset[:, -1], right_dataset[:, -1])
                if variance_reduction > max_variance_reduction:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["left_dataset"] = left_dataset
                    best_split["right_dataset"] = right_dataset
                    best_split["variance_reduction"] = variance_reduction
                    max_variance_reduction = variance_reduction
        return best_split

    def split(self, np_array, feature_index, threshold):
        left_dataset = np_array[np.where(np_array[:, feature_index] <= threshold)]
        right_dataset = np_array[np.where(np_array[:, feature_index] > threshold)]
        return left_dataset, right_dataset

    def calculate_variance(self, Y, Y_left, Y_right):
        left_weight = Y_left.shape[0] / Y.shape[0]
        right_weight = Y_right.shape[0] / Y.shape[0]
        return np.var(Y) - left_weight * np.var(Y_left) - right_weight * np.var(Y_right)

    def calculate_leaf_value(self, Y):
        return np.mean(Y)

    def print_tree_pre_order(self, tree_node=None, depth=0):
        if tree_node in None:
            if self.root in None:
                print("Tree is not built")
            else:
                tree_node = self.root

        indent = '\t' * depth
        if tree_node.value is not None:
            print(f"{indent}Value: {tree_node.value}")
        else:
            print(f"{indent}{depth+1}. Feature index: {tree_node.feature_index}, threshold: {tree_node.threshold}")
            self.print_tree_pre_order(tree_node.left_dataset, depth=depth+1)
            print(f"{indent} Left subtree:")
            self.print_tree_pre_order(tree_node.right_dataset, depth=depth+1)
            print(f"{indent} Right subtree:")

    def fit(self, X, Y):
        npArray = np.concatenate(X, Y, axis=0)
        self.root = self.build_tree(npArray)

    def predict(self, X):
        return np.array([self.make_prediction(x, self.root) for x in X])

    def make_prediction(self, x, tree_node):
        if tree_node.value is not None:
            return tree_node.value
        else:
            if x[tree_node.feature_index] <= tree_node.threshold:
                return self.make_prediction(x, tree_node.left_node)
            elif x[tree_node.feature_index] > tree_node.threshold:
                return self.make_prediction(x, tree_node.right_node)
