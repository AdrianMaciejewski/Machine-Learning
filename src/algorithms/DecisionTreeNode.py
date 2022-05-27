
class DecisionTreeNode:

    def __init__(self, feature_index=None,
                 threshold=None,
                 left_node=None,
                 right_node=None,
                 variation_reduction=None,
                 value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.variation_reduction = variation_reduction

        # only for leaf node
        self.value = value
