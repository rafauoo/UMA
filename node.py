

class Node:
    def __init__(self, split_feature=None, value=None, children=None) -> None:
        self._value = value
        self._children = children or {}
        self._split_feature = split_feature
    
    def add_child(self, child, attr_values_for_child) -> None:
        for attr_value in attr_values_for_child:
            self._children[attr_value] = child
    
    def set_value(self, value) -> None:
        self._value = value

    def set_split_feature(self, feature) -> None:
        self._split_feature = feature
    
    def get_children(self):
        return self._children
    
    def get_value(self):
        return self._value
    
    def get_split_feature(self):
        return self._split_feature

    
