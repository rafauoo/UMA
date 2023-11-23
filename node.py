

class Node:
    def __init__(self, split_feature=None, value=None, children=None, default_child_value=None) -> None:
        self._value = value
        self._children = children or {}
        self._split_feature = split_feature
        self._depth = 0
        self._default_child_value = default_child_value
    
    def add_child(self, child, attr_values_for_child) -> None:
        for attr_value in attr_values_for_child:
            self._children[attr_value] = child
    
    def set_default_child_value(self, child_value):
        self._default_child_value = child_value
    
    def get_default_child_value(self):
        return self._default_child_value
    
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

    def set_depth(self, depth):
        self._depth = depth
    
    def get_depth(self):
        return self._depth
    
    def get_childrenset(self):
        inverted_children = {}
        for key, value in self.get_children().items():
            if value not in inverted_children:
                inverted_children[value] = [key]
            else:
                inverted_children[value].append(key)
        return inverted_children

    def is_leaf(self):
        return len(self.get_children().keys()) == 0
            
    
