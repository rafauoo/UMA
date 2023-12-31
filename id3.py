'''
Created by
Rafal Budnik
Zuzanna Damszel
'''

from node import Node
from enum import Enum
import math

def most_common(data):
    '''
    Returns most common value in given dataset.
    '''
    return data.value_counts().idxmax()[0]

def entropy(class_count, num_samples):
    '''
    Calculates entropy for given class.
    '''
    p = class_count / num_samples
    return -p * math.log(p, 2)

def calc_entropy_for_set(data):
    '''
    Calculates entropy for given dataset
    '''
    sum = 0
    for value_count in data.value_counts():
        sum = sum + entropy(value_count, len(data))
    return sum

# BINARY
def get_best_value_for_attribute(elements, attribute, target_attribute):
    '''
    When attribute was found it is needed to find best value.
    '''
    all_classes_entropy = calc_entropy_for_set(target_attribute)
    max_inf_gain = 0
    best_value = ''
    for value in set(elements[attribute]):
        childa_classes = target_attribute[elements[attribute] == value]
        childb_classes = target_attribute[elements[attribute] != value]
        a_len = len(childa_classes)
        b_len = len(childb_classes)
        a_entropy = calc_entropy_for_set(childa_classes)
        b_entropy = calc_entropy_for_set(childb_classes)
        entropy_after_split = (a_entropy * a_len + b_entropy * b_len) / (a_len + b_len)
        inf_gain = all_classes_entropy - entropy_after_split
        if max_inf_gain < inf_gain:
            max_inf_gain = inf_gain
            best_value = value
    return (max_inf_gain, best_value)


# BINARY
def choose_best_attribute_binary(elements, target_attribute, attributes: list):
    '''
    Function used for choosing best attribute when division is binary.
    '''
    best_attribute = ''
    best_value = ''
    max_inf_gain = -1
    for attribute in attributes:
        inf_gain, value = get_best_value_for_attribute(elements, attribute, target_attribute)
        if inf_gain > max_inf_gain:
            best_attribute = attribute
            best_value = value
            max_inf_gain = inf_gain
    return (best_attribute, best_value)

# IDENTITY
def choose_best_attribute_identity(elements, target_attribute, attributes: list):
    '''
    Function used for choosing best attribute when division is identity.
    We do not have to choose best value for attribute since all values will be represented.
    '''
    best_attribute = ''
    max_inf_gain = -1
    all_classes_entropy = calc_entropy_for_set(target_attribute)
    for attribute in attributes:
        entropy = 0
        for value in set(elements[attribute]):
            classes_for_value = target_attribute[elements[attribute] == value]
            set_len = len(classes_for_value)
            set_entropy = calc_entropy_for_set(classes_for_value)
            entropy += set_entropy * set_len
        entropy /= len(target_attribute)
        inf_gain = all_classes_entropy - entropy
        if inf_gain > max_inf_gain:
            best_attribute = attribute
            max_inf_gain = inf_gain
    return best_attribute

class ID3:
    '''
    ID3 tree representation class
    '''
    class TreeType(Enum):
        '''
        Enum used for defining type of the tree (BINARY/IDENTITY).
        '''
        BINARY = 0
        IDENTITY = 1

    def __init__(self, type: TreeType, root=None) -> None:
        self.type = type
        self.root = root

    def clear(self):
        self.root = None

    def build(self, elements, target_attribute):
        attributes = elements.columns.tolist()
        self.root = self._build_tree(elements, target_attribute, attributes, 0)
    
    def _build_tree(self, elements, target_attribute, attributes, depth=0) -> Node:
        # Tworzymy korzeń drzewa
        root = Node()
        root.set_depth(depth)

        # Jeśli wszystkie elementy zbioru mają tą samą klasę to zwracamy drzewo jednoelementowe z tą klasą
        if(len(set(target_attribute.iloc[:, 0])) == 1):
            root.set_value(next(iter(set(target_attribute.iloc[:, 0]))))
            return root
        
        # Jeśli jest brak atrybutów na których można wykonać predykcję 
        # to zwracamy drzewo jednoelementowe z klasą większościową
        if(len(attributes) == 0):
            root.set_value(most_common(target_attribute))
            return root
        
        # W przeciwnym wypadku:
        #   Wybieramy atrybut, według którego podział skutkuje najmniejszą entropią. 
        #   Wybór takiego atrybutu następuje poprzez policzenie entropii zbioru przed podziałem 
        #   oraz wyliczenie entropii po potencjalnych podziałach według różnych atrybutów.
        #   Najmniejsza entropia oznacza większe uporządkowanie zbioru - jest pożądana.
        possible_value_nodes = []
        best_attribute = ''
        if(self.type == self.TreeType.BINARY):
            best_attribute, best_value = choose_best_attribute_binary(elements, target_attribute, attributes)
            possible_value_nodes.append(list([best_value]))
            other_values = []
            root._default_child_value = best_value
            for value in set(elements[best_attribute]):
                if (value != best_value):
                    other_values.append(value)
                    root._default_child_value = value
            possible_value_nodes.append(other_values)

        if(self.type == self.TreeType.IDENTITY):
            best_attribute = choose_best_attribute_identity(elements, target_attribute, attributes)
            for value in set(elements[best_attribute]):
                possible_value_nodes.append(list([value]))
                root._default_child_value = value
        
        #   Przypisujemy wybrany_atrybut jako atrybut dzielący (dzielnik) dla naszego korzenia.
        root.set_split_feature(best_attribute)

        #   Dla każdej możliwej wartości wybranego atrybutu (v[i]):
        for values_for_node in possible_value_nodes:
            # Dodajemy nową gałąź do korzenia naszego drzewa, 
            # do której klasyfikowane będą elementy dla których wybrany_atrybut = v[i].
            filtered_elements = elements[elements[root.get_split_feature()].isin(values_for_node)]
            filtered_targets = target_attribute[elements[root.get_split_feature()].isin(values_for_node)]

            # Jeśli nie ma elementów spełniających powyższą równość (wybrany atrybut = v[i]), 
            # to do gałęzi dodajemy liść z klasą dominującą wśród elementów
            if (len(filtered_elements) == 0):
                child_leaf = Node()
                child_leaf.set_value(most_common(target_attribute))
                root.add_child(child_leaf, values_for_node)
            
            # W przeciwnym wypadku do gałęzi dołączamy poddrzewo, 
            # które zostanie stworzone przy wywołaniu 
            # ID3(elementy spełniające wybrany_atrybut = v[i], atrybut_klasyfikacyjny, atrybuty - wybrany_atrybut)
            else:
                new_attributes = [attr for attr in attributes if attr != best_attribute]
                child_tree = self._build_tree(filtered_elements, filtered_targets, new_attributes, depth+1)
                root.add_child(child_tree, values_for_node)
            
        return root
    
    def print(self):
        self._print_tree(self.root)
    
    def _print_tree(self, node: Node, indent=""):
        if node.get_split_feature() is not None:
            print(indent + f"Split feature: {node.get_split_feature()}")
            childrenset = node.get_childrenset()
            for child, values in childrenset.items():
                str = indent + f"  Values: "
                for value in values:
                    str += f"{value}"
                    if value == node.get_default_child_value():
                        str += " [DEFAULT]"
                    str += ", "
                print(str)
                self._print_tree(child, indent + "    ")
        else:
            print(indent + f"Leaf node, class: {node.get_value()}")
    
    def predict(self, data):
        return self._predict_tree(self.root, data)
    
    def _predict_tree(self, node: Node, data):
        if node.get_value() is None:
            if data[node.get_split_feature()] in node.get_children():
                return self._predict_tree(node.get_children()[data[node.get_split_feature()]], data)
            else:
                return self._predict_tree(node.get_children()[node.get_default_child_value()], data)
        else:
            return node.get_value()
    
    def predict_set(self, data):
        predictions = []
        for x in data.iloc:
            result = self.predict(x)
            predictions.append(result)
        return predictions

    def get_tree_node_count(self):
        return self._get_node_count(self.root)

    def _get_node_count(self, node: Node):
        sum = 0
        for child in node.get_childrenset().keys():
            sum += self._get_node_count(child)
        return 1 if node.is_leaf() else sum + 1

    def get_tree_leaf_count(self):
        return self._get_leaf_count(self.root)

    def _get_leaf_count(self, node: Node):
        sum = 0
        for child in node.get_childrenset().keys():
            sum += self._get_leaf_count(child)
        return 1 if node.is_leaf() else sum
        
