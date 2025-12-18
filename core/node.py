from collections import defaultdict

import core.tensor as t

'''Super class used for building the graph'''
class Node:
    countPerClassName = defaultdict[str, int](int)
    
    def __init__(self, className: str, given_name: str = None):
        self.name = given_name if given_name else self.generate_unique_name(className)
        self.nextNodes = list[Node]()

    def add_next_node(self, node: Node):
        self.nextNodes.append(node)

    def apply(self, *output_grads: t.Tensor):
        raise NotImplementedError

    def __repr__(self):
        return self.name

    def generate_unique_name(self, className: str):
        Node.countPerClassName[className] += 1
        return f"{className}{Node.countPerClassName[className]}"

