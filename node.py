import numpy as np
from queue import Queue


class Node():
    def __init__(self, name, value=None, parent=None, children=[]):
        self.name = name
        self.value = value
        self.parent = parent
        self.children = set(children)
        self.reachable_leaves = None
        self.height = None
        self.depth = None
        
    def connect_child(self, child):
        child.parent = self
        self.children.add(child)
        
    def __str__(self):
        parent_name = None
        if self.parent is not None:
            parent_name = self.parent.name
        return f"{self.name} ({self.value}) parent={parent_name} num children={len(self.children)}"
    
    def __repr__(self):
        return self.name
    
    def set_reachable_leaves(self):
        if len(self.children) == 0:
            self.reachable_leaves = set([self])

        if self.reachable_leaves is None:
            child_reachable_leaves = [child.set_reachable_leaves() for child in self.children]
            self.reachable_leaves = set().union(*child_reachable_leaves)
            
        return self.reachable_leaves
        
    def set_depth(self):
        if self.parent is None: 
            self.depth = 0
        if self.depth is None:
            self.depth = self.parent.set_depth() + 1
        return self.depth
    
    def set_height(self):
        if len(self.children) == 0:
            self.height = 0
        if self.height is None:
            child_heights = [child.set_height() for child in self.children]
            self.height = max(child_heights) + 1
        return self.height
        
    def is_connected(self):
        return len(self.children) > 0 or self.parent is not None
    
    
class GraphNode():
    def __init__(self, name, values=[], parents=[], children=[]):
        self.name = name
        self.values = []
        self.parents = []
        self.children = set(children)
        self.reachable_leaves = None
        self.height = None
        self.depth = None
        
    def connect_child(self, child):
        if self not in child.parents:
            child.parents.append(self)
        self.children.add(child)
        
    def __str__(self):
        return f"{self.name} values={self.values} parents={[parent.name for parent in self.parents]} num children={len(self.children)} depth={self.depth} height={self.height}"
    
    def __repr__(self):
        return self.name
    
    def set_reachable_leaves(self):
        if len(self.children) == 0:
            self.reachable_leaves = set([self])

        if self.reachable_leaves is None:
            child_reachable_leaves = [child.set_reachable_leaves() for child in self.children]
            self.reachable_leaves = set().union(*child_reachable_leaves)
            
        return self.reachable_leaves
        
    def set_depth(self):
        if len(self.parents) == 0:
            self.depth = 0
        if self.depth is None:
            parent_depths = [parent.set_depth() for parent in self.parents]
            self.depth = max(parent_depths) + 1
        return self.depth
    
    def set_height(self):
        if len(self.children) == 0:
            self.height = 0
        if self.height is None:
            child_heights = [child.set_height() for child in self.children]
            self.height = max(child_heights) + 1
        return self.height
        
    def is_connected(self):
        return len(self.children) > 0 or len(self.parents) > 0