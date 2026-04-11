class Value:
    def __init__(self, data: float, _prev = None, _op = None):
        self.data = data
        self._prev = _prev
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other_value: Value):
        # return Value(self.data + other_value.data, (self.data, self._prev), '+')
        return Value(self.data + other_value.data, (Value(self.data, self._prev), Value(other_value.data, other_value._prev)), '+')
    
    def __mul__(self, other_value: Value):
        # return Value(self.data * other_value.data, (self.data, self._prev), '*')
        return Value(self.data + other_value.data, (Value(self.data, self._prev), Value(other_value.data, other_value._prev)), '*')


"""
Initialize an empty list visited list to track visited nodes and an empty list stack to store the result.
Iterate through each vertex in the graph. If a vertex has not been visited, recursively call the (DFS) function on it.
Inside the DFS function for a vertex u:
    Mark u as visited.
    Recursively call the DFS function for all unvisited neighbors (adjacent vertices) of u.
    After all neighbors and their descendants have been fully explored, push the current vertex u onto the stack. This ensures a node is pushed only after all its dependencies are processed.
After the DFS completes for all nodes, the stack will contain the vertices in reverse topological order. Pop the elements from the stack one by one (or reverse the list) to get the final topological ordering.
"""

def trace(value: Value):
    visited = []
    stack = []

    # returns "TypeError: 'NoneType' object is not iterable"
    for prev in value._prev:
        if prev not in visited:
            visited.append(prev)


    return visited, stack

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')

if __name__ == '__main__':
    main()