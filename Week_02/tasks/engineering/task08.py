from value import Value
from task06 import trace
import graphviz
from string import ascii_lowercase


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    count = 0
    for n in nodes:
        letter = ascii_lowercase[count]
        count += 1
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ {letter}| data: {n.data} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def main() -> None:
    w = Value(5.0)
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = (x * y + z) * w
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)

if __name__ == '__main__':
    main()