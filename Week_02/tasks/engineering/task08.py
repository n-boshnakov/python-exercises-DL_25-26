from value import Value
from task06 import trace
import graphviz
from string import ascii_lowercase


import graphviz
from string import ascii_lowercase

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        filename='01_result',
        format='svg',
        graph_attr={'rankdir': 'LR'}
    )

    nodes, edges = trace(root)

    # assign letters to all but last; last is 'L'
    labels = {}
    for i, n in enumerate(nodes):
        if i == len(nodes) - 1:
            labels[n] = 'L'
        else:
            labels[n] = ascii_lowercase[i]  # a, b, c, ...

    # draw value + op nodes
    for n in nodes:
        uid = str(id(n))
        letter = labels[n]
        dot.node(uid, label=f'{{ {letter} | data: {n.data} }}', shape='record')

        if n._op:
            op_id = uid + n._op
            dot.node(op_id, label=n._op)
            dot.edge(op_id, uid)

    # draw edges parent → op → child
    for parent, child in edges:
        dot.edge(str(id(parent)), str(id(child)) + child._op)

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