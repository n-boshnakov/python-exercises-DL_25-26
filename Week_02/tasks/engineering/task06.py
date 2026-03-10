from value import Value

def trace(value):
    # original idea
    
    # nodes = []
    # edges = []
    # nodes.append(value)
    # while (value._prev != None):
    #     nodes.append(value)
    #     edges.append((value._prev, value))
    #     value = value._prev

    # with help from Gemini
    nodes = set()
    edges = []

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.append((child, v))
                build(child)

    build(value)

    return nodes, edges

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