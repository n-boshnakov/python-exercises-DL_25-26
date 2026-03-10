class Value:
    def __init__(self, data, _prev=(), _op='',  grad=0.0):
        self.data = data
        self._prev = set(_prev)
        self._op = _op
        self.grad = grad

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')
    
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
def manual_der(self, node, manual_score, epsilon=0.001):
    self.assertEqual(manual_score, node.grad)