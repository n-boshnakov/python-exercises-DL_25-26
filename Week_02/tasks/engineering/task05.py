class Value:
    def __init__(self, data: float, _prev = None, _op = ''):
        self.data = data
        self._prev = _prev
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other_value: Value):
        return Value(self.data + other_value.data, (Value(self.data, self._prev), Value(other_value.data, other_value._prev)), '+')
        # return Value(self.data + other_value.data, (self.data, self._prev), '+')
    
    def __mul__(self, other_value: Value):
        # return Value(self.data * other_value.data, (self.data, self._prev), '*')
        return Value(self.data + other_value.data, (Value(self.data, self._prev), Value(other_value.data, other_value._prev)), '*')

    
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)

if __name__ == '__main__':
    main()