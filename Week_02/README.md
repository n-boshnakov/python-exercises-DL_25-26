# Goal for week 02

1. See backpropagation in action by implementing a `Value` class.
2. Visualize a computational graph.
3. Create our own class `Neuron`.
4. Use the class `Neuron` to create a multilayer perceptron and train it.

## Table of contents

- [Goal for week 02](#goal-for-week-02)
  - [Table of contents](#table-of-contents)
  - [Data science](#data-science)
  - [Engineering](#engineering)
    - [Building up the class `Value`](#building-up-the-class-value)
      - [Task 1](#task-1)
      - [Task 2](#task-2)
      - [Task 3](#task-3)
      - [Task 4](#task-4)
      - [Task 5](#task-5)
      - [Task 6](#task-6)
      - [Task 7](#task-7)
      - [Task 8](#task-8)
      - [Task 9](#task-9)
      - [Task 10](#task-10)
      - [Task 11](#task-11)
      - [Task 12](#task-12)
      - [Task 13](#task-13)
      - [Task 14](#task-14)
      - [Task 15](#task-15)
      - [Task 16](#task-16)
      - [Task 17](#task-17)
      - [Task 18](#task-18)
      - [Task 19](#task-19)
    - [Using `Value`s to create `Neuron`s](#using-values-to-create-neurons)
      - [Task 1](#task-1-1)
      - [Task 2](#task-2-1)
      - [Task 3](#task-3-1)
      - [Task 4](#task-4-1)
      - [Task 5](#task-5-1)
      - [Task 6](#task-6-1)
      - [Task 7](#task-7-1)
    - [Implementing activation functions](#implementing-activation-functions)
      - [Task 1](#task-1-2)
      - [Task 2](#task-2-2)
      - [Task 3](#task-3-2)
      - [Task 4](#task-4-2)
      - [Task 5](#task-5-2)
      - [Task 6](#task-6-2)

## Data science

This week has no data science tasks.

## Engineering

Building systems and implementing models.

### Building up the class `Value`

#### Task 1

**Description:**

Create a class `Value` that stores a single floating point number and implements the output operator.

**Test cases:**

```python
def main() -> None:
    value1 = Value(5)
    print(value1)

    value2 = Value(6)
    print(value2)
```

```console
Value(data=5)
Value(data=6)
```

#### Task 2

**Description:**

Extend the `Value` class by implementing functionality to add two values.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    result = x + y
    print(result)
```

```console
Value(data=-1.0)
```

#### Task 3

**Description:**

Extend the `Value` class by implementing functionality to multiply two values.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result)
```

```console
Value(data=4.0)
```

#### Task 4

**Description:**

Extend the `Value` class with another state variable that holds the values that produced the current value.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)
```

```console
{Value(data=-6.0), Value(data=10.0)}
```

#### Task 5

**Description:**

Extend the `Value` class with another state variable that holds the operation that produced the current value.

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)
```

```console
+
```

#### Task 6

**Description:**

Implement a function that takes a `Value` object and returns the nodes and edges that lead to the passed object.

**Test case:**

```python
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
```

```console
x
nodes={Value(data=2.0)}
edges=set()
y
nodes={Value(data=-3.0)}
edges=set()
z
nodes={Value(data=10.0)}
edges=set()
result
nodes={Value(data=10.0), Value(data=-3.0), Value(data=4.0), Value(data=-6.0), Value(data=2.0)}
edges={(Value(data=-6.0), Value(data=4.0)), (Value(data=10.0), Value(data=4.0)), (Value(data=-3.0), Value(data=-6.0)), (Value(data=2.0), Value(data=-6.0))}
```

#### Task 7

**Description:**

Let's visualize the directed acyclic graph (`DAG`) leading up to a certain value. We'll be using the Python package [graphviz](https://pypi.org/project/graphviz/). Note, that you should have `graphviz` installed on your operating system. Graphviz is available for installation [in their official website](https://graphviz.org/download/). After installing, run the command `pip install graphviz` (or install all packages listed in the file `requirements.txt`).

Add the following code and ensure the test case runs successfully. Note that if by now you've run all scripts from the integrated terminal in vscode, you might now have run this script from a terminal/command prompt that is **not** in VSCode to see the generated results / not get errors.

```python
def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{ data: {n.data} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

**Test case:**

```python
def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(result).render(directory='./graphviz_output', view=True)
```

![w02_01_result](../assets/w02_01_result.svg?raw=true "w02_01_result.png")

#### Task 8

**Description:**

Include a label of the node in the visualization shown by `graphviz` and add another node - `f`.

**Test case:**

```console
python task08.py
```

![w02_02_result](../assets/w02_02_result.svg?raw=true "w02_02_result.png")

#### Task 9

**Description:**

Now we're going to start back-propagating derivatives to see how slight nudges (changes) to each of the variables change the value of the final variable - `L`, which we'll treat as the value of our loss function.

Extend the `Value` class by adding a new state variable that holds the gradient (derivative value) of that value with respect to `L`. By default it should be initialized with `0`, meaning "no effect" (initially we're assuming that the values do not affect the output).

Visualize the gradient of each node via `graphviz`.

**Test case:**

```console
python task09.py
```

![w02_03_result](../assets/w02_03_result.svg?raw=true "w02_03_result.png")

#### Task 10

**Description:**

Manually do backpropagation on the generated computational graph by setting `node.grad = your_calculated_value` for all the nodes. Above each assignment put the calculation you did in a comment.

To check your calculations, define a function `manual_der` outside of the class `Value`. It should calculate the derivative of $L$ with respect to a given node using `epsilon = 0.001` and the result should equal `node.grad` calculated with backpropagation. Don't calculate the derivative in `manual_der` - instead, use the approximation technique from last week.

**Acceptance criteria:**

1. The process through which the values for the gradients are calculated is shown in comments.
2. A function `manual_der` is defined that verifies the manual calculations.

**Test case:**

![w02_04_result](../assets/w02_04_result.svg?raw=true "w02_04_result.png")

#### Task 11

**Description:**

Perform a gradient step. Print the initial and new value of the loss function.

**Test case:**

```console
python task11.py
```

```console
Old L = 20.0
New L = 16.293599999999998
```

#### Task 12

**Description:**

Implement a perceptron with two inputs (for now without an activation function). Name the output node (on which you're calling `draw_dot`) `logit` - this is the term for a value that has not been passed through an activation function.

Here's what the perceptron model looks like:

![w02_neuron_model](../assets/w02_neuron_model.jpeg?raw=true "w02_neuron_model.jpeg")

You can see from the test case what configuration you have to use for `x1`, `x2`, `w1`, `w2` and `b`.

**Test case:**

```console
python task12.py
```

![w02_05_result](../assets/w02_05_result.svg?raw=true "w02_05_result.svg")

#### Task 13

**Description:**

Add the hyperbolic tangent as an activation function.

Let's also change the value of the bias to be `6.8813735870195432` (so that we get derivative values with a small amount of digits after the comma) and display the computational graph.

**Test case:**

```console
python task13.py
```

![w02_06_result](../assets/w02_06_result.svg?raw=true "w02_06_result.svg")

#### Task 14

**Description:**

Manually backpropagate the gradients.

**Test case:**

```console
python task14.py
```

![w02_07_result](../assets/w02_07_result.svg?raw=true "w02_07_result.svg")

#### Task 15

**Description:**

Codify the differentiation process so that it can be executed automatically using a `backward` method that is called on the final (right-most node).

To do this, we'll need to:

- add another property to the `Value` class called `_backward` for automatic differentiation of the addition operation;
- define a function - `top_sort`, that accepts a list of `Value` objects and sort them topologically.
- integrate `top_sort` in a method called `backward` (notice that the `_backward` properties will stay).

**Acceptance criteria:**

1. The gradient for addition is calculated automatically.

**Test case:**

```console
python task15.py
```

![w02_08_result](../assets/w02_08_result.svg?raw=true "w02_08_result.svg")

#### Task 16

**Description:**

Implement `_backward` for the multiplication and hyperbolic tangent operations.

**Acceptance criteria:**

1. The gradient for multiplication is calculated automatically.
2. The gradient for hyperbolic tangent application is calculated automatically.
3. The `backward` method is called from `main()`.

**Test case:**

```console
python task16.py
```

![w02_09_result](../assets/w02_09_result.svg?raw=true "w02_09_result.svg")

#### Task 17

**Description:**

Currently, when we use a variable more than once the gradient gets overwritten. It can be seen below that the gradient of `x` should be `2` because `y = 2 * x`, but it is instead `1`.

![w02_10_result_bug](../assets/w02_10_result_bug.svg?raw=true "w02_10_result_bug.svg")

To fix this, we can accumulate the gradient instead of resetting it every time `_backward` is called.

**Acceptance criteria:**

1. The bug is fixed.

**Test case:**

```console
python task17.py
```

![w02_10_result](../assets/w02_10_result.svg?raw=true "w02_10_result.svg")

#### Task 18

**Description:**

Extend the value class to allow the following operations:

- adding a float to a `Value` object;
- multiplying a `Value` object with a float;
- dividing a `Value` object by a float;
- exponentiation of a `Value` object with a float;
- exponentiation of Euler's number with a `Value` object.

We'll add the backpropagation (i.e. the implementation of the `_backward` function) in another task, so you needn't add it here.

**Test cases:**

```python
def main() -> None:
    x = Value(2.0, label='x')

    expected = Value(4.0)

    actuals = {
        'actual_sum_l': x + 2.0,
        'actual_sum_r': 2.0 + x,
        'actual_mul_l': x * 2.0,
        'actual_mul_r': 2.0 * x,
        'actual_div_r': (x + 6.0) / 2.0,
        'actual_pow_l': x**2,
    }

    assert x.exp().data == np.exp(
        2), f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, f'Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}.'

    print('All tests passed!')
```

```console
All tests passed!
```

#### Task 19

**Description:**

Break down the hyperbolic tangent into the expressions that comprise it and backpropagate through them.

**Test case:**

```console
python task19.py
```

![w02_11_result](../assets/w02_11_result.svg?raw=true "w02_11_result.svg")

### Using `Value`s to create `Neuron`s

#### Task 1

**Description:**

Implement a class `Neuron` that models one neuron. It should accept the number of inputs that will be passed to it. Initialize the weights and bias by drawing a random value from the uniform distribution between `-1` and `1`. Upon calling the neuron with a list holding floating-point values, it should return the `Value` obtained after applying the hyperbolic tangent activation function.

**Test case:**

Running this:

```python
def main() -> None:
    np.random.seed(42)
    n = Neuron(2)
    x = [2.0, 3.0]
    print(n(x))
```

should produce the following output:

```console
Value(data=0.9903860468404846)
```

#### Task 2

**Description:**

Implement a class `Linear` that models one layer of neurons. It should accept the number of inputs and the number of desired outputs.

**Test case:**

Running this:

```python
def main() -> None:
    np.random.seed(42)
    n = Linear(in_features=2, out_features=3)
    x = [2.0, 3.0]
    print(n(x))
```

should produce the following output:

```console
[Value(data=0.9903860468404846), Value(data=-0.9822311315316595), Value(data=0.5591676286573155)]
```

#### Task 3

**Description:**

Implement a class `MLP` that models a multilayer perceptron. It should accept the number of inputs and a list in which each element is the size of the corresponding hidden layer.

For the sake of readability if the last layer contains only a single neuron, just output the `Value` object instead of a list with one element inside it.

Examples:

- if the parameters are `in_channels=3, hidden_channels=[4, 2]`, we'd create the following model:

![w02_mlp_ex1](../assets/w02_mlp_ex1.png "w02_mlp_ex1.png")

- if the parameters are `in_channels=3, hidden_channels=[4, 4, 4, 1]`, we'd create the following model:

![w02_mlp_ex2](../assets/w02_mlp_ex2.png "w02_mlp_ex2.png")

**Test case:**

Running this:

```python
def main() -> None:
    np.random.seed(42)
    x = [2.0, 3.0, -1.0]
    
    n1 = MLP(in_channels=3, hidden_channels=[4, 4, 1])
    print(n1(x))
    
    n2 = MLP(in_channels=3, hidden_channels=[4, 4, 2])
    print(n2(x))
```

should produce the following output:

```console
Value(data=-0.6490834594731474)
[Value(data=0.715728690853701), Value(data=0.6594519942431283)]
```

and the following `svg` file representing the first model:

![w02_p2_03_res](../assets/w02_p2_03_res.svg?raw=true "w02_p2_03_res.png")

#### Task 4

**Description:**

Create a multilayer perceptron that takes inputs with three features and has the following architecture:

- four neurons in the first layer;
- four neurons in the second layer;
- one neuron in the third layer.

Use list comprehension to calculate the mean squared error of the model given the input data.

Backpropagate the error and output the gradient of the first weight in the first neuron in the first layer.

Add a method `parameters` to each of the classes `Neuron`, `Layer`, and `MLP` that returns a list of the parameters of the corresponding instance.

**Test case:**

Running this:

```python
def main() -> None:
    np.random.seed(42)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = ... # fill in the blank
    y_preds = ... # fill in the blank
    print(f'{y_preds=}')

    loss = ... # fill in the blank

    # fill in code here if any

    print(
        f'Gradient of the first weight in the first neuron in the first layer: {...}.') # fill in the blank

    print(f'Parameters: {n.parameters()}')
```

should produce the following:

```console
y_preds=[Value(data=-0.8846794611396062), Value(data=-0.7655833251200287), Value(data=-0.9256532572066426), Value(data=-0.6535216456575766)]
Loss = Value(data=6.346629119525382)
Gradient of the first weight in the first neuron in the first layer: 0.18568556339367487.
Parameters: [Value(data=-0.08349520625906504), Value(data=-0.757783290179419), Value(data=0.7456953285658869), Value(data=-0.07822853995669132), Value(data=0.4228662760491595), Value(data=-0.05391857755080154), Value(data=0.820925467757954), Value(data=0.27423668525003597), Value(data=-0.45539799607252074), Value(data=-0.4023983717258177), Value(data=-0.9124163469404012), Value(data=0.57519058348993), Value(data=-0.29908061257389806), Value(data=0.4694451549099634), Value(data=0.7529125565303192), Value(data=-0.3283177176813512), Value(data=-0.9548213736184752), Value(data=-0.263165100116167), Value(data=0.806242720079227), Value(data=0.42442150853518523), Value(data=-0.9067891341587697), Value(data=-0.35060677921742167), Value(data=-0.33241568633034024), Value(data=-0.34189605752468966), Value(data=0.28969332240347456), Value(data=0.9057043946477801), Value(data=0.20529429085351714), Value(data=-0.51790827248185), Value(data=0.17767013161301426), Value(data=-0.2752278534568091), Value(data=0.17091117977672687), Value(data=-0.360523556799748), Value(data=-0.07809609132969375), Value(data=0.17963725596278945), Value(data=0.324658772643595), Value(data=-0.13407270208072242), Value(data=0.7286205745164751), Value(data=-0.5246256316422828), Value(data=0.15718187764009262), Value(data=-0.622756518410482), Value(data=-0.537024340390462)]
```

#### Task 5

**Description:**

Make a gradient step with with rate `0.1` and print the new value for the loss.

**Test case:**

The output should be similar to:

```console
Before gradient step: Value(data=6.151016417765895)
After gradient step: Value(data=3.101746496829328)
```

#### Task 6

**Description:**

Train the neural network for 10 epochs. Before each epoch, print the value for the gradient of the first weight in the first neuron in the first layer. After each epoch print the value of the loss function. Print the predictions after the model has been trained.

Answer the following two questions in comments:

1. What do you notice about the gradient of the first weight in the first neuron in the first layer?
2. Why is this a problem?

**Test case:**

```console
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=0.8800466349637817
Gradient of the first weight in the first neuron in the first layer: -2.2111900022519513.
Loss=0.12163317937937812
Gradient of the first weight in the first neuron in the first layer: -2.238355613536774.
Loss=0.023551924796726468
Gradient of the first weight in the first neuron in the first layer: -2.2380494593740137.
Loss=0.0029305439393419045
Gradient of the first weight in the first neuron in the first layer: -2.238015391045233.
Loss=0.0003524054621705247
Gradient of the first weight in the first neuron in the first layer: -2.2380129539687306.
Loss=4.921231786613826e-05
Gradient of the first weight in the first neuron in the first layer: -2.238012811546087.
Loss=8.182235721900147e-06
Gradient of the first weight in the first neuron in the first layer: -2.238012805507104.
Loss=1.5659281904199408e-06
Gradient of the first weight in the first neuron in the first layer: -2.2380128055344684.
Loss=3.3146113125658624e-07
Gradient of the first weight in the first neuron in the first layer: -2.2380128055941713.
Loss=7.546109301885475e-08
Predictions:
[0.9999043780596674, -0.9999977438723988, -0.9999971681031702, 0.9999048717855669]
```

#### Task 7

**Description:**

Fix the bug in the previous task and train the model again.

**Test case:**

Now, the output should be similar to the following:

```console
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=7.8379562353265575
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=7.787174174653229
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=7.691364553815586
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=7.448844880945114
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=6.36591085371192
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=5.125592101808614
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=4.864137579611206
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=1.371604846312807
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=0.7152830758881096
Gradient of the first weight in the first neuron in the first layer: 0.0.
Loss=0.38116458989288804
Predictions:
[0.8177107225177095, -0.9194337489861797, -0.6755458517388393, 0.7053918306965871]
```

### Implementing activation functions

#### Task 1

**Description:**

Since we'll be defining different modules in `dl_lib` which we would like to store in containers, let's define a common interface that they should follow. Define a class `Module` that inherits `abc.ABC` in a module `nn` that has two methods:

- `forward`: runs the forward logic of the module.
- `__call__`: calls the method `forward`.

**Acceptance criteria:**

1. The interface is defined as described.

#### Task 2

**Description:**

Let's begin adding modules to `dl_lib`. Define a class `Sigmoid` in the module `nn` that has one method:

- `forward`: takes an input and applies the sigmoid function element-wise.

When testing the module instead of providing it lists of values or `numpy` array, try to pass PyTorch `tensor` objects. Just like `numpy` you can generate random numbers `torch.randn(how_many_to_generate)` or construct your own tensor via `torch.tensor([1, 2, 3])`.

**Acceptance criteria:**

1. Tests are present.
2. The class accepts and returns PyTorch `tensor` objects.
3. The class is defined in a module `dl_lib.nn`.
4. The module inherits `dl_lib.nn.Module`.

#### Task 3

**Description:**

Let's continue with an implementation for the activation function hyperbolic tangent. Define a class `Tanh` in the module `nn` that is analogous to the class `Sigmoid`, but for the hyperbolic tangent.

**Acceptance criteria:**

1. Tests are present.
2. The class accepts and returns PyTorch `tensor` objects.
3. The class is defined in a module `dl_lib.nn`.
4. The module inherits `dl_lib.nn.Module`.

#### Task 4

**Description:**

Next is the rectified linear unit. Define a class `ReLU` in the module `nn` that applies the rectified linear unit function element-wise.

**Acceptance criteria:**

1. Tests are present.
2. The class accepts and returns PyTorch `tensor` objects.
3. The class is defined in a module `dl_lib.nn`.
4. The module inherits `dl_lib.nn.Module`.

#### Task 5

**Description:**

Let's try to implement a variant of `ReLU` that instead of assigning $0$ to negative values, returns a small, non-zero negative value.

$$\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)$$

Define a class `LeakyReLU` in the module `nn` that applies the leaky-relu function element-wise. It should have one parameter:

- `negative_slop`: Controls the angle of the negative slope. Default to `1e-2` (`0.01`).

**Acceptance criteria:**

1. Tests are present.
2. The class accepts and returns PyTorch `tensor` objects.
3. The class is defined in a module `dl_lib.nn`.
4. The module inherits `dl_lib.nn.Module`.

#### Task 6

**Description:**

When defining neural networks we're often interested in a more declarative style for the definition: we would like to define the layers using classes and use their order as an implicit way of passing the results through each other.

Because of that let's define a class `Sequential`. It'll be a sequential container. It should have the following API:

- `forward(input)`: Runs the input through the chain of classes/layers.
- `append(module)`: Append a given module to the end.
- `extend(sequential)`: Extends the current container with layers from another container.
- `insert(index, module)`: Inserts a module into the container at the specified index.

Here's an example use-case:

```python
model = nn.Sequential(nn.ReLU(), nn.Tanh(), nn.Sigmoid())
result = model(some_inputs)
```

When `model(some_inputs)` is run:

1. Input will first be passed to `ReLU`.
2. The output of the first `ReLU` will become the input for `Tanh`.
3. The output of `Tanh` will be used as input to `Sigmoid`.
4. Finally, the output of the `Sigmoid` will be the output of `model(some_inputs)` and stored in `result`.

**Acceptance criteria:**

1. Tests are present.
2. The class is defined in a module `dl_lib.nn`.
3. The class accepts and returns PyTorch `tensor` objects.
4. All four methods are present.
5. The module inherits `dl_lib.nn.Module`.
