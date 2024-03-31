---
title: Intro into Neural Networks
date: 2024/3/31
description: Building a MLP based of Andrej Karpathy's micrograd
tag: ml
author: Me
---

I just completed watching [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&ab_channel=AndrejKarpathy) from Andrej Karpathy and wanted to summarize and go over what I've learned to get a better understanding.

First, although it may seem arbitrary, it is crucial to understand what a derivative is. So a derivative, or the slope, is how much a function/value would change given the input. This comes into play later on when calculating the gradient, which is basically the derivative of the output with respect to the input, of each neuron and for back propogation.

A Neural Network contains of neurons, which for first half of the tutorial looks into it in more detail by observing a Value node. For this node, we defined its basic operations needed for propagation.

```py
class Value:

  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out

# There are more operations but I'm only showing __add__ here.
```

Some of the important notes are:

1. Gradients are initially set to 0, as its value is determined once the back propagation is complete.
2. The \_backward() for the add() function is 1 \* output of the operation. For example: y = x and x = a + b ; Then the gradient for a, which is in this case dy/da would be (dy/dx)(dx/da) = (dy/dx)(1) = (out.grad)(1) due to the chain rule.
3. The signs for assigning the gradients are **+=** instead of **=** because when there are **Value** nodes that are propagated backwards by more than one node, the gradients should be summed instead of being replaced by the last propogation affected by the loop.

```py
# Continued from the previous codeblock (Still inside Value class)
  def backward(self):

    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
```

This backwards function utilizes a topological sort, which I'm not really familar with, but it basically ensures that the edges are directed in one way. Therefore, this function calls \_backward() for each node in reverse topological order from the last node in order to distribute the gradient backwards based off the gradients.

```py
# Continued from the previous codeblock (Still inside Value class)
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out
```

Lastly, I want to talk about the activation function, which in this case is tanh(). tanh() is just a hyperbolic function(bounds between -1 and 1) that is used for this tutorial but other functions could be and are used. In short, the activation function decides whether the neuron would be activated or not and is needed to guarantee non-linearity of its output, and provides some sort of limit to the out value(which is why it is sometimes called the squashing function).

As we now have all the necessary building blocks, we can define a Neuron, which takes in the sum of all the weight\*values and its biases and would output a value based off the activation function(tanh() in this case).

```py
class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
```

The values of the weights and the biases are randomized as we don't want symmetry in the hidden layer. This [post](https://stackoverflow.com/questions/20027598/why-should-weights-of-neural-networks-be-initialized-to-random-numbers) helped me understand why.

Layers just define a list of neurons that take in the number of input neurons(nin) and number of output neurons(nout). The MLP, multi-layer perceptron, takes the number/dimension of input neurons(nin) and the dimension of the layers as a list(nouts). Therefore, call() for the MLP class performs a forward propogation as it calls the propogation from layer to layer, where each layer performs a call() for its neurons.

```py
class Layer:

  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

```

Now, we process this neural network, which involves the forward pass, which allows to calculate the all the values of each neuron goes through the hidden layers and is propagated until it reaches a final output.

```py
for k in range(20):

  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
```

It is important to notice here that for this case, the loss function is the MSE, which measures the mean square of the the error between the predicted and desired values. We want to minimize the loss function, which in other words suggest a more accurate prediction.

Recall from definition of the gradient and the derivative that the gradients of the weights with respect to the loss function will determine the amount that the weight has to change in order to affect the error, which is done here by back propagation.

Furthermore, by considering the gradient as some sort of vector that points in the direction of an increased loss, if we move against the gradient, so in the negative direction of the gradient, we know that the MSE would be minimized, which is shown in the #update section of the code. This step is called the gradient descent, which generally completes the basic process of each iteration.

In conclusion, conducting multiple iterations of: **forward propogation** -> **calculating loss function** -> **backwards propogation** -> **gradient descent** is really what allows the prediction of the neural network to be accurate.

Learning this was pretty fun but I should really study for the finals so hopefully I'll get to work on the makemore from Andrej next time!
