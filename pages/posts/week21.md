---
title: PyTorch - Classification with Nerual Networks
date: 2024/5/20
description: Binary and multi-class classification
tag: machine_learning
author: Me
---

<Image
  src="/images/p21.png"
  alt="Photo"
  width={1125}
  height={200}
  priority
  className="next-image"
/>

This my solution and a brief summary to classification section of the PyTorch course that I'm going through right now.

The first exercise was to classify a moon data(interleaving half-circles) from the scikit learn dataset. As always, the first step is to visualize the data of the given problem.

<Image
  src="/images/p21-1.png"
  alt="Photo"
  width={1125}
  height={200}
  priority
  className="next-image"
/>

It is possible to observe that a linear activation function might not be able to accurately describe the dataset as the data itself appears to be non-linear.

```py
# Linear Model
class MoonModelV1(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=10):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)

model_moon = MoonModelV1(input_features=2, output_features=1)

# Bincary cross entropy with Sigmoid layer loss function with SGD optimizer
loss_fn = nn.BCEWithLogitsLoss()()
optimizer = torch.optim.SGD(model_moon.parameters(), lr=0.1)
```

In fact, without a non-linear layer within the model, it is difficult for the model to accurately capture the pattens of the data.

<Image
  src="/images/p21-3.png"
  alt="Photo"
  width={1125}
  height={200}
  priority
  className="next-image"
/>

However, if we utilize non-linear activation functions between our layers, it is possible to capture the non-linear patterns from the moon data set. For this model, the ReLU function was used.

```py
class MoonModelV2(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self, x):
        return self.linear_layer_stack(x)
        # return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))) <-- equivalent

model_moon2 = MoonModelV2(input_features=2, output_features=1, hidden_unites=10)

# Binary Cross Entropy Loss Fn with SGD optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_moon.parameters(), lr=0.1)
```

Now, it is possible to observe a significant improvement in accuracy compared to the previous non-linear model.

<Image
  src="/images/p21-2.png"
  alt="Photo"
  width={1125}
  height={200}
  priority
  className="next-image"
/>

The other exercise was to generate a model that could classify a spiral shaped data set with 3 classes as shown below. The visualized data appears to be non-linear, and the ReLU activation function was used here to adress that as well.

<Image
  src="/images/p21-4.png"
  alt="Photo"
  width={1125}
  height={200}
  priority
  className="next-image"
/>

This model is very similar to the previous classifications; however, achieving the predicted data during the training and testing process, as well as the loss function is a bit different from binary classifications.

For binary classifictions:

```py
  # Binary cross entropy loss function is used
  loss = torch.nn.BCELossWithLogits or torch.nn.BCELoss
  # logits --> prediction probabilities --> prediciton data
  y_pred_probs = torch.sigmoid(y_logits) # convert the logits into probabilities using sigmoid function
  y_pred = torch.round(y_pred_probs) # classify 1 or 0 at the break point of 0.5
```

For multi-class classifictions:

```py
  # Cross entropy loss function is used
  loss = torch.nn.CrossEntropyLoss
  # logits --> prediction probabilities --> prediciton data
  y_pred_probs = torch.softmax(y_logits) # convert the logits into probabilities using softmax function
  y_pred = torch.argmax(y_pred_probs) # return the class with the highest probability
```

To summarize, classification requires some type of activation functions that could normalize the data into some probability distributions. Binary classification utilizes the sigmoid function as it maps/squeezes the values into some values between 0 and 1(proabilities!). Multi-class classification uses the softmax function to squeeze the values into a probability distribution. Then, these probabilities can be retrieved to identify a class that is more likely.

```py
class SpiralModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

model_spiral = SpiralModel(input_features=2, output_features=3, hidden_units=10)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_spiral.parameters(), lr=0.1)
```

Using this non-linear multi-class classification model, it was possible to accurately classify the spiral dataset and its features as shown below. Thanks for reading!

<Image
  src="/images/p21-5.png"
  alt="Photo"
  width={1125}
  height={200}
  priority
  className="next-image"
/>
