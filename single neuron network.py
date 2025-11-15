import random
import math

# training off of AND gate
data = [
    ([0, 0], 0),
    ([1, 0], 0),
    ([0, 1], 0),
    ([1, 1], 1)
]
# init weights and bias
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-1, 1)

# activation (turns result into one or zero)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# sigmoid derivative (how the ai realises and fixes errors)
def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# training
lr = 0.1  # learning rate

for epoch in range(5000): # train 5000 times, each full loop is called an 'epoch'
    for (x1, x2), y in data:
        # forward pass
        z = x1*w1 + x2*w2 + b
        # the line above multiplies input 1 by weight 1
        # then multiplies input 2 by weight 2
        # and finally adds the bias

        pred = sigmoid(z) # simplifiy number into 1 or 0

        # account for error
        loss = (pred - y)

        # backward pass (calculates how much we need to change the weights and bias)
        grad = loss * sigmoid_derivative(z) 

        w1 -= lr * grad * x1 # edit weights based on opposite of error
        w2 -= lr * grad * x2 # same here
        b  -= lr * grad # same with bias

# final results
print("Results after training:")
for (x1, x2), y in data:
    pred = sigmoid(x1*w1 + x2*w2 + b) # calculate guess after one epoch (5000 trains)
    print(f"{x1} AND {x2} = {round(pred)} (raw: {pred})") # prints rounded answer, and raw answer, and the raw answer shows how correct the ai is (if its exactly 1 or 0 then ai has done the perfect amount of training
