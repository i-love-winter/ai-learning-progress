# Architecture: 2 inputs, one hidden layer containing 2 neurons, and one output
# Is trained to do simple addition (with randomized data) for numbers between zero and one

import random

# activation functions with leaky relu to prevent dead neurons error

def lrelu(x, a=0.1):
    return x if x > 0 else a * x # if less than zero, make zero, if above keep value

def lrelu_derivative(x, a=0.1):
    return 1 if x > 0 else a # if input is positive, return 1, if negative return 0

class NeuralAdd:
    def __init__(self):
        # 2 inputs, 2 neurons in 1 hidden layer, 1 output
        self.w1 = [[random.uniform(-1,1) for _ in range(2)] for _ in range(2)]  # input -> hidden
        self.b1 = [random.uniform(-1,1) for _ in range(2)]
        self.w2 = [random.uniform(-1,1) for _ in range(2)]  # hidden -> output
        self.b2 = random.uniform(-1,1)
        self.lr = 0.01

    def forward(self, x):
        # pre-activation values
        self.z1 = [
            x[0]*self.w1[0][0] + x[1]*self.w1[1][0] + self.b1[0],
            x[0]*self.w1[0][1] + x[1]*self.w1[1][1] + self.b1[1]
        ]

        # using leaky relu to prevent dead neurons
        self.h = [lrelu(self.z1[0]), lrelu(self.z1[1])]

        # linear output
        out = self.h[0]*self.w2[0] + self.h[1]*self.w2[1] + self.b2
        return out

    def train(self, data, epochs=5000):
        for epoch in range(epochs):
            total_loss = 0
            for x1, x2, y_true in data:
                x = [x1, x2]
                # forward 
                y_pred = self.forward(x)
                loss = (y_pred - y_true) ** 2
                total_loss += loss

                # backprop
                error = y_pred - y_true

                # weight 2 and bias 2 gradients
                dw2 = [error * self.h[0], error * self.h[1]]
                db2 = error

                # hidden layer neuron gradients
                dh = [
                    error * self.w2[0] * lrelu_derivative(self.z1[0]),
                    error * self.w2[1] * lrelu_derivative(self.z1[1])
                ]

                dw1 = [[dh[0]*x[0], dh[1]*x[0]],
                       [dh[0]*x[1], dh[1]*x[1]]]
                db1 = dh

                # update weights (part of backprop)
                for i in range(2):
                    self.w2[i] -= self.lr * dw2[i]
                    self.b1[i] -= self.lr * db1[i]
                    for j in range(2):
                        self.w1[j][i] -= self.lr * dw1[j][i]
                self.b2 -= self.lr * db2

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

# generate randomized training data for addition between 0 and 10
def generate_training_data(n=20, low=0.0, high=10.0, step=0.1):
    data = []
    for _ in range(n):
        # two random numbers in range
        a = round(random.uniform(low, high), 1)
        b = round(random.uniform(low, high), 1)
        # third one is the sum
        c = round(a + b, 1)
        data.append((a, b, c))
    return data

training_data = generate_training_data(20)



# train
nn = NeuralAdd()
nn.train(training_data)

# final test
for x1, x2, y in training_data:
    pred = nn.forward([x1, x2])
    print(f"{x1} + {x2} = predicted: {pred:.3f}, real: {y}")

