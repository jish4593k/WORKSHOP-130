
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from keras.models import Sequential
from keras.layers import Dense


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = np.arange(len(ys), dtype=np.float64)

    return xs, np.array(ys, dtype=np.float64)

def linear_regression(xs, ys):
    n = len(xs)
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    
    m = (n * np.sum(xs * ys) - np.sum(xs) * np.sum(ys)) / (n * np.sum(xs**2) - (np.sum(xs))**2)
    b = mean_y - m * mean_x

    return m, b

def plot_regression_line(xs, ys, m, b):
    regression_line = m * xs + b
    plt.scatter(xs, ys)
    plt.plot(xs, regression_line, label='Regression Line')
    plt.legend()
    plt.show()

def calculate_r_squared(ys, m, b):
    mean_y = np.mean(ys)
    a1 = np.sum((ys - mean_y)**2)
    a2 = np.sum((ys - m * xs - b)**2)

    r_squared = 1 - (a2 / a1)
    return r_squared

def main_linear_regression():
    xs, ys = create_dataset(40, 20, 2, correlation='pos')
    m, b = linear_regression(xs, ys)

    print(f"Slope (m): {m}, Intercept (b): {b}")

    

    # Plot the regression line
    plot_regression_line(xs, ys, m, b)

    # Calculate the coefficient of determination (r_squared)
    r_squared = calculate_r_squared(ys, m, b)
    print(f"Coefficient of Determination (R^2): {r_squared}")

# Neural Network-based Regression using PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train_torch_model(xs, ys):
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    xs_tensor = torch.from_numpy(xs.reshape(-1, 1)).float()
    ys_tensor = torch.from_numpy(ys.reshape(-1, 1)).float()

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(xs_tensor)
        loss = criterion(outputs, ys_tensor)
        loss.backward()
        optimizer.step()

    return model

def main_torch_regression():
    xs, ys = create_dataset(40, 20, 2, correlation='pos')
    model = train_torch_model(xs, ys)

 
    test_point = torch.Tensor([[8]])

    # Make prediction
    predicted_y = model(test_point)
    print(f"Predicted y for x={test_point.item()}: {predicted_y.item()}")

# Neural Network-based Regression using Keras
def train_keras_model(xs, ys):
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='linear'))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000, verbose=0)
    return model

def main_keras_regression():
    xs, ys = create_dataset(40, 20, 2, correlation='pos')
    model = train_keras_model(xs, ys)

    test_point = np.array([[8]])
    predicted_y = model.predict(test_point)
    print(f"Predicted y for x={test_point[0][0]}: {predicted_y[0][0]}")

if __name__ == "__main__":
    # Linear Regression
    main_linear_regression()

    main_torch_regression()


    main_keras_regression()
