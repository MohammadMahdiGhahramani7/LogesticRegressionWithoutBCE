import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class My_Sigmoidal_Model:

  def __init__(self):

    pass

  def _sigmoid_(self, num):

    return 1 / (1 + np.exp(-num))

  def _forward_(self, X, W):

    """
    X : [1, x]
    W : [b, w]
    """

    summation = np.dot(W.T, X)

    return self._sigmoid_(summation)

  def backward(self, X, W, t, lr):

    """
    t : y_real
    lr : learning rate
    """

    o = self._forward_(X, W)

    W = W + (lr * (t - o) * self._sigmoid_(o) * (1 - self._sigmoid_(o)) * X)

    return W

  def fit(self, training_df, inp_cols, target_col, epochs, lr):

    """
    epochs : number of iterations
    lr : learning rate
    """

    features, target = training_df[inp_cols], training_df[target_col]
    self.W = np.zeros(len(features.columns) + 1)

    for epoch in range(epochs):

      for idx in range(len(training_df)):

        new_sample = features.iloc[idx, :].to_numpy()
        new_label = target[idx]        

        X = np.append(1, new_sample)  # X = [1, x]        

        self.W = self.backward(X, self.W, new_label, lr)
        # Y = mX + h
    self.slope = -self.W[1] / self.W[2]
    self.intercept = -self.W[0] / self.W[2]

    print(f"The number of iteration: {epochs}\nLearning rate: {lr}\nb: {self.W[0]}\nw1: {self.W[1]}\nw2: {self.W[2]}")

  def predict(self, pre_sample):

    X_pre = np.append(1, pre_sample) # X = [1, x]

    if self._forward_(X_pre, self.W) >= 0.5:

      return 1

    else:

      return 0
      
"""
x = [i for i in range(1, 11)]
y = [4, 1, 1.5, 5, 8, 6, 3, 1, 7.5, 0]
target = [1, 0, 0, 1, 1, 1, 0, 0, 1, 0]

training_data = pd.DataFrame(None)
training_data["feature1"] = x
training_data["feature2"] = y
training_data["target"] = target

epochs = 5000
lr = 0.001

model = My_Sigmoidal_Model()

model.fit(training_df = training_data, inp_cols = ["feature1", "feature2"], target_col = "target", epochs = epochs, lr = lr)

plt.figure(figsize=(11, 6))

m = model.slope
h = model.intercept

training_data['f_pred'] = m * training_data["feature1"] + h

sns.scatterplot(data = training_data, x = "feature1", y = "feature2", hue = "target", palette = ["red", "green"], s = 40)
sns.lineplot(data = training_data, x = "feature1", y = "f_pred")

plt.show()

"""

"""
unseen_data = np.array([11, 8])

model.predict(unseen_data)
"

