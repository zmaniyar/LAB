import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Generate some random data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.where(X[:,0] + X[:,1] > 0, 1, -1)

# Train the perceptron classifier
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)

# Plot the decision boundary
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1),np.arange(ymin, ymax, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()
