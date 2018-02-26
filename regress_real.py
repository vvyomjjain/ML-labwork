import matplotlib.pyplot as plt
import numpy as np
from pylab import *

x = np.genfromtxt("sample_stocks.csv", delimiter = ',')
print(x)

plt.scatter(x[:,0], x[:,1], s=10)
plt.show()

theta0 = 0
theta1 = 0
arrayX = x[:,0]
arrayY = x[:,1]
maxX = round(max(arrayX))
minX = round(min(arrayX))
tol = 0.001
max_iter = 300
count = 0
alpha = 0.01
m=len(x)

for i in range(max_iter):

    oldTheta0 = theta0
    oldTheta1 = theta1

    y = polyval([theta1, theta0], arrayX)

    theta0 = theta0 - ( alpha/m )*( np.sum(y) - sum(x[0:,1]) )

    for k in range(len(y)):
        y[k] = (y[k] - arrayY[k])*arrayX[k]

    theta1 = theta1 - (alpha/m)*(sum(y))

    if((abs(theta0 - oldTheta0) + abs(theta1 - oldTheta1)) <= tol):
        break

print("theta0")
print(theta0)
print("theta1")
print(theta1)

y = [i for i in range(int(minX), int(maxX))]
yp = polyval([theta1, theta0], y)
plt.scatter(x[:,0], x[:,1], s=10)
plt.plot(y,yp)
plt.show()
