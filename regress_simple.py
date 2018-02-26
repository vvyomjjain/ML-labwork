import matplotlib.pyplot as plt
import numpy as np
from pylab import *

data = np.genfromtxt("test.csv", delimiter = ',')
print(data)

plt.scatter(data[:,0], data[:,1], s=10)
plt.show()

alpha = 0
beta = 0

def fit(x):
    sumX = np.mean(x[:,0])
    sumY = np.mean(x[:,1])
    arrayY = x[:,1]
    maxY = round(max(arrayY))
    print(maxY)

    print(sumX)
    print(sumY)

    errorX = x[:,0] - sumX
    print(errorX)
    sumX2 = np.sum(np.square(errorX))
    print(sumX2)

    errorY = x[:,1] - sumY
    print(errorY)

    errorXY = errorX * errorY
    print(errorXY)

    errorSum = np.sum(errorXY)
    print(errorSum)

    beta = errorSum/sumX2
    print("Beta")
    print(beta)

    alpha = sumY - beta*sumX
    print("Alpha")
    print(alpha)

    y = [i for i in range(int(maxY))]
    yp = polyval([beta, alpha], y)

    plt.scatter(x[:,0], x[:,1], s=10)
    plt.plot(y,yp, color = 'r')
    plt.show()

    return [alpha, beta]

def predict(value):
    ans = alpha + beta*value
    print(ans)

[alpha, beta] = fit(data)
predict(46)
