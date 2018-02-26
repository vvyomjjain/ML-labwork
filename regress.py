import matplotlib.pyplot as plt
import numpy as np
from pylab import *

x = np.genfromtxt("coordinates.csv", delimiter = ',')
print("data:\n", x)

plt.scatter(x[:,0], x[:,1], s = 10)
plt.show()

class regress:
    # constructor
    def __init__(self, epoch=15000, tol=0.001, alpha=0.001, theta0=0, theta1=0):
        self.tol = tol
        self.epoch = epoch
        self.alpha = alpha
        self.theta0 = theta0
        self.theta1 = theta1

    # function to fit the data, drawing the line
    def fit(self, data):
        x_vals = data[:,0]
        min_x = int(min(x_vals))
        max_x = int(max(x_vals))
        y_vals = data[:,1]
        x2_vals = data[:,0]*data[:,0]
        xy_vals = data[:,0]*data[:,1]

        sum_x = sum(x_vals)
        sum_x_2 = sum(x2_vals)
        sum_xy = sum(xy_vals)
        sum_y = sum(y_vals)
        m = len(data)

        for i in range(self.epoch):
            oldTheta0 = self.theta0
            oldTheta1 = self.theta1

            self.theta0 = oldTheta0 - self.alpha*(oldTheta0*m + oldTheta1*sum_x - sum_y)/m
            print("theta0  ", self.theta0)
            self.theta1 = oldTheta1 - self.alpha*(oldTheta0*sum_x + oldTheta1*sum_x_2 - sum_xy)/m
            print("theta1  ", self.theta1)

            diff = abs(self.theta0 - oldTheta0) + abs(self.theta1 - oldTheta1)
            if(diff <= self.tol):
                y = [i for i in range(min_x, max_x + 1)]
                yp = np.polyval([self.theta1, self.theta0], y)

                plt.scatter(x_vals, y_vals, s=10)
                plt.plot(y,yp, color = 'r')
                plt.show()
                break



    def predict(value):
        ans = self.theta0 + self.theta1*value
        print("Predicted value is: ",ans)

obj = regress()
obj.fit(x)
