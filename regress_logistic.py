import matplotlib.pyplot as plt
import numpy as np
from pylab import *

x = np.genfromtxt("data_classification.csv", delimiter = ',')
print("data:\n", x)

class regress:
    # constructor
    def __init__(self, epoch=15000, tol=0.001, alpha=0.001, theta0=0, theta1=0, theta2=0):
        self.tol = tol
        self.epoch = epoch
        self.alpha = alpha
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2

    # function to fit the data, drawing the line
    def fit(self, data):
        x1_vals = data[:,0]
        x2_vals = data[:,1]
        y_vals = data[:,2]
        x1y_vals = x1_vals*y_vals
        x2y_vals = x2_vals*y_vals

        sum_x1y = sum(x1y_vals)
        sum_x2y = sum(x2y_vals)
        sum_y = sum(y_vals)
        m = len(data)

        for i in range(self.epoch):
            oldTheta0 = self.theta0
            oldTheta1 = self.theta1
            oldTheta2 = self.theta2

            sig = 1 / ( 1 + exp((-1)*self.theta0 - self.theta1*x1_vals - self.theta2*x2_vals) )
            sum_sig = sum(sig)
            sigx1 = sig*x1_vals
            sum_sigx = sum(sigx1)
            sigx2 = sig*x2_vals
            sum_sigx2 = sum(sigx2)

            self.theta0 = oldTheta0 - self.alpha*(sum_sig - sum_y)/m
            print("theta0  ", self.theta0)
            self.theta1 = oldTheta1 - self.alpha*(sum_sigx - sum_x1y)/m
            print("theta1  ", self.theta1)
            self.theta2 = oldTheta2 - self.alpha*(sum_sigx2 - sum_x2y)/m
            print("theta2  ", self.theta2)

            diff = abs(self.theta0 - oldTheta0) + abs(self.theta1 - oldTheta1) + abs(self.theta2 - oldTheta2)
            if(diff <= self.tol):
                errorArr = np.absolute(data[:,2] -  np.round(self.theta0 + self.theta1*data[:,0] + self.theta2*data[:,1]))
                predictedArr = np.round(self.theta0 + self.theta1*data[:,0] + self.theta2*data[:,1])
                print("predicted array:\n", predictedArr)
                print("error array:\n",errorArr)
                error = sum(errorArr)/len(errorArr)
                print("error = ", error)
                
                break



    def predict(self, a, b):
        ans = self.theta0 + self.theta1*a + self.theta2*b
        sig_ans = 1/(1 + exp((-1)*ans))
        print("Predicted value is: ",round(sig_ans))


obj = regress()
obj.fit(x)
obj.predict(4.866015334,2.042671293)
