import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGradientMethod():
    def __init__(self,alphaGiven=0.001,iterGiven=50,thetaGiven=np.zeros(11)):
        self.ALPHA = alphaGiven
        self.MAX_ITER = iterGiven
        self.THETA = thetaGiven
    
    def gradientDescent(self,X, y, theta, alpha, numIterations):
        m = len(y)
        arrCost = [];
        transposedX = np.transpose(X) # transpose X into a vector  -> XColCount X m matrix
        for interation in range(0, numIterations):
            residualError = np.matmul(theta,transposedX)-y
            gradient = np.matmul(residualError,X)
            change = [alpha * x for x in gradient]
            theta = np.subtract(theta, change)  # theta = theta - alpha * gradient
            atmp = (1 / m) * np.sum(residualError ** 2)
            arrCost.append(atmp)
        self.THETA = theta      #Updates Theta for Class
        return arrCost
    
    def getAlpha(self):
        return self.ALPHA
    
    def getMaxIter(self):
        return self.MAX_ITER
    
    def getTheta(self):
        return self.THETA
    
    def updateTheta(self,theta):
        self.THETA = theta
        return
    
    def getGradient(self,xValues, yValues):
        arrCost = self.gradientDescent(xValues, yValues, self.THETA, self.ALPHA, self.MAX_ITER)
        return arrCost

 
    def showCostGraph(self,arrCost):
        plt.plot(range(0,len(arrCost)),arrCost);
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('alpha = {}  theta = {}'.format(self.ALPHA, self.THETA))
        plt.show()
        return
    
    def testModel(self,testXValues,testYValues):
        testXValues
        tVal =  testXValues.dot(self.THETA)
        tError = np.sqrt([x**2 for x in np.subtract(tVal, testYValues)])
        return np.mean(tError), np.std(tError)
