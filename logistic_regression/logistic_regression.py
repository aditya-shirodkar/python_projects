#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:05:51 2020

@author: a
"""

import random

random.seed(0)

import math

class LogisticRegression():
    
    def __init__(self, alpha = 0.1):
        
        #initialise learning rate
        self.alpha = alpha
        
        #initialise starting weights
        self.betas = [0.1, 0.1, 0.1]
        
        #empty list which stores the zipped package of data + labels
        self.combined = []
    
    def train(self, dataset, labels, n=100):
        #if data + labels have not yet been initialised:
        data = dataset.copy()
        labels = labels.copy()
        
        #zipping the data and labels so that they shuffle together
        self.combined = list(zip(data, labels))
        
        MSEList = [] #empty list to store MSE values
        
        for i in range(n):
            SEsum = 0 #resets sum squared error to zero at the start of the epoch
            random.shuffle(self.combined) #randomly shuffles the (data+labels)
            
            for j in self.combined:
                self.betas, SE = self._weight_update(j, self.betas)
                SEsum += SE #adds squared error of current point to sum
            
            #appending the MSE value for the list
            MSEList.append(SEsum/len(self.combined))
            
        return MSEList
    
    def _sigma_calc(self, x, betas): #logistic regression function
        
        t = betas[0] + betas[1]*x[0] + betas[2]*x[1]
        sigma = 1/(1+math.exp(-t))
        return sigma
        
    def _weight_update(self, c, betas):
        
        x, l = c #unzip package of data and labels
        b = betas.copy() #copies beta values
        tmp = [1, x[0], x[1]] #x0, x1, x2 values where x0 = 1
        
        sigma = self._sigma_calc(x, betas) #logistic regression function
        
        SE = (sigma - l)**2 #calculate squared error
        
        for i in range(len(betas)): #updates weights
            b[i] -= self.alpha*(sigma-l)*sigma*(1-sigma)*tmp[i]
        
        return b, SE #returns updated betas and squared error
    
    def classify(self, dataset): #classification function
        data = dataset.copy()
        pred = []
        
        #calculating sigma, and returning a predicted label value
        for x in data:
            sigma = self._sigma_calc(x, self.betas)
            if sigma <= 0.5:
                pred.append(0)
            else:
                pred.append(1)
                
        return pred

def printRounded(myList):

    print("[",end="")

    for i in range(len(myList)-1):

        print(str(round(myList[i],7)),end=", ")

    print(str(round(myList[-1],7)),end="]\n")


dataset = [[0,0],[1,0],[2,1],[1,2],[3,1],[4,1],[5,2],[3,3],[2,5]]
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]

lr = LogisticRegression()

MSEList = lr.train(dataset,labels,20)
# MSEList1 = lr.train(dataset,labels,10)
# MSEList2 = lr.train(dataset,labels,10)

predictedLabels = lr.classify(dataset)
print(predictedLabels)
printRounded(MSEList)
# printRounded(MSEList1 + MSEList2)


# dataset = [[-3,-3],[5,4],[-1,0.5],[3.7,9.5],[0.8,-3],[4,-5.6],[1.5,3],[0.4,3.5],[2.5,3.4],[2,-1]]
# labels = [0,1,0,1,1,1,1,0,0,0]

# lr = LogisticRegression(0.015)

# MSEList1 = lr.train(dataset,labels)
# MSEList2 = lr.train(dataset,labels)

# predictedLabels = lr.classify(dataset)
# print(predictedLabels)
# printRounded(MSEList1 + MSEList2)
# # printRounded(MSEList1)
