#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 19:12:03 2021

@author: a
"""

#these libraries are only used for testing and comparison purposes:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
import pandas as pd #used for one hot encoding for the scikit-learn dataset

#no imported libraries are used to implement the actual decision tree


def loadData(filePath): #data loading function
    
    data = [] #output list
    file = open(filePath, "r") #opens file for reading

    while True:
        line = file.readline() #reads current line (creates list)
        if len(line) == 0: #exits while loop if no line exists
            break
        line = line.replace("\n", "") #removes newline code
        line_parts = line.split(",") #splits line into comma separated parts
        
        for i in range(len(line_parts)):
            try: #attempts type conversion; if possible, does so
                float(line_parts[i])
                line_parts[i] = float(line_parts[i])
            except ValueError: #if conversion not possible, retains string
                continue
        
        data.append(line_parts) #adds to output list
    
    file.close()
    
    if data[-1] == [""]: #in case last line is empty (originally \n)
        data = data[:-1]
        
    return data

#to get stats of the implemented decision tree versus the decision tree in
#scikit-learn for 10 cross validation runs:
def getStats(data, dataName, depth = None, data2=None):
    
    #data2 is used in case of categorical variables for the purposes of the
    #scikit-learn implementation, which requires encoding
    
    #this is used for the scikit-learn.metrics class, which throws errors for
    #undefined cases
    warnings.filterwarnings("ignore")
        
    #to store stats for the constructed decision tree:
    Main_time_taken = []
    Main_macro_precision = []
    Main_macro_recall = []
    Main_macro_f1 = []
    Main_accuracy = []
    
    #the following is used because the depth mechanics of the two trees differ:
    if depth is None:
        maxLevel = float("inf")
    else:
        maxLevel = depth
    
    
    #to store stats for the SKL implementation of a decision tree:
    SKL_time_taken = []
    SKL_macro_precision = []
    SKL_macro_recall = []
    SKL_macro_f1 = []
    SKL_accuracy = []
    
    #the following implies the data has no categorical variables:
    if data2 is None:
        data2 = data
    
    for i in range(10):
    
        X = [x[:-1] for x in data]
        y = [x[-1] for x in data]
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)
        
        start_time = time.time()
        dt = DecisionTree(X_train, y_train, maxLevel = maxLevel)
        y_pred = dt.predict(X_test)
        Main_time_taken.append(time.time() - start_time)
        
        report = metrics.classification_report(y_test, y_pred, output_dict=True)
        
        Main_macro_precision.append(report["macro avg"]["precision"])
        Main_macro_recall.append(report["macro avg"]["recall"])
        Main_macro_f1.append(report["macro avg"]["f1-score"])
        Main_accuracy.append(report["accuracy"])
        
        #Comparing with the sklearn implemention of the decision tree.
        
        X = [x[:-1] for x in data2]
        y = [x[-1] for x in data2]
        
        
        labelEncoder = LabelEncoder()
        labelEncoder.fit(y)
        y = labelEncoder.transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                            random_state=42)
        
        start_time = time.time()
        dt = DecisionTreeClassifier(max_depth = depth)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        SKL_time_taken.append(time.time() - start_time)
        
        report = metrics.classification_report(y_test, y_pred, output_dict=True)
        
        SKL_macro_precision.append(report["macro avg"]["precision"])
        SKL_macro_recall.append(report["macro avg"]["recall"])
        SKL_macro_f1.append(report["macro avg"]["f1-score"])
        SKL_accuracy.append(report["accuracy"])
    
    
    Main_time_taken.append(round(np.mean(Main_time_taken), 5))
    Main_macro_precision.append(round(np.mean(Main_macro_precision), 5))    
    Main_macro_recall.append(round(np.mean(Main_macro_recall), 5))
    Main_macro_f1.append(round(np.mean(Main_macro_f1), 5))
    Main_accuracy.append(round(np.mean(Main_accuracy), 5))
    
    SKL_time_taken.append(round(np.mean(SKL_time_taken), 5))
    SKL_macro_precision.append(round(np.mean(SKL_macro_precision), 5))    
    SKL_macro_recall.append(round(np.mean(SKL_macro_recall), 5))
    SKL_macro_f1.append(round(np.mean(SKL_macro_f1), 5))
    SKL_accuracy.append(round(np.mean(SKL_accuracy), 5))
                    
    
    comparative_time_taken = np.array([Main_time_taken[-1], 
                                       SKL_time_taken[-1]])
    comparative_precision = np.array([Main_macro_precision[-1], 
                                      SKL_macro_precision[-1]])                               
    comparative_recall = np.array([Main_macro_recall[-1], 
                                   SKL_macro_recall[-1]])
    comparative_f1 = np.array([Main_macro_f1[-1], 
                               SKL_macro_f1[-1]])
    comparative_accuracy = np.array([Main_accuracy[-1], 
                                     SKL_accuracy[-1]])
    
    
    method_names = ("implemented DT", "scikit-learn DT")
    
    plt.figure(figsize = (12,1))
    col_label = ("Method", "Avg. Time Taken", "Avg. Macro Precision",
                 "Avg. Macro Recall", "Avg. Macro f1-score", "Avg. Accuracy")
    
    plt.title("Comparative avg. metrics for 10 runs: {}".format(dataName))
    plt.axis("off")
    comparative_full = np.array([method_names, comparative_time_taken,
                                 comparative_precision, comparative_recall,
                                 comparative_f1, comparative_accuracy]).T
    
    table = plt.table(cellText=comparative_full, colLabels=col_label,
                      loc="center")
    
    plt.show()
    plt.close()
    return table

class Node:
    
    def __init__(self, level=None):
        
        self.split = 0 #stores value to split at
        self.index = 0 #stores column index
        
        self.pred = None #stores prediction
        
        self.left = None
        self.right = None
        
        self.level = level

###
#the following class adapts some code from the week 9 assignment
#this code does not use any imported libraries
#this code has inbuilt tree metrics and visualisation features
#this class does not have a fit function; the data argument is fit by default
###
    
class DecisionTree:
    
    #note: true labels can either be passed separately or with the data
    #if passed with the data, they must be in the last column
    
    #note: please clean the data in advance
    #this programme is not built to handle missing or NaN values
    
    #note: maxLevel and level are used for the depth of the decision tree
    
    def __init__(self, data, label = None, maxLevel = float("inf")):
        
        self.maxLevel = maxLevel
        self.data = data.copy()
        self.tree = None
        
        if label is not None:
            
            #if label and data lengths differ:
            if len(label) != len(self.data):
                raise SystemExit("Error: Data and label sizes do not match.")
                
            for i in range(len(self.data)):
                self.data[i].append(label[i])
        
        uniqueVals = []
        
        for i in range(len(self.data[0])): #extracting unique values of every column
            tmp = [x[i] for x in self.data]
            uniqueVals.append(list(set(tmp)))
        
        #note: uniqueVals: each row (but the last) represents a variable
        #last row represents the label
        
        #marking variable columns as continuous:
        self.isContinuous = [True]*(len(uniqueVals) - 1)
        
        #determining if variables are continuous or discreet categories
        #guideline used -> var is categorical if:
        #   number unique values are less than 5% of number of total values 
        #   values are not numeric
        
        # for i in range(len(uniqueVals) - 1):
        #     if len(uniqueVals[i]) < 0.05*len(self.data):
        #         self.isContinuous[i] = False
                
        #     if len(uniqueVals[i]) <= 5:
        #         self.isContinuous[i] = False

        #     try:
        #         float(self.data[0][i])
        #         continue
        #     except ValueError:
        #         self.isContinuous[i] = False

        for i in range(len(self.data[0]) - 1):
            
            if not self.isContinuous[i]:
                
                for j in range(len(self.data)):
                
                    self.data[j][i] = str(self.data[j][i])
        
        
        self.temp = self.data
        #creating decision tree:
        self.tree = self._makeTree(self.data)
    
    
    
    
    #to calculate nCr combinations of list l
    #n = len(l); r = (parameter)
    def _combinator(self, l, r):
        
        if r == 0: #special case for null set (not used)
            return [[]]
        
        tmp = []
        for i in range(len(l)):
            
            #when len(l) = 0, this does nothing, thus recursion terminates
            
            fixed = l[i]
            remaining = l[i+1:]
            
            #as function returns a list, it can be iterated through
            for x in self._combinator(remaining, r-1):
                tmp.append([fixed] + x)
                
        return tmp
    
    def _findBestSplit(self, vals):
        
        vals = vals.copy() #data used for current split
        
        #the following are initialised with impossible values:
        gini = 2.0 #output gini for best split
        split = float("inf") #output current split value
        index = float("inf") #output current split variable index
        
        label = [x[-1] for x in vals]
        
        #if all label values are the same, move to topmost right list:
        if label.count(label[0]) == len(label):
            
            #returns only the prediction
            return None, None, label[0]
            
        uniqueVals = []
        
        for i in range(len(vals[0])): #extracting unique values of every column
            tmp = [x[i] for x in vals]
            uniqueVals.append(list(set(tmp)))
        
        
        for i in range(len(self.isContinuous)): #iterating over vars only
            
            if self.isContinuous[i]: #for continuous variables
                
                #if all values are the same, continue:
                if vals.count(vals[0][i]) == len(vals):
                    continue
                
                #storing and sorting unique values of current variable:
                tmp = uniqueVals[i]
                tmp.sort()
                
                #to store splitting points for current variable:
                split_points = [0]*(len(tmp) - 1)
                
                for j in range(len(tmp) - 1):
                    split_points[j] = (tmp[j] + tmp[j+1])/2
                
                for x in split_points: #iterates through possible split points
                    
                    left = [] #empty vector to store left split
                    right = [] #empty vector o store right split
                    partGiniLeft = 1.0 #partial gini storage for left split
                    partGiniRight = 1.0 #partial gini storage for right split
                    
                    for j in range(len(vals)):
                        
                        #creating the two split parts:
                        if vals[j][i] < x:
                            left.append(vals[j])
                        else:
                            right.append(vals[j])
                    
                    labelLeft = [x[-1] for x in left]
                    labelRight = [x[-1] for x in right]
                    
                    for j in range(len(uniqueVals[-1])):
                        
                        calcLeft = labelLeft.count(uniqueVals[-1][j])
                        partGiniLeft -= (calcLeft/len(left))**2
                        calcRight = labelRight.count(uniqueVals[-1][j])
                        partGiniRight -= (calcRight/len(right))**2
                                
                    tempGini = (len(left)*partGiniLeft)/len(vals) +\
                        (len(right)*partGiniRight)/len(vals)
                    
                    # print(tempGini)
                    
                    if tempGini < gini:
                        
                        gini = tempGini
                        split = x
                        index = i
                
            if not self.isContinuous[i]: #for categorical variables
                
                #if all values are the same, continue:
                if vals.count(vals[0][i]) == len(vals):
                    continue
                
                #storing and sorting unique values of current variable:
                tmp = uniqueVals[i]
                tmp.sort()
                
                #length of set = n
                #all possible combinations of a set = n^2
                #({nC0} + {nC1} + {nC2} + ... + {nCn})
                #discarding empty set {nC0} and full set {nCn}
                #therefore, (n^2 - 2) possible sets as split conditions
                
                combo_temp = [] #to store output of _combinator()
                
                #calculating nCr combinations, ignoring r = 0 and r = n
                for j in range(len(tmp) - 2):
                    combo_temp.append(self._combinator(tmp, j+1))
                
                #combo_temp is a list of list and needs to be flattened:
                #read this as nested for loop:
                #   for sublist in list -> for x in sublist -> x
                
                combinations = [x for sublist in combo_temp for x in sublist]
                
                for A in combinations: #iterating through combinations
                    
                    left = [] #empty vector to store left split
                    right = [] #empty vector o store right split
                    partGiniLeft = 1.0 #partial gini storage for left split
                    partGiniRight = 1.0 #partial gini storage for right split
                    
                    for j in range(len(vals)):
                        
                        #creating the two split parts:
                        if vals[j][i] in A:
                            left.append(vals[j])
                        else:
                            right.append(vals[j])
                    
                    labelLeft = [x[-1] for x in left]
                    labelRight = [x[-1] for x in right]
                    
                    for j in range(len(uniqueVals[-1])):
                        
                        calcLeft = labelLeft.count(uniqueVals[-1][j])
                        partGiniLeft -= (calcLeft/len(left))**2
                        calcRight = labelRight.count(uniqueVals[-1][j])
                        partGiniRight -= (calcRight/len(right))**2
                                
                    tempGini = (len(left)*partGiniLeft)/len(vals) +\
                        (len(right)*partGiniRight)/len(vals)

                    if tempGini < gini:
                        
                        gini = tempGini
                        split = A
                        index = i
                    
                    #B is the complement of A (B = A')
                    B = list(set(tmp).difference(A))
                    
                    #as B = A', A = B'
                    #removing B from combinations to prevent redundancies
                    
                    B.sort() #to properly locate B in combinations
                    
                    try: #used in case B isn't in combinations
                        combinations.remove(B)
                    except ValueError:
                        continue
        
        #the following is used to give best possible prediction at current
        #stage, even if the data can be split further
        #this is used when the depth provided is less than the splits needed
        pred = max(label, key=label.count)
        
        return split, index, pred
    
    def _makeTree(self, vals, level = 0):
        
        node = Node(level = level)
        
        split, index, pred = self._findBestSplit(vals)
        
        if level < self.maxLevel:
        
            if split and index: #checks if these values are not None
                
                #if the variable to split at is continuous:
                if self.isContinuous[index]:
                    
                    #getting selection indices to left and right of split value:
                    sel_l = [i for i in range(len(vals)) if vals[i][index] < split]
                    sel_r = [i for i in range(len(vals)) if vals[i][index] > split]
                
                #if the variable to split at is categorical:
                if not self.isContinuous[index]:
                    
                    #getting selection indices to in or out of split set:
                    sel_l = [i for i in range(len(vals)) if vals[i][index] in split]
                    sel_r = [i for i in range(len(vals)) if vals[i][index] not in split]
                    
                left = [vals[x] for x in sel_l]
                right = [vals[x] for x in sel_r]
                
                #storing split, index and prediction values in node:
                node.split = split
                node.index = index
                
                #recursively splitting the left and right splits
                node.left = self._makeTree(left, level+1)
                node.right = self._makeTree(right, level+1)
            
        #this triggers only when split and index are None:
        #can trigger if level condition is not matched
        node.pred = pred
        return node
    
    def predict(self, vals):
        
        pred = []
        vals = vals.copy()
        
        for x in vals:
            
            #for every value, starting at the top of the tree:
            node = self.tree
            
            #if node.left is None, it implies node.right is also None:
            while node.left:
                
                #if the variable to split at is continuous:
                if self.isContinuous[node.index]:
                    
                    if x[node.index] < node.split:
                        node = node.left
                        
                    else:
                        node = node.right
                
                #if the variale to split at is categorical:
                if not self.isContinuous[node.index]:
                    
                    if x[node.index] in node.split:
                        node = node.left
                        
                    else:
                        node = node.right
            
            #predict only once the end of the tree is reached:
            pred.append(node.pred)
            
        return pred


#for the iris dataset:
data = loadData("iris_data.txt")
table = getStats(data, "Iris")
table

#using depth = 1 for the iris dataset:
table = getStats(data, "Iris (depth 1)", 1)
table

#using depth = 2 for the iris dataset:
table = getStats(data, "Iris (depth 2)", 2)
table