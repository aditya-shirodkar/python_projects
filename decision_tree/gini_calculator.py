#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 19:26:31 2020

@author: a
"""

#first stage of a decision tree

class DecisionTree:
    def __init__(self, data):
        
        def first(item): #function to return first element of list
            return item[0]
        
        #key=first will allow sorting according only to the first element
        #data.copy is done because direct assignment a = b.sort() gives NoneType object
        self.data = data.copy()
        self.data.sort(key = first)
        
 
        
        #for checking if labels are identical/values are the same:
        self.vals = [i[0] for i in self.data]
        self.labels = [i[1] for i in self.data]
        
        #converts to set to capture unique values, and then back to list:
        uniqueVals = list(set(self.vals))
        uniqueVals.sort() #for some reason, the sorting from before is lost
        
        #creates vector to store potential splitting locations
        self.splits = [0]*(len(uniqueVals)-1)
        
        #calculates all possible split values
        for i in range(len(uniqueVals)-1):
            self.splits[i] = (uniqueVals[i]+uniqueVals[i+1])/2
        
    
    def _bestSplit(self):
        gini = 2.0 #to output gini of best split; default is impossible value
        split = float("inf") #best split; original value is impossible
        left = [[]] #empty list of lists
        right = [[]] #empty list of lists
        
        if self.labels.count(self.labels[0]) == len(self.labels):
            return False #returns false if labels are identical
        
        if self.vals.count(self.vals[0]) == len(self.vals):
            return False #returns false if all values are identical
        
        for i in range(len(self.splits)): #iterates over all possible splits
            
            v = [[], []] #stores split data as list of list of lists
            
            partGini = [1, 1] #base, unreduced gini of the two split parts
            giniTemp = 0 #temporary gini storage

            for j in range(len(self.data)):
                
                #to create the two split parts
                if self.data[j][0] < self.splits[i]:
                    v[0].append(self.data[j])
                else:
                    v[1].append(self.data[j])
            
            for j in range(2):
                #calculator of gini for both split parts
                calcList0 = [1 for k in v[j] if k[1] == 0] #counter for label 0
                calcList1 = [1 for k in v[j] if k[1] == 1] #counter for label 1
                #stores gini for part:
                partGini[j] -= (sum(calcList0)/len(v[j]))**2 + (sum(calcList1)/len(v[j]))**2
                
            #combining ginis of both parts:
            giniTemp = (len(v[0])*partGini[0])/len(self.data) + (len(v[1])*partGini[1])/len(self.data)
            
            if (giniTemp < gini): #checking for lowest gini value so far
                gini = giniTemp
                split = self.splits[i] #identifies best split so far
                #assigning best splits so far to left and right:
                left = v[0]
                right = v[1]
            
        return (gini, split, left, right) #returning tuple


dataset = [[5,0],[10,1],[15,0],[5,1],[10,0],[20,1],[20,0],[35,0],[40,1],[50,0],[50,1]]
dt = DecisionTree(dataset)
print(dt._bestSplit())


        
                