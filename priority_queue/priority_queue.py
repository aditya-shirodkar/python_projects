#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:54:00 2020

@author: a
"""

#Creates a queue in which inserted items are assigned a priority which determines removal order

class PriorityNode: #creates nodes with additional priority value
    def __init__(self, content = None, priority = float("inf"), next = None):
        self.content = content
        self.priority = priority
        self.next = next

class PriorityQueue:
    def __init__(self):
        self.length = 0
        self.head = None
    
    def insert(self, content, priority):
        temp = PriorityNode(content, priority) #creates temporary node
        
        if not self.head: #if head is empty
            self.head = temp #attaches new value to head
        else:
            if temp.priority < self.head.priority: #this is done to preserve head index
                temp2 = self.head #preserving the current linked list
                self.head = temp #adding new high priority value
                self.head.next = temp2 #this operation shifts the entire list by one step
                
            else:
                iterator = self.head #fresh pointer instead of using head
                while iterator.next: #if the next node is not None
                    if temp.priority >= iterator.next.priority: #want to see if next...
                        iterator = iterator.next #...element has better (lower) priority
                    else:
                        break #stops iterating at point where new value is inserted
                
                if iterator.next: #similar to the head code, places the value after...
                    temp2 = iterator.next #...end of same priority and before worse...
                    iterator.next = temp #...(higher) priority, shifting the worse...
                    iterator.next.next = temp2 #...elements to the right.
                else: #if there is no worse priority, simply adds value to the end
                    iterator.next = temp
                
        self.length += 1 #increments length of linked list
        
        return True #returns if insert was possible
        
    def remove(self): #same as regular queue
        if (self.length > 0):
            temp = self.head.content
            self.head = self.head.next
            self.length = self.length - 1
            return temp
        else:
            return False
    
    def clear(self):
        self.length = 0
        self.head = None
        
    def is_empty(self):
        return (self.length == 0)



queue = PriorityQueue()

queue.insert(5,1)
queue.insert(7,3)
queue.insert(8,2)
queue.insert(4,3)
queue.insert(30, 40)
queue.insert(5,2)
queue.insert(10,1)

while (not queue.is_empty()):
    print(queue.remove())