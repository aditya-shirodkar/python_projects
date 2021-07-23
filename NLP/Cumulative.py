#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:49:02 2021

@author: a
"""

#Method 2: Cumulative Normalised Frequency

#NOTE: The corpora are split twice: first, a train:test split, where the test
#split bigrams are NOT added to the corpus of known bigrams. While training,
#an additional split is created to form a validation set from the training set.
#The bigrams of the validation set are already present in the corpus of known
#bigrams. The initial test set is then used to capture the final accuracy.

#NOTE: Please use the provided .txt files for the corpora.
# 'english.txt'
# 'czech.txt'
# 'igbo.txt'

from itertools import groupby
import nltk
import random
import regex
import unidecode
from io import StringIO
from html.parser import HTMLParser
import time

#global parameters
n = 100 #top n most frequent bigrams used
m = 20 #top m most informative features displayed
t = 1.0 #fraction of the test data used to create the test set

start = time.time()

#removing XML tags using regex can lead to several rare errors
#therefore, the following XML tag remover was used (credits on next line)
#https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
class XML_Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, data):
        self.text.write(data)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(text):
    s = XML_Stripper()
    s.feed(text)
    return s.get_data()


def fileReader(filePath):
    lines = []
    
    with open(filePath, 'r') as file:
        for line in file:
            #stripping HTML tags:
            line = strip_tags(line)
            #converting to lower case:
            line = line.lower()
            #replacing non-letter characters with spaces:
            line = regex.sub(r'\P{L}', ' ', line)
            #including only those lines which have atleast a letter:
            if regex.sub(r'\P{L}', '', line) != '':
                lines.append(line)
    return lines

def diacriticRemover(lines):
    unaccented = []
    
    for line in lines:
        #converting accented characters to unaccented versions:
        line = unidecode.unidecode(line)
        unaccented.append(line)
        
    return unaccented

def bigramExtractor(lines): #splits corpus into list of bigrams
    bigrams = []
    
    for line in lines:
        for i in range(len(line) - 1):
            #appending as bigram if next character is not a space:
            if line[i] != ' ' and line[i+1] != ' ':
                bigrams.append(line[i] + line[i+1])
    
    return bigrams

#global parameters
n = 100 #top n most frequent bigrams used
m = 20 #top m most informative features displayed

#reading corpora (accented and non-accented versions)
engLines = fileReader('english.txt')
czechLines = fileReader('czech.txt')
igboLines = fileReader('igbo.txt')

#The 'NA' postscript refers to 'no accent'
engLinesNA = diacriticRemover(engLines)
czechLinesNA = diacriticRemover(czechLines)
igboLinesNA = diacriticRemover(igboLines)

#splitting the data into training and testing data, as the classifier should
#be unaware of the bigrams occurring in the test data while it is trained

#for comparison purposes, both, the accented and non-accented versions, are
#shuffled together

eng_shuffle = list(zip(engLines, engLinesNA))
random.shuffle(eng_shuffle)
czech_shuffle = list(zip(czechLines, czechLinesNA))
random.shuffle(czech_shuffle)
igbo_shuffle = list(zip(igboLines, igboLinesNA))
random.shuffle(igbo_shuffle)

#unzipping back to two separate tuples
engLines2, engLinesNA2 = zip(*eng_shuffle)
czechLines2, czechLinesNA2 = zip(*czech_shuffle)
igboLines2, igboLinesNA2 = zip(*igbo_shuffle)

#extracting train and test splits at a 90:10 ratio and converting to lists
engTrain = list(engLines2[int(len(engLines2)*0.1):])
engTest = list(engLines2[:int(len(engLines2)*0.1)])

czechTrain = list(czechLines2[int(len(czechLines2)*0.1):])
czechTest = list(czechLines2[:int(len(czechLines2)*0.1)])

igboTrain = list(igboLines2[int(len(igboLines2)*0.1):])
igboTest = list(igboLines2[:int(len(igboLines2)*0.1)])

engTrainNA = list(engLinesNA2[int(len(engLinesNA2)*0.1):])
engTestNA = list(engLinesNA2[:int(len(engLinesNA2)*0.1)])

czechTrainNA = list(czechLinesNA2[int(len(czechLinesNA2)*0.1):])
czechTestNA = list(czechLinesNA2[:int(len(czechLinesNA2)*0.1)])

igboTrainNA = list(igboLinesNA2[int(len(igboLinesNA2)*0.1):])
igboTestNA = list(igboLinesNA2[:int(len(igboLinesNA2)*0.1)])

#finding all bigrams for each language from training set
engBigrams = bigramExtractor(engTrain)
czechBigrams = bigramExtractor(czechTrain)
igboBigrams = bigramExtractor(igboTrain)

engBigramsNA = bigramExtractor(engTrainNA)
czechBigramsNA = bigramExtractor(czechTrainNA)
igboBigramsNA = bigramExtractor(igboTrainNA)

#finding unique bigrams which are common among multiple corpora

#concatenating together the bigrams from all corpora
allBigrams = engBigrams + czechBigrams + igboBigrams
numBigrams = len(allBigrams)

allBigramsNA = engBigramsNA + czechBigramsNA + igboBigramsNA
numBigramsNA = len(allBigramsNA) #equal to numBigrams

#identifying unique bigrams
uniqueBigrams = sorted(list(set(allBigrams)))

uniqueBigramsNA = sorted(list(set(allBigramsNA)))

#there are too many unique bigrams for classification to converge in a
#reasonable amount of time, especially when considering bigrams with accents
#therefore, it would be reasonable to select only regularly occurring bigrams
#as each corpus has a different length, using just the frequency of occurrence
#would give an incorrect picture; it makes more sense to capture the
#normalised frequency
###

#calculating normalised frequencies
engFreq = [len(list(group))*1000000/len(engBigrams) for key, group in groupby(sorted(engBigrams))]
czechFreq = [len(list(group))*1000000/len(czechBigrams) for key, group in groupby(sorted(czechBigrams))]
igboFreq = [len(list(group))*1000000/len(igboBigrams) for key, group in groupby(sorted(igboBigrams))]

engFreqNA = [len(list(group))*1000000/len(engBigramsNA) for key, group in groupby(sorted(engBigramsNA))]
czechFreqNA = [len(list(group))*1000000/len(czechBigramsNA) for key, group in groupby(sorted(czechBigramsNA))]
igboFreqNA = [len(list(group))*1000000/len(igboBigramsNA) for key, group in groupby(sorted(igboBigramsNA))]

#creating sorted lists of unique bigrams
engBigramsUnique = sorted(list(set(engBigrams)))
czechBigramsUnique = sorted(list(set(czechBigrams)))
igboBigramsUnique = sorted(list(set(igboBigrams)))

engBigramsUniqueNA = sorted(list(set(engBigramsNA)))
czechBigramsUniqueNA = sorted(list(set(czechBigramsNA)))
igboBigramsUniqueNA = sorted(list(set(igboBigramsNA)))

#creating dictionaries of bigram:frequency
engDict = dict(zip(engBigramsUnique, engFreq))
czechDict = dict(zip(czechBigramsUnique, czechFreq))
igboDict = dict(zip(igboBigramsUnique, igboFreq))

engDictNA = dict(zip(engBigramsUniqueNA, engFreqNA))
czechDictNA = dict(zip(czechBigramsUniqueNA, czechFreqNA))
igboDictNA = dict(zip(igboBigramsUniqueNA, igboFreqNA))

allFreq = [0]*len(uniqueBigrams)
allDict = dict(zip(uniqueBigrams,allFreq))

allFreqNA = [0]*len(uniqueBigramsNA)
allDictNA = dict(zip(uniqueBigramsNA,allFreqNA))

#finding cumulative frequency
for key in allDict:
    if key in engDict:
        allDict[key] += engDict[key]
    if key in czechDict:
        allDict[key] += czechDict[key]
    if key in igboDict:
        allDict[key] += igboDict[key]
            
for key in allDictNA:
    if key in engDictNA:
        allDict[key] += engDictNA[key]
    if key in czechDictNA:
        allDictNA[key] += czechDictNA[key]
    if key in igboDictNA:
        allDictNA[key] += igboDictNA[key]

#finding n most frequent bigrams                 
bigramsFrequent = sorted(allDict, key=allDict.get, reverse=True)[:n]

bigramsFrequentNA = sorted(allDictNA, key=allDictNA.get, reverse=True)[:n]

#tagging and concatenating all lines:
labelledTrain = [(line, 'English') for line in engTrain] +\
    [(line, 'Czech') for line in czechTrain] +\
        [(line, 'Igbo') for line in igboTrain]
        
labelledTrainNA = [(line, 'English') for line in engTrainNA] +\
    [(line, 'Czech') for line in czechTrainNA] +\
        [(line, 'Igbo') for line in igboTrainNA]
        
labelledTest = [(line, 'English') for line in engTest] +\
    [(line, 'Czech') for line in czechTest] +\
        [(line, 'Igbo') for line in igboTest]
        
labelledTestNA = [(line, 'English') for line in engTestNA] +\
    [(line, 'Czech') for line in czechTestNA] +\
        [(line, 'Igbo') for line in igboTestNA]

#shuffling both versions together for accurate comparison
train_shuffle = list(zip(labelledTrain, labelledTrainNA))

random.shuffle(train_shuffle)

labelledTrain2, labelledTrainNA2 = zip(*train_shuffle)
labelledTrain2 = list(labelledTrain2)
labelledTrainNA2 = list(labelledTrainNA2)

test_shuffle = list(zip(labelledTest, labelledTestNA))

random.shuffle(test_shuffle)

labelledTest2, labelledTestNA2 = zip(*test_shuffle)
labelledTest2 = list(labelledTest2)
labelledTestNA2 = list(labelledTestNA2)

#feature extractors for accented and non-accented versions of corpora:
def featureExtractor(line):
    features = {}
    for b in bigramsFrequent:
        features['contains({})'.format(b)] = b in line
    return features

def featureExtractorNA(line):
    features = {}
    for b in bigramsFrequentNA:
        features['contains({})'.format(b)] = b in line
    return features

#preparing the test sets
#fraction t of the test sets is used in testing
test_set = [(featureExtractor(line), lab) for (line, lab) in labelledTest[:int(len(labelledTest)*t)]]
test_setNA = [(featureExtractor(line), lab) for (line, lab) in labelledTestNA[:int(len(labelledTest)*t)]]

#training classifier for accented versions of corpora, and validating
featuresets = [(featureExtractor(line), lab) for (line, lab) in labelledTrain]
train_set = featuresets[int(len(featuresets)*0.1):]
validation_set = featuresets[:int(len(featuresets)*0.1)]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print('Both classifiers use the top {} most frequent bigrams.'.format(n))
print('Classifying data with diacritics intact...')
print('Classifier accuracy on validation set:')
print(nltk.classify.accuracy(classifier, validation_set))

print('Classifier accuracy on test set:')
print(nltk.classify.accuracy(classifier, test_set))

print('Classifier accuracy on test set with diacritics removed:')
print(nltk.classify.accuracy(classifier, test_setNA))

print('{} most informative features:'.format(m))
classifier.show_most_informative_features(m)

#training classifier for unaccented versions of corpora
featuresetsNA = [(featureExtractorNA(line), lab) for (line, lab) in labelledTrainNA]
train_setNA = featuresetsNA[int(len(featuresetsNA)*0.1):]
validation_setNA = featuresetsNA[:int(len(featuresetsNA)*0.1)]

classifierNA = nltk.NaiveBayesClassifier.train(train_setNA)

print('Classifying data with diacritics removed...')
print('Classifier accuracy on validation set:')
print(nltk.classify.accuracy(classifierNA, validation_setNA))

print('Classifier accuracy on test set:')
print(nltk.classify.accuracy(classifierNA, test_setNA))

print('Classifier accuracy on test set with diacritics present:')
print(nltk.classify.accuracy(classifierNA, test_set))

print('{} most informative features:'.format(m))
classifierNA.show_most_informative_features(m)

end = time.time()
print('Runtime: {}'.format(end - start))