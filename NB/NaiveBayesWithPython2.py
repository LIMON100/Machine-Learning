import math
import random
import numpy as np

path = 'H:/Software/Machine learning/Dataset/Diabetis/diabetes.csv'

dataset = np.genfromtxt(path , delimiter = ',' , skip_header = 1 , filling_values = -999 , dtype = 'float' , usecols = [0,1,2,3,4,5,6,7,8])

split_ratio = 0.67

x = dataset[:, :-1]
y = dataset[:, -1]


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * split_ratio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
        
	return separated


def mean(numbers):
	return sum(numbers)/float(len(numbers))



def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)



def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del(summaries[-1])
	return summaries



def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries



def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(0.000001+2*math.pow(stdev,2))))
	return (1 / (0.000001+math.sqrt(2*math.pi) * stdev)) * exponent





def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities




def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel



 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions



 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



def pred():
    trainset ,  testset = splitDataset(dataset, split_ratio)
    summaries = summarizeByClass(trainset)
    prediction = getPredictions(summaries , testset)
    print('Prediction: ',prediction)
    
pred()

def accy():
    trainset ,  testset = splitDataset(dataset, split_ratio)
    summaries = summarizeByClass(trainset)
    prediction = getPredictions(summaries , testset)
    Accuracy = getAccuracy(testset , prediction)
    print('Prediction: ',Accuracy)
    
accy()