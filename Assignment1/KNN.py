import sys
import operator
import math
import pandas as pd
import numpy as np
class KNN:
	def __init__(self,training_data_file,testing_data_file):
		self.preprocess_data(training_data_file,testing_data_file)
		#print self.training_data
		self.distance_method=self.euclideanDistance  #by default the distance method is euclidean 
		self.K=1 #by default the k value is 1
	def preprocess_data(self,training_data_file,testing_data_file):
		self.training_data,self.testing_data=[],[]
		with open(training_data_file) as training_file:
			for line in training_file:
				self.training_data.append([  float(item) for item in line.strip().split(",")][1:])
		with open(testing_data_file) as testing_file:
			for line in testing_file:
				self.testing_data.append([float (item) for item in line.strip().split(",")][1:])
		self.standardize()

	def standardize(self):
		df=pd.DataFrame(self.training_data)
		mean=list(df.mean())
		mean[-1]=0
		dv=list(df.std())
		dv[-1]=1
		#print dv
		self.training_data=((np.array(self.training_data)-mean)/dv).tolist()
		self.testing_data=((np.array(self.testing_data)-mean)/dv).tolist()
		##print self.testing_data[0]
	def euclideanDistance(self,instance1, instance2):
		distance = 0
		for x in range(len(instance1)-1):
			distance += pow((instance1[x] - instance2[x]), 2)
		return math.sqrt(distance)

	def manhattanDistance(self,instance1,instance2):
		distance=0
		for x in range(len(instance1)-1):
			distance+=abs(instance1[x]-instance2[x])
		return distance
	def useL1Distance(self):
		self.distance_method=self.manhattanDistance
	def useL2Distance(self):
		self.distance_method=self.euclideanDistance
	def setK(self,k):
		self.K=k
	def train(self):
		acc=0.0
		for i in range(len(self.training_data)):
			c=self.classify(self.training_data[i],self.training_data[0:i]+self.training_data[i+1:])
			if c==self.training_data[i][-1]: 
				acc+=1
		return acc/len(self.training_data)
	def nNearestNeighbor(self,training_set,testing_instance):
		distances=[]
		for item in training_set:
			d=self.distance_method(item,testing_instance)
			distances.append((d,item[-1],item))
		
		return sorted(distances,key=operator.itemgetter(0))[0:self.K]
	def classify(self,testing_instance,training_set=None):
		if training_set==None : training_set=self.training_data
		neighbor=self.nNearestNeighbor(training_set,testing_instance)
		# print neighbor
		votes={}
		for item in neighbor:
			if item[1] in votes:
				votes[item[1]]+=1
			else:
				votes[item[1]]=1
		##print votes
		w=sorted(votes.iteritems(),key=operator.itemgetter(1),reverse=True)
		#print w
		return w[0][0]
	def test(self):
		acc=0.0
		for item in self.testing_data:
			if self.classify(item)==item[-1]:
				acc+=1
		return acc/len(self.testing_data)
if __name__ == '__main__':
	if len(sys.argv)==3:
		training_data_file=sys.argv[1]
		testing_data_file=sys.argv[2]
	else :
		training_data_file="train.txt"
		testing_data_file="test.txt"
	knn=KNN(training_data_file,testing_data_file)
	k=[1,3,5,7]
	for _k in k:
		knn.setK(_k)
		knn.useL2Distance()
		print  "When k = " , _k , "using L1 the training accuracy is",knn.train()," testing accuracy is " ,knn.test() 
		knn.useL2Distance()
		print  "When k = " , _k , "using L2 the training accuracy is",knn.train()," testing accuracy is " ,knn.test() 
	

	
