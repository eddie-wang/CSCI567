import sys
import numpy as np
import warnings
import math
class NBC:
	def __init__(self,training_data_file,testing_data_file):
		self.training_data,self.testing_data=[],[]
		self.preprocess_data(training_data_file,testing_data_file)
		self.train()
	def preprocess_data(self,training_data_file,testing_data_file):
		with open(training_data_file) as training_file:
			for line in training_file:
				self.training_data.append([  float(item) for item in line.strip().split(",")][1:])
		with open(testing_data_file) as testing_file:
			for line in testing_file:
				self.testing_data.append([float (item) for item in line.strip().split(",")][1:])

	def  train(self):
		self.attribute_num=9
		self.class_num=7
		self.mu=[[0 for j in range(self.class_num)] for i in range(self.attribute_num)]
		self.theta=[[0 for j in range(self.class_num)] for i in range(self.attribute_num)]
		self.p=[0.0 for i in range(self.class_num)]
		n=[[[] for j in range(self.class_num)] for i in range(self.attribute_num)]
		for item in self.training_data:
			_class=int(item[-1])
			self.p[_class-1]+=1
			for attribute in range(len(item)-1):
				n[attribute][_class-1].append(item[attribute])
		self.p=[i/len(self.training_data) for i in self.p]
		for attribute in range(self.attribute_num):
			for _class in range(self.class_num):
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					self.mu[attribute][_class]=np.array(n[attribute][_class]).mean()
					self.theta[attribute][_class]=np.array(n[attribute][_class]).std()
	def  get_train_accuracy(self):
		correct=0.0
		for item in self.training_data:
			if self.classify(item)==item[-1] : correct+=1 
		return correct/len(self.training_data)
	def  test(self):
		correct=0.0
		for item in self.testing_data:
			if self.classify(item)==item[-1] : correct+=1 
		return correct/len(self.testing_data)			
	def classify(self,test_data):
		_class,possiblity=-1,0
		for i in range(self.class_num):
			cur=1.0
			for attribute in range(self.attribute_num):
				cur*=self.normal_distribution(self.mu[attribute][i],self.theta[attribute][i],test_data[attribute])
			cur*=self.p[i]
			if cur>possiblity:
				possiblity=cur
				_class=i+1
		return _class
	def normal_distribution(self,mu,theta,x):
		if math.isnan(mu) or math.isnan(theta) or theta==0:
			return 1.0 if x==0 else 0
		return 1/math.sqrt(2*math.pi*(theta**2))*math.exp(-(x-mu)**2/(2*theta**2)) 
if __name__ == '__main__':
	if len(sys.argv)==3:
		training_data_file=sys.argv[1]
		testing_data_file=sys.argv[2]
	else:
		training_data_file="train.txt"
		testing_data_file="test.txt"
	nbc=NBC(training_data_file,testing_data_file)
	print "Training accuracy is", nbc.get_train_accuracy(), "Testing accuracy is " ,nbc.test()