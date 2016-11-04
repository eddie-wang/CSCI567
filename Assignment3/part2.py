import scipy.io as sio
import numpy as np
from svmutil import *
def preprocess(data_set):
	features=data_set["features"]
	newfeatures=[]
	feature_to_expand=[2,7,8,14,15,16,20,29]
	for index ,feature in enumerate(features.T):
		if index not in feature_to_expand:
			newfeatures.append(feature)
		else:
			new_feature_1=[0]*len(feature)
			new_feature_2=[0]*len(feature)	
			new_feature_3=[0]*len(feature)	
			for i,j in enumerate(feature):
				if j==-1:new_feature_1[i]=1
				if j==0:new_feature_2[i]=1
				if j==1:new_feature_3[i]=1
			newfeatures.append(new_feature_1)  
			newfeatures.append(new_feature_2) 
			newfeatures.append(new_feature_3) 
	data_set["features"]=np.asarray(newfeatures).T		   		


if __name__ == '__main__':
	train_set=sio.loadmat("phishing-train.mat")
	test_set=sio.loadmat("phishing-test.mat")
	preprocess(train_set)
	preprocess(test_set)
	print len(train_set["lable"]）, len(train_set["features"]) 
	prob = svm_problem(train_set["label"],train_set["features"])
	param = svm_parameter('-c 4 -v 3')
	m=svm_train(prob,param) 
	print m

	