from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import math
class Data(dict):
	def __init__(self,** kwargs):
		super(Data,self).__init__(kwargs)
	def __setattr__(self,key,value):
		self[key]=value
	def __getattr__(self, key):	
		return self[key]
def split_data(boston):
	test_data_data,test_data_target=[],[]
	train_data_data,train_data_target=[],[]
	for  index, (data, target) in enumerate (zip(boston.data,boston.target)):
		if index%7==0:
			test_data_data.append(data)
			test_data_target.append(target)
		else:
			train_data_data.append(data)
			train_data_target.append(target)
	return Data(data=np.array(train_data_data),target=np.array(train_data_target)), \
			Data(data=np.array(test_data_data),target=np.array(test_data_target))

def data_plot(train_data):
	for i in range(13):
		ax = fig.add_subplot(5,3,i+1)
		data=train_data.data.T[i]
		# b=list(np.arange(data.min(),data.max(),(data.max()-data.min())/10))

		ax.hist(data,bins=10)
	plt.show()

def pearson_correlation(X,Y):
	meanX,meanY=X.mean(),Y.mean()
	stdX,stdY=X.std(),Y.std()
	return    abs(((X-meanX)*(Y-meanY)).mean()/(stdX*stdY))
def standardize(train_data,test_data):
	data=train_data.data.T
	mean,std=data.mean(axis=1),data.std(axis=1)
	def processer(data,mean,std):
		return (data-mean)/std

	for attribute_id,item in enumerate(data) :
		data[attribute_id]=[processer(i,mean[attribute_id],std[attribute_id]) for i in item ]

	data=test_data.data.T
	for attribute_id , item in enumerate(data):
		data[attribute_id]=[processer(i,mean[attribute_id],std[attribute_id]) for i in item ]
	# normalize target values
	# mean,std=train_data.target.mean(),train_data.target.std()
	# train_data.target=[processer(i,mean,std) for i in train_data.target]
def highestCorrelation(train_data):
	return sort_features(train_data,pearson_correlation)
def select_features(data,features):
	new_data=(data.data.T[features]).T
	return Data(data=new_data,target=data.target)

def combination(data,result,cur):
	if len(cur) == 4:
		result.append(cur)
	else :
		for i ,item in enumerate(data):
			temp=list(cur)
			temp.append(item)
			combination(data[i+1:],result,temp)
	return result
def expand_feature(data):
	new_data = []
	original_data = data.data
	new_data.append(original_data.T)
	for i in range(len(original_data.T)):
		for j in range(i,len(original_data.T)):
			new_attribute=original_data.T[i]*original_data.T[j]
			new_data.append(new_attribute)
	return Data(data=np.vstack(new_data).T, target=data.target)
def sort_features(train_data,func):
	data,target = train_data.data,train_data.target
	result=[]
	for attribute_id , attribute in enumerate(data.T):
		result.append((attribute_id,func(attribute,target)))
	result.sort(key=itemgetter(1),reverse=True)
	return [ item[0] for item in result[:4] ]
def helper(l1,l2):
	for i in l1:
		if i not in l2 : return i
class LinearRegression:
	def __init__(self,train_data):
		self.train_data=train_data
		self.train_data_num,self.attribute_num=train_data.data.shape
		self.test_data_num=test_data.data.shape[0]
		self.train()
	def train(self):
		'''
			w=(X^TX)^-(1)X^Ty
			X=(1,x1 ^T,x2^T,x3^T...xn^T)      N*(D+1) 
			y=(y1,y2...)   n*1
		'''
		X = np.empty([self.train_data_num, self.attribute_num+1])
		X[:,0]=np.ones((self.train_data_num))
		X[:,1:]=self.train_data.data
		Y=self.train_data.target.T
		self.w= np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)

	def test(self,data):
		mse=0.0
		for i in range(len(data.target)):
		 	#print data.data[i].dot(self.w[1:])+self.w[0]-data.target[i]
			mse+=(data.data[i].dot(self.w[1:])+self.w[0]-data.target[i])**2
		return mse/len(data.target)
	def predict(self,data):
		return [  data.data[i].dot(self.w[1:])+self.w[0]  for i in range(len(data.target))]
class RidgeRegression:
	def __init__(self,train_data,_lambda,test_data):
		self.train_data=train_data
		self._lambda=_lambda
		self.train_data_num,self.attribute_num=train_data.data.shape
		self.test_data=test_data
		self.train()

	def train(self,train_data=None,_lambda=None):

		if train_data==None : train_data=self.train_data 
		if _lambda==None : _lambda=self._lambda 
		train_data_num=len(train_data.data)

		X = np.empty([train_data_num, self.attribute_num+1])
		X[:,0]=np.ones((train_data_num))
		X[:,1:]=train_data.data
		Y=train_data.target.T
		w=np.linalg.pinv(X.T.dot(X)+_lambda*np.identity(self.attribute_num+1)).dot(X.T).dot(Y)
		self.w=w
		return w
	def test(self,data):
		return self.mse(self.w,data)
	def mse(self,w,data):
		mse=0.0
		for i in range(len(data.target)):
			#print data.data[i].dot(self.w[1:])+self.w[0]-data.target[i]
			mse+=(data.data[i].dot(w[1:])+w[0]-data.target[i])**2
		return mse/len(data.target)
	def cross_validation(self):
		_lambdas=[  0.0001*(10**i) for i in range(6)]
		train_datasets=[]
		cross_validation_fold=10

		datas=[]
		targets=[]
		for i in range(cross_validation_fold):
			datas.append(self.train_data.data[i:self.train_data_num:cross_validation_fold])
			targets.append(self.train_data.target[i:self.train_data_num:cross_validation_fold])
		data=[  np.vstack(datas[:i]+datas[i+1:] ) for i in range(cross_validation_fold)]
		target=[np.hstack(targets[:i]+targets[i+1:] ) for i in range(cross_validation_fold)]
		train_datasets=[Data(data = data[i] , target=target[i]) for i in range(cross_validation_fold)]
		test_datasets=[Data(data = datas[i] , target=targets[i]) for i in range(cross_validation_fold)]
		for _lambda in _lambdas:
			w=np.array([self.train(train_datasets[i],_lambda=_lambda) for i in range(cross_validation_fold)])
			mse= np.array([self.mse(_w,test_datasets[index]) for index,_w in enumerate(w)])		
			print "when lambda = " , _lambda,  "CV result is " , mse.mean(), "the mse for test set is ",RidgeRegression(self.train_data,_lambda,self.test_data).test(self.test_data)	

if __name__ == '__main__':
	
	'''
		3.1 
	'''
	boston = load_boston()
	train_data,test_data=split_data(boston)
	data_plot(train_data) #plot data#
	standardize(train_data,test_data) #standardize data#
	for attribute_id , attribute in enumerate(train_data.data.T):	
		print "attribute #" ,attribute_id ,"'s pearson_correlation is",pearson_correlation(attribute,train_data.target)
	'''
		3.2 Linear regression
	'''
	linear_regression=LinearRegression(train_data)
	train_result=linear_regression.test(train_data)
	test_result=linear_regression.test(test_data)
	print "3.2 Linear Regression : the mse for training set is ",train_result," for testing set is ",test_result

	print "3.2 Ridge Regression:"
	for _lambda in [0.01 ,0.1,1.0]:
		ridgeRegression =RidgeRegression(train_data,_lambda,test_data)
		train_result=ridgeRegression.test(train_data)
		test_result=ridgeRegression.test(test_data)
		print "when lambda is ", _lambda," : the mse for training set is ",train_result," for testing set is ",test_result
	ridgeRegression =RidgeRegression(train_data,0,test_data)
	print "3.2 Ridge Regression with Cross-Validation :"
	ridgeRegression.cross_validation()

	'''
		3.3 Feature Selection
	'''

	'''
	(a)
	'''
	features=highestCorrelation(train_data) # index starting from 0
	linear_regression=LinearRegression(select_features(train_data,features))
	train_result=linear_regression.test(select_features(train_data,features))
	test_result=linear_regression.test(select_features(test_data,features))
	print "3.3 Selection with Correlation (a) :  Four selected features is ",features , " MSE for training set is ",train_result," MSE for testing set is ",test_result
	'''
	(b)
	'''
	predicted=np.zeros(len(train_data.data))
	features=[]
	for i in range(4):
		new_train_data=Data(data=train_data.data,target=train_data.target-predicted)
		t= helper(highestCorrelation(new_train_data), features)
		features.append(t)
		linear_regression=LinearRegression(select_features (train_data,features))
		predicted=linear_regression.predict(select_features(train_data,features)) #asfas#
		linear_regression=LinearRegression(select_features (train_data,features))	
	print "3.3 Selection with Correlation (b):  Four selected features is ",features , " MSE for training set is ",linear_regression.test(select_features(train_data,features))," MSE for testing set is ",linear_regression.test(select_features(test_data,features))	

	'''
	Selection with Brute-force Search
	'''
	best_train_result=100000
	best_test_result=100000
	for features in combination(range(len(train_data.data[0])),[],[]):
		linear_regression=LinearRegression(select_features(train_data,features))
		train_result=linear_regression.test(select_features(train_data,features))
		test_result=linear_regression.test(select_features(test_data,features))
		if train_result<best_train_result:
			best_train_result,best_features,best_test_result=train_result,features,test_result 
	print "3.3 Selecting with Brute-force search, the best combination features is ",best_features ," MSE for training set is ", best_train_result ," MSE for testing set is",best_test_result


	'''
	Polynomial feature Expansion
	'''
	new_train_data=expand_feature(train_data)
	new_test_data=	expand_feature(test_data)
	standardize(new_train_data,new_test_data) #standardize data#
	linear_regression=LinearRegression(new_train_data)
	train_result=linear_regression.test(new_train_data)
	test_result=linear_regression.test(new_test_data)	
	print "3.4 Polynomial feature Expansion: MSE for training set is ",train_result,"MSE for testing set is" ,test_result
