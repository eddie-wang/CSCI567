from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
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
	data=train_data.data.T[0]
	print  data.min() , data.max()
	b=list(np.arange(data.min(),data.max(),(data.max()-data.min())/10))
	print b
	plt.hist(data,bins=10)
	plt.show()

def pearson_correlation(X,Y):
	meanX,meanY=X.mean(),Y.mean()
	stdX,stdY=X.std(),Y.std()
	return ((X-meanX)*(Y-meanY)).mean()/(stdX*stdY)
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
		self.w= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

	def test(self,data):
		mse=0.0
		for i in range(len(data.target)):
			#print data.data[i].dot(self.w[1:])+self.w[0]-data.target[i]
			mse+=(data.data[i].dot(self.w[1:])+self.w[0]-data.target[i])**2
		print mse/len(data.target)
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
		w=np.linalg.inv(X.T.dot(X)+_lambda*np.identity(self.attribute_num+1)).dot(X.T).dot(Y)
		self.w=w
		return w
	def test(self,data):
		mse=0.0
		for i in range(len(data.target)):
			#print data.data[i].dot(self.w[1:])+self.w[0]-data.target[i]
			mse+=(data.data[i].dot(self.w[1:])+self.w[0]-data.target[i])**2
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
			w=w.mean(axis=0)
			self.w=w
			print "when lambda = " , _lambda, "the mse for test set is ",self.test(self.test_data)		
if __name__ == '__main__':
	boston = load_boston()
	train_data,test_data=split_data(boston)
	#data_plot(train_data) #plot data#

	'''
		3.2 Linear regression
	'''
	standardize(train_data,test_data) #standardize data#
	linear_regression=LinearRegression(train_data)
	linear_regression.test(train_data)
	linear_regression.test(test_data)

	ridgeRegression =RidgeRegression(train_data,10,test_data)
	ridgeRegression.test(train_data)
	ridgeRegression.test(test_data)
	ridgeRegression.cross_validation()

	'''
		3.3 Feature Selection
	'''
	fearures=highestCorrelation(train_data)
