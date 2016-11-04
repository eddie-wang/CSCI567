import numpy as np
import matplotlib.pyplot as plt
import math
def generateDatasets(n):
	datasets=[]
	for i in range(100):
		data=[ (x,2*(x**2)+np.random.normal(0,math.sqrt(0.1),1)[0]) for x in np.random.uniform(-1,1,n) ]
		datasets.append(data)
	# print datasets

	return np.asarray(datasets)

def calculateW(X,Y,regularized=False,_lambda=0):
	if X.size==0: return None
	if not regularized:
		return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)
	else :
		return np.linalg.pinv(X.T.dot(X)+_lambda*np.identity(len(X.T))).dot(X.T).dot(Y)		
def data_plot (plot_data):
	fig = plt.figure()
	for i in range(6):
		ax = fig.add_subplot(2,3,i+1)
		ax.hist(plot_data[i])
	plt.show()
def sum_square_error(result,Y):
	return sum((result-Y)**2)/len(result)
def gaussian_distribution(x):
	sigma=0.1
	mu=0
	return 1/math.sqrt(2*math.pi*(sigma**2))*math.exp(-(x-mu)**2/(2*sigma**2))

def generateSumUsedDatasets(n):
	datasets=[]
	for   i in range(100):
		data=[ (x,2*(x**2)+np.random.normal(0,0.1,1)[0]) for x in np.arange(-1,1, 2.0/n ) ]
		datasets.append(data)
	return np.asarray(datasets)
def bias(ws):

	datasets=generateSumUsedDatasets(200)
	# print datasets[0][:,0]
	_bias=0.0
	ws_avg=np.mean(ws,axis=0) if not ws[0] is None else None
	for _x in datasets[0][:,0]:
		x= np.asarray([ _x**g for g in range(len(ws[0])) ])	if not ws[0] is None else None
		edhdx=x.dot(ws_avg) if not ws [0] is None else 1.0 
		
		ey=2*(_x**2)
		_bias+=(edhdx-ey)**2
	return _bias/200

def variance(ws):
	datasets=generateSumUsedDatasets(200)
	var=0.0;
	ws_avg=np.mean(ws,axis=0)if not ws[0] is None else None
	for d in range(len(datasets)):
		dataset=datasets[d]
		_X=dataset[:,0]
		for _x in _X:
			x= np.asarray([ _x**g for g in range(len(ws[0])) ])	if not ws[0] is None else None
			edhdx=x.dot(ws_avg) if not ws [0] is None else 1.0 
			
			hdx=x.dot(ws[d]) if not ws[d] is None else 1
			py_given_x=gaussian_distribution(_x)
			px=1.0/len(_X)
			var+=((hdx-edhdx)**2)*px
	return var/100

def procedure(n):
	datasets=generateDatasets(n)
	plots_data=[]
	for g in range(6):
		error=[]
		ws=[]
		Ys=[]
		for dataset in datasets:
			_X=dataset[:,0]
			Y=dataset[:,1]
			Ys.append(Y)
			X=np.asarray([ _X**n for n in range(g) ]).T	
			#print X.shape
			w=calculateW(X,Y) 
			ws.append(w)
			result=X.dot(w)if not w is None else np.ones(len(dataset))
			error.append(sum_square_error(result,Y))
		plots_data.append(error)
		# if g==3 : print np.mean(ws,axis=0)
		# if g==4 : print np.mean(ws,axis=0)
		# print  "bias",bias
		# print "variance " ,variance(ws)
		print g+1,"&",bias(ws),"&",variance(ws) ,"\\\\"
	data_plot(plots_data)
if __name__ == '__main__':
	'''
	(a)
	'''
	# procedure(10)
	
	'''
	(b)
	'''
	# procedure(100)

	'''
	(d)
	'''
	plots_data=[]
	_lambdas=[0.001,0.003,0.01,0,03,0.1,0.3,1]
	for _lambda in _lambdas:
		ws=[]
		error=[]
		for dataset in generateDatasets(100):
			_X=dataset[:,0]
			Y=dataset[:,1]
			X=np.asarray([_X**n for n in range(3)]).T
			w=calculateW(X,Y)
			ws.append(w)
			result=X.dot(w)if not w is None else np.ones(len(dataset))
			error.append(sum_square_error(result,Y))
		plots_data.append(error)
		print _lambda,"&",bias(ws),"&",variance(ws) ,"\\\\"
	# data_plot(plots_data)
		


	
