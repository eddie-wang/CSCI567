import numpy as np
import matplotlib.pyplot as plt
import math
def read_data():
	with open('hw5_blob.csv') as f1, open('hw5_circle.csv') as f2:
		blob=np.loadtxt(f1,delimiter=',')
		circle=np.loadtxt(f2,delimiter=',')
	return blob,circle
def init_centers(data,k):
	# _min,_max=data.min(axis=0),data.max(axis=0)
	# return np.random.rand(k,2)*(_max-_min)+_min
	index=np.random.randint(len(data), size=k)
	return data[index]
def calculate_center(data,r,k):
	centers=np.full((k,2),0.0)
	n=np.full(k,0.0000001) # give a very small value in case the divider is 0
	for i in range(len(r)):
		centers[r[i]]+=data[i]
		n[r[i]]+=1
	return (centers.T/n).T
def distance(x,y):
	return sum((x-y)**2)
def kernel(x,y):
	# RBF
	# gamma=0.1
	# return math.exp(-gamma*distance(x,y))
	# polynimial
	# c=0
	# d=1
	# return (x.T.dot(y)+c)**3
	return math.sqrt(sum(x**2))*math.sqrt(sum(y**2))
def kmeans(data,k):
	centers=init_centers(data,k)
	r=[-1]*len(data)
	changed=True
	while changed:
		changed=False
		for n in range(len(data)):
			distances=np.asarray([ distance(data[n],centers[i]) for i in range(k)])
			c=distances.argmin()
			if r[n]!=c: changed=True
			r[n]=c
		centers=calculate_center(data,r,k)	
	return r

def init_r(centers,data):
	r=[-1]*len(data)
	for n in range(len(data)):
		distances=np.asarray([kernel(data[n],data[n])+kernel(centers[i],centers[i])-2*kernel(data[n],centers[i]) for i in range(len(centers))])
		c=distances.argmin()
		r[n]=c
	return r
def kernel_distance(point,c,r,data,b,c_index):
	a=sum([ kernel(data[i],point) for i in c_index])
	return kernel(point,point)-2.0*a/(len(c_index)+0.000001)+1.0*b/(len(c_index)**2+0.000001)
def kernel_kmeans(data,k):
	centers=init_centers(data,k)
	r=init_r(centers,data)
	changed=True
	index=[-1]*k
	b=[0]*k
	while changed:
		changed=False
		for _k in range(k):
			index[_k]=[ i for i in range(len(r)) if r[i]==_k]
			b[_k]=sum([ kernel(data[i],data[j]) for i in index[_k] for j in index[_k] ])
		for n in range(len(data)):
			distances=np.asarray([ kernel_distance(data[n],i,r,data,b[i],index[i]) for i in range(k)])
			c=distances.argmin()
			if r[n]!=c: 
				changed=True
			r[n]=c
		print "new round"
	return r

def gaussian(mu,sigma,x):
	'''
		two dimensional gaussion ditribution
		all parameter need to be numpy array type
	'''
	return math.exp(-0.5* ((x-mu).T).dot(np.linalg.inv(sigma)).dot(x-mu) )/math.sqrt(((2*math.pi)**2)* np.linalg.det(sigma))

def init_parameters(data,k):
	r=kmeans(data,k)
	d=[[],[],[]]
	for i in range(len(r)):
		d[r[i]].append(data[i])
	return {"mu":[np.array(d[0]).mean(axis=0),np.array(d[1]).mean(axis=0),np.array(d[2]).mean(axis=0)]  , "sigma":[np.cov(np.array(d[0]).T),np.cov(np.array(d[1]).T),np.cov(np.array(d[2]).T)], "w":[len(d[0]),len(d[1]),len(d[2])]}

def compute_gamma(data,theta):
	gamma=[0]*len(data)
	for n in range(len(data)):
		x_given_z = [ theta["w"][i]*gaussian(theta["mu"][i],theta["sigma"][i],data[n]) for i in range(len(theta["mu"]))]
		s=sum(x_given_z)
		gamma[n]=np.array(x_given_z)/s
	return np.asarray(gamma)
def update_theta(data,gamma):
	w=gamma.sum(axis=0)/gamma.sum()
	mu= [(gamma[:,k:k+1]*data).sum(axis=0)/gamma[:,k].sum(axis=0)  for k in range(len(gamma[0])) ]
	sigma= [(gamma[:,k:k+1]*(data-mu[k])).T.dot(data-mu[k]) / gamma[:,k].sum(axis=0) for k in range(len(gamma[0]))]
	return {"mu":mu,"sigma":sigma,"w":w}

def log_likelihood(data,theta):
	result=0.0
	for d in data:
		result+=math.log(sum([  theta["w"][i]*gaussian(theta["mu"][i],theta["sigma"][i],d)  for i  in range(2)]))
	return result	
def Gaussian_Mixture(data,k):
	'''
	theta=[{"mu": ,"sigma"=,"w"=  }]
	gamma=[] n*k
	'''
	values=[]
	theta=init_parameters(data,k)
	print theta
	for i in range(5):
		gamma=compute_gamma(data,theta)
		theta=update_theta(data,gamma)
		values.append(log_likelihood(data,theta))
	plt.plot(range(5), values, '-o')
	plt.show()
	r=gamma.argmax(axis=1)
	colors=["r","g","b"]
	plt.scatter(data[:,0],data[:,1],color=[ colors[i] for i in r] )
	plt.show()
if __name__ == '__main__':
	blob,circle=read_data()

	'''
	Kmeans
	'''
	# normalize()?#
	fig=plt.figure()
	colors=['r','g','b','c','m']
	plot_index=0
	for dataset in [blob,circle]:
		for k in [2,3,5]:
			plot_index+=1
			r=kmeans(dataset,k)
			plot_colors=[colors[i] for i in r]
			ax=fig.add_subplot(2,3,plot_index)
			ax.scatter(dataset[:,0],dataset[:,1],color=plot_colors)
	plt.show()
	
	'''
		kernel K means 
	'''

	r=kernel_kmeans(circle,2)
	plot_colors=[colors[i] for i in r]
	plt.scatter(circle[:,0],circle[:,1],color=plot_colors)
	plt.show()

	'''
		Guassian Mixture Model with EM algorithms
	'''
	Gaussian_Mixture(blob,3)