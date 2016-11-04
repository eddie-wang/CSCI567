from hw_utils import *
import time
X_tr,y_tr,X_te,y_te=loaddata("MiniBooNE_PID.txt")
X_tr, X_te=normalize(X_tr, X_te)
din,dout=len(X_tr[0]),2
print din

'''
	Linear activations 
'''


archs=[[din,dout],[din,50,dout],[din,50,50,dout],[din,50,50,50,dout]]
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=1)

start=time.time()
archs=[[din,50,dout],[din,500,dout],[din,500,300,dout],[din,800,500,300,dout],[din,800,800,500,300,dout]]
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='linear', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=1)
stop=time.time()
print "Linear time " , stop-start 
'''
	Sigmoid activation
'''
start=time.time()
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='sigmoid', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=1e-3, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=1)
stop=time.time()
print "Sigmoid activatio",stop-start
'''
	RELU activation
'''
start=time.time()
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=1)
stop=time.time()
print "RELU activatio",stop-start
'''	
	L2_Regularization
'''
archs=[[din,800,500,300,dout]]
best_l2=testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[1e-7,5e-7,1e-6,5e-6,1e-5], 
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=1)[1]
'''
	Early stopping and L2_regularuzation
'''
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[1e-7,5e-7,1e-6,5e-6,1e-5], 
				num_epoch=30, batch_size=1000, sgd_lr=5e-4, sgd_decays=[0.0], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=True, verbose=1)
'''
	SGD with weight decay
'''
best_decay=testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[5e-7], 
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[1e-5,5e-5,1e-4,3e-4,7e-4,1e-3], sgd_moms=[0.0], 
					sgd_Nesterov=False, EStop=False, verbose=1)[2]
''' 
	Momentum
'''
best_momentum=testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[0.0], 
				num_epoch=50, batch_size=1000, sgd_lr=1e-5, sgd_decays=[best_decay], sgd_moms=[0.99,0.98,0.95,0.9,0.85], 
					sgd_Nesterov=True, EStop=False, verbose=1)[3]
'''
combination
'''
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[best_l2], 
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[best_decay], sgd_moms=[best_momentum], 
					sgd_Nesterov=True, EStop=True, verbose=1)
'''
Grid search with cross-validation 
'''
archs=[[din,50,dout],[din,500,dout],[din,500,300,dout],[din,800,500,300,dout],[din,800,800,500,300,dout]]
testmodels(X_tr, y_tr, X_te, y_te, archs, actfn='relu', last_act='softmax', reg_coeffs=[1e-7,5e-7,1e-6,5e-6,1e-5], 
				num_epoch=100, batch_size=1000, sgd_lr=1e-5, sgd_decays=[1e-5,5e-5,1e-4], sgd_moms=[0.99], 
					sgd_Nesterov=True, EStop=True, verbose=1)
