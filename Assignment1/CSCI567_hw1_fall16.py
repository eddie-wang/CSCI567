from NBC import NBC
from KNN import KNN
import sys
if __name__ == '__main__':
	if len(sys.argv)==3:
		training_data_file=sys.argv[1]
		testing_data_file=sys.argv[2]
	else:
		training_data_file="train.txt"
		testing_data_file="test.txt"
	nbc=NBC(training_data_file,testing_data_file)
	print "For Naive Bayes classifier:"
	print "Training accuracy is", nbc.get_train_accuracy(), " ,Testing accuracy is " ,nbc.test()


	print "\n"
	print "For KNN classifier:"
	knn=KNN(training_data_file,testing_data_file)
	k=[1,3,5,7]
	for _k in k:
		knn.setK(_k)
		knn.useL1Distance()
		print  "When k = " , _k , "using L1 the training accuracy is",knn.train()," ,testing accuracy is " ,knn.test() 
		knn.useL2Distance()
		print  "When k = " , _k , "using L2 the training accuracy is",knn.train()," ,testing accuracy is " ,knn.test() 