import NN, data_loader, perceptron
import numpy as np
import matplotlib.pyplot as plt
#training_data, test_data = data_loader.load_circle_data()
training_data, test_data = data_loader.load_mnist_data()
"""
#dividing the training data into five folds
five_folds=[]
l=len(training_data)
for i in range(5):
	if i<4:
		five_folds.append(training_data[i*(l/5):(i+1)*(l/5)])
	else:
		five_folds.append(training_data[i*(l/5):l])
dataset_pairs=[]
for i in range(5):
	temp=[]
	for j in range(5):
		if i!=j:
			temp+=five_folds[j]
	dataset_pairs.append([temp,five_folds[i]])
"""

#domain = 'circles'
domain = 'mnist'
batch_size = 10
learning_rate = 0.1
activation_function = 'tanh'
hidden_layer_width = 10
data_dim = len(training_data[0][0])

"""
domain = 'mnist'
batch_size = [10,50,100]
learning_rate = [0.1,0.01]
activation_function = ['relu','tanh']
hidden_layer_width = [10,50]
data_dim = len(training_data[0][0])

# initialize the parameters set
parameters=[]
average_accuracy=[]
for i in batch_size:
	for j in learning_rate:
		for k in activation_function:
			for z in hidden_layer_width:
				parameters.append([domain,i,j,k,z])


for i in range(len(parameters)):
	acc=0
	for j in range(5):
		net=NN.create_NN(parameters[i][0],parameters[i][1],parameters[i][2],parameters[i][3],parameters[i][4])
		net.train(dataset_pairs[j][0])
		acc+=net.evaluate(dataset_pairs[j][1])
	average_accuracy.append(acc/5)
for i in range(len(average_accuracy)):
	print parameters[i],":",average_accuracy[i]
idx=np.argmax(average_accuracy)
print "best:"
print parameters[idx],":",average_accuracy[idx]
"""

acc=[]

net = NN.create_NN(domain, batch_size, learning_rate, activation_function, hidden_layer_width)
NN_curve=net.train_with_learning_curve(training_data)
NN_acc=[]
for tuple in NN_curve:
	NN_acc.append(tuple[1])
acc.append(net.evaluate(test_data))
plt.plot(np.arange(1,101,1),NN_acc,label='NN')
perc = perceptron.Perceptron(data_dim)
perc_curve=perc.train_with_learning_curve(training_data)
perc_acc=[]
for tuple in perc_curve:
	perc_acc.append(tuple[1])
acc.append(perc.evaluate(test_data))
plt.plot(np.arange(1,101,1),np.multiply(perc_acc,100),label='perc')
plt.legend()
plt.show()
print "accuracy:", acc

