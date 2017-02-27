from Perceptron import Perceptron
from Winnow import Winnow
from AdaGrad import AdaGrad
import numpy as np
import matplotlib.pyplot as plt
from gen import gen
"""
#Q1
(y, x) = gen(10, 100, 500, 50000, False)
#print(x.shape)
#print(y.shape)

#Perceptron without margin
perceptron=Perceptron(x.shape[1],0,None)
perceptron.parameter_tuning(x,y)
perceptron.train(x,y)
print("perceptron with margin 1\n","learning rate:",perceptron.learning_rate)

#Perceptron with margin 1
perceptron_m=Perceptron(x.shape[1],1,None)
perceptron_m.parameter_tuning(x,y)
perceptron_m.train(x,y)
print("perceptron with margin 1\n","learning rate:",perceptron_m.learning_rate)

#Winnow without margin
winnow=Winnow(x.shape[1],0,None)
winnow.parameter_tuning(x,y)
winnow.train(x,y)
print("winnow without margin\n","alpha:",winnow.alpha)

#Winnow with margin
winnow_m=Winnow(x.shape[1],2.0,None)
winnow_m.parameter_tuning(x,y)
winnow_m.train(x,y)
print("winnow with margin\n","alpha:",winnow_m.alpha,"\n","margin:",winnow_m.margin)

#AdaGrad
adaGrad=AdaGrad(x.shape[1],None)
adaGrad.parameter_tuning(x,y)
adaGrad.train(x,y)
print("adaGrad\n","learning rate:",adaGrad.learning_rate)

#plotting the # of mistakes -- # of samples
plt.plot(perceptron.N,perceptron.W)
plt.plot(perceptron_m.N,perceptron_m.W)
plt.plot(winnow.N,winnow.W)
plt.plot(winnow_m.N,winnow_m.W)
plt.plot(adaGrad.N,adaGrad.W)
plt.legend(['perceptron', 'perceptron with margin', 'winnow', 'winnow with margin',"adaGrad"], loc='upper left')
plt.show()
"""

"""
#Q2
Y=[]
X=[]
n=[]
W=[]
for i in range(0,5):
    (y, x) = gen(10, 20, 40*(i+1), 50000, False)
    Y.append(y)
    X.append(x)
    n.append((i+1)*40)
perceptron=[]
perceptron_m=[]
winnow=[]
winnow_m=[]
adaGrad=[]
print("Perceptron without margin\n")
W0=[]
for i in range(0,5):
    size=X[i].shape[1]
    perceptron.append(Perceptron(size,0,None))
    perceptron[i].parameter_tuning(X[i],Y[i])
    print("n=",(i+1)*40,"\n","learning rate:",perceptron[i].learning_rate)
    perceptron[i].converge_train(X[i],Y[i])
    W0.append(perceptron[i].converge_W)
W.append(W0)
print("Perceptron with margin 1\n")
W1=[]
for i in range(0,5):
    size=X[i].shape[1]
    perceptron_m.append(Perceptron(size,1,None))
    perceptron_m[i].parameter_tuning(X[i],Y[i])
    print("n=",(i+1)*40,"\n","learning rate:",perceptron_m[i].learning_rate)
    perceptron_m[i].converge_train(X[i],Y[i])
    W1.append(perceptron_m[i].converge_W)
W.append(W1)
print("Winnow without margin\n")
W2=[]
for i in range(0,5):
    size=X[i].shape[1]
    winnow.append(Winnow(size,0,None))
    winnow[i].parameter_tuning(X[i],Y[i])
    print("n=",(i+1)*40,"\n","alpha:",winnow[i].alpha)
    winnow[i].converge_train(X[i],Y[i])
    W2.append(winnow[i].converge_W)
W.append(W2)
print("Winnow with margin\n")
W3=[]
for i in range(0,5):
    size=X[i].shape[1]
    winnow_m.append(Winnow(size,2.0,None))
    winnow_m[i].parameter_tuning(X[i],Y[i])
    print("n=",(i+1)*40,"\n","alpha:",winnow_m[i].alpha,"\n","margin:",winnow_m[i].margin)
    winnow_m[i].converge_train(X[i],Y[i])
    W3.append(winnow_m[i].converge_W)
W.append(W3)
print("AdaGrad\n")
W4=[]
for i in range(0,5):
    size=X[i].shape[1]
    adaGrad.append(AdaGrad(size,None))
    adaGrad[i].parameter_tuning(X[i],Y[i])
    print("n=",(i+1)*40,"\n","learning rate:",adaGrad[i].learning_rate)
    adaGrad[i].converge_train(X[i],Y[i])
    W4.append(adaGrad[i].converge_W)
W.append(W4)

for i in range(0,5):
    plt.plot(n,W[i])
plt.legend(['perceptron', 'perceptron with margin', 'winnow', 'winnow with margin',"adaGrad"], loc='upper left')
plt.show()
"""

"""
#Q3
for i in [100,500,1000]:
    (train_y,train_x)=gen(10,i,1000,50000,True)
    (test_y,test_x)=gen(10,i,1000,10000,False)
    print("m=",i)
    #Perceptron without margin
    perceptron=Perceptron(train_x.shape[1],0,None)
    perceptron.parameter_tuning(train_x,train_y)
    for i in range(0,20):
        perceptron.train(train_x,train_y)
    print("perceptron without margin\n","learning rate:",perceptron.learning_rate)
    print("accuracy:",perceptron.test(test_x,test_y))

    #Perceptron with margin 1
    perceptron_m=Perceptron(train_x.shape[1],1,None)
    perceptron_m.parameter_tuning(train_x,train_y)
    for i in range(0,20):
        perceptron_m.train(train_x,train_y)
    print("perceptron with margin 1\n","learning rate:",perceptron_m.learning_rate)
    print("accuracy:",perceptron_m.test(test_x,test_y))
    
    #Winnow without margin
    winnow=Winnow(train_x.shape[1],0,None)
    winnow.parameter_tuning(train_x,train_y)
    for i in range(0,20):
        winnow.train(train_x,train_y)
    print("winnow without margin\n","alpha:",winnow.alpha)
    print("accuracy:",winnow.test(test_x,test_y))
    
    #Winnow with margin
    winnow_m=Winnow(train_x.shape[1],2.0,None)
    winnow_m.parameter_tuning(train_x,train_y)
    for i in range(0,20):
        winnow_m.train(train_x,train_y)
    print("winnow with margin\n","alpha:",winnow_m.alpha,"\n","margin:",winnow_m.margin)
    print("accuracy:",winnow_m.test(test_x,test_y))
    
    #AdaGrad
    adaGrad=AdaGrad(train_x.shape[1],None)
    adaGrad.parameter_tuning(train_x,train_y)
    for i in range(0,20):
        adaGrad.train(train_x,train_y)
    print("adaGrad\n","learning rate:",adaGrad.learning_rate)
    print("accuracy",adaGrad.test(test_x,test_y))
"""
#Q4
(data_y,data_x)=gen(10,20,40,10000,True)
adaGrad=AdaGrad(data_x.shape[1],None)
Round=[]
W=[]
L=[]
for i in range(0,50):
    Round.append(i+1)
    adaGrad.train(data_x,data_y)
    adaGrad.hingeLoss=0
    W.append((1-adaGrad.test(data_x,data_y))*len(data_x))
    L.append(adaGrad.hingeLoss)
    adaGrad.hingeLoss = 0
    adaGrad.W=[0]
    adaGrad.N=[0]
#print(W,"\n")
#print(L)
plt.figure()
plt.plot(Round,W)
plt.legend(['Mistakes numbers'], loc='upper left')
plt.show()
plt.figure()
plt.plot(Round,L)
plt.legend(['Loss function'], loc='upper left')
plt.show()
