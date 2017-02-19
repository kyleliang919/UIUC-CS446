import numpy as np
class Winnow:
    def __init__(self,size,margin,alpha):
        self.weight=np.ones(size)
        self.bias= -size 
        self.margin=margin if margin > 0 else 0
        self.alpha= alpha if alpha != None else 1.1
        self.W=[0]
        self.N=[0]
        self.converge_W=0
    def converge_train(self,x,y):
        R=0
        while R<1000:
            if self.margin == 0:
                for i in range(0,len(x)):
                    temp = y[i]*(sum(self.weight*x[i]) + self.bias)
                    if temp <= self.margin:
                        self.update_weight(x[i],y[i])
                    if temp <= 0:
                        self.converge_W = self.converge_W+1
                        R=0
                    else:
                        R = R + 1 
            else:
                for i in range(0,len(x)):
                    temp = y[i]*(sum(self.weight*x[i]) + self.bias)
                    if temp < self.margin:
                        self.update_weight(x[i],y[i])
                    if temp <= 0:
                        self.converge_W = self.converge_W+1
                        R=0
                    else:
                        R = R + 1
        pass
    def train(self,x,y):
        if self.margin == 0:
            for i in range(0,len(x)):
                self.N.append(i+1)
                temp = y[i]*(sum(self.weight*x[i]) + self.bias)
                if temp <= self.margin:
                    self.update_weight(x[i],y[i])
                if temp <= 0:
                    self.W.append(self.W[i]+1)
                else:
                    self.W.append(self.W[i])
        else:
            for i in range(0,len(x)):
                self.N.append(i+1)
                temp = y[i]*(sum(self.weight*x[i]) + self.bias)
                if temp < self.margin:
                    self.update_weight(x[i],y[i])
                if temp <= 0:
                    self.W.append(self.W[i]+1)
                else:
                    self.W.append(self.W[i])
        pass
    def update_weight(self,x,y):
        self.weight = self.weight * self.alpha**(y*x)
        pass
    def test(self,x,y):
        num_c = 0
        for i in range(0,len(x)):
            if y[i]*(sum(self.weight*x[i]) + self.bias) > 0:
                num_c = num_c + 1 
        return num_c/len(x)
    def parameter_tuning(self,x,y):
        if self.margin != 0:
            (D1_x,D1_y)=self.subset(x,y,0.1)
            (D2_x,D2_y)=self.subset(x,y,0.1)
            accuracy=[]
            alpha=[1.1, 1.01, 1.005, 1.0005, 1.0001]
            margin = [0.3, 2.0, 0.04, 0.006, 0.001]
            if self.margin == 0:
                for rate in alpha:
                    self.alpha=rate
                    self.train(D1_x,D1_y)
                    self.W=[0]
                    self.N=[0]
                    accuracy.append(self.test(D2_x,D2_y))
                    self.bias = -self.weight.size
                    self.weight=np.ones(self.weight.size)
                self.alpha = alpha[np.argmax(accuracy)]
            else:
                for rate in alpha:
                    self.alpha=rate
                    for ma in margin:
                        self.margin = ma
                        for i in range(0,20):
                            self.train(D1_x,D1_y)
                            self.W=[0]
                            self.N=[0]
                        accuracy.append(self.test(D2_x,D2_y))
                        self.bias = -self.weight.size
                        self.weight=np.ones(self.weight.size)
                self.alpha = alpha[np.argmax(accuracy)//len(alpha)]
                self.margin = margin[np.argmax(accuracy)%len(alpha)]
            #print("accuracy",accuracy)
        pass
    def subset(self,x,y,percentage):
        sub_x = []
        sub_y = []
        for i in range(0,int(len(x)*percentage)):
            idx=np.random.randint(len(x))
            sub_x.append(x[idx])
            sub_y.append(y[idx])
        return np.array(sub_x),np.array(sub_y)