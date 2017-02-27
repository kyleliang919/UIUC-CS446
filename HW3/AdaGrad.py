import numpy as np
class AdaGrad:
    def __init__(self,size,learning_rate):
        self.weight=np.zeros(size)
        self.bias= 0  
        self.learning_rate= learning_rate if learning_rate != None else 1.5
        self.G = np.zeros(size+1)
        self.W=[0]
        self.N=[0]
        self.converge_W=0
        self.hingeLoss= 0
    def converge_train(self,x,y):
        R=0
        while R<1000:
             for i in range(0,len(x)):
                temp = y[i]*(sum(self.weight*x[i]) + self.bias)
                if temp <= 1:
                    g_t = np.append(-y[i]*x[i],-y[i])
                    self.G = self.G + g_t**2
                    self.update_weight(x[i],y[i])
                    self.update_bias(y[i])
                if temp <= 0:
                    self.converge_W=self.converge_W+1
                    R=0
                else:
                    R=R+1
        pass
    def train(self,x,y):
        for i in range(0,len(x)):
            self.N.append(i+1)
            temp = y[i]*(sum(self.weight*x[i]) + self.bias)
            if temp <= 1:
                g_t = np.append(-y[i]*x[i],-y[i])
                self.G = self.G + g_t**2
                self.update_weight(x[i],y[i])
                self.update_bias(y[i])
            self.hingeLoss= self.hingeLoss + max(0,1-temp)
            if temp <= 0:   
                self.W.append(self.W[i]+1)
            else:
                #print("W",len(self.W))
                #print("i",i)
                self.W.append(self.W[i])
                
        pass
    def divide(self,a,b):
        if a.size!=b.size:
            return None
        for i in range(0,a.size):
            if b[i] != 0:
                a[i]= a[i]/b[i]
        return a 
    def update_weight(self,x,y):
        self.weight = self.weight + self.divide(self.learning_rate*y*x,np.sqrt(self.G[:self.G.size-1]))
        pass
    def update_bias(self,y):
        self.bias = self.bias + self.learning_rate*y/np.sqrt(self.G[self.G.size-1])
        pass
    def test(self,x,y):
        num_c = 0    
        for i in range(0,len(x)):
            temp = y[i]*(sum(self.weight*x[i]) + self.bias)
            self.hingeLoss= self.hingeLoss + max(0,1-temp)
            if temp  > 0:
                num_c = num_c + 1
        return num_c/len(x)
    def parameter_tuning(self,x,y):
        (D1_x,D1_y)=self.subset(x,y,0.1)
        (D2_x,D2_y)=self.subset(x,y,0.1)
        accuracy=[]
        learning_rate=[1.5, 0.25, 0.03, 0.005, 0.001]
        for rate in learning_rate:
            self.learning_rate=rate
            for i in range(0,20):
                self.train(D1_x,D1_y)
                self.W=[0]
                self.N=[0]
            accuracy.append(self.test(D2_x,D2_y))
            self.bias = 0
            self.weight=np.zeros(self.weight.size)
            self.G = np.zeros(self.G.size)
        self.learning_rate = learning_rate[np.argmax(accuracy)]
        
        #print(accuracy)
        pass
    def subset(self,x,y,percentage):
        sub_x = []
        sub_y = []
        for i in range(0,int(len(x)*percentage)):
            idx=np.random.randint(len(x))
            sub_x.append(x[idx])
            sub_y.append(y[idx])
        return np.array(sub_x),np.array(sub_y)