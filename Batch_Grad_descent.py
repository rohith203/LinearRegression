import pandas as pd
import numpy as np
import random

class Gradient_descent:
    def __init__(self):
        self.learn_rate = 0.1
        random.seed(5)
        self.W = [random.random()*10 ,random.random()*10 ,random.random()*10 ]
        print(self.W)
        self.X_train =[]
        self.Y_train = []
        self.W_arr = []
        self.cost_arr = []
        self.X1_train = []
        self.X2_train = []
        self.train_df =pd.DataFrame()
        

    def pre_process(self):
        Data = pd.read_csv("./Data.csv",header = None)
        Data = Data.drop([0],axis=1) #remove id column
        self.Y_train = np.array(Data[3]) # Y_train is just series/array

        #Normalizing two columns of X1,X2
        X1_train =np.array(Data[1])
        X2_train =np.array( Data[2])
        X1_mean = np.mean(X1_train) 
        X2_mean = np.mean(X2_train)
        X1_SD = np.std(X1_train)
        X2_SD = np.std(X2_train)
        self.X1_train =(X1_train-X1_mean)/X1_SD
        self.X2_train =(X2_train-X2_mean)/X2_SD

        # creating dataframe for X_train
        self.train_df = pd.DataFrame({1:self.X1_train,2:self.X2_train})
        self.train_df.insert(0,0,1)

        # creating a 2-d array for X_train
        array_1 = np.ones(len(self.X1_train))
        self.X_train = np.column_stack((array_1,self.X1_train,self.X2_train))

    def cost_val(self):
        # using matrix multliplication 
        err_value = np.sum( np.square( np.subtract( np.matmul( self.X_train,self.W ) ,self.Y_train) ) )
        return err_value/(2*(len(self.X1_train)))
    
    def update_W(self):
        # lr*grad
        #  grad[0] = sigma(w0+w1*x1+w2*x2-y)
        # grad[1] = sigma((w0+w1*x1+w2*x2)*x1-y)
        # appending three parts to grad array
        diff = np.subtract( np.matmul( self.X_train,self.W ) ,self.Y_train)
        grad = []
        grad.append(np.sum(diff))
        grad.append(np.sum( np.multiply(diff,self.X1_train) ))
        grad.append(np.sum( np.multiply(diff,self.X2_train) ))
        # print(grad)
        self.W = np.subtract( self.W , np.multiply(grad,[0.000001,0.000001,0.000001]) ) 
        new_W = self.W
        new_cost = self.cost_val()
        self.W_arr.append(new_W)
        self.cost_arr.append(new_cost)
        print(new_W)
        # print("\n")
        print(new_cost)
        # print("\n")
        # return self.Wprint(new_W)

    def train(self):
        # in each iteration cost decreases and w changes
        for i in range(100):
            self.update_W()        

    def cost_stochastic(self,index):
        # print(self.W)
        # print(self.X_train[index])
        # print(np.dot(self.W,self.X_train[index]))
        # print(self.Y_train[index])
        cost = ( np.dot(self.W,self.X_train[index])-(self.Y_train[index]) )**2
        return (cost/2)
    
    def update_stochastic(self,index):
        diff = np.dot(self.W,self.X_train[index])-self.Y_train[index]
        grad = []
        grad.append(diff)
        grad.append(diff*self.X_train[index][1])
        grad.append(diff*self.X_train[index][2])
        self.W = np.subtract( self.W , np.multiply(grad,[0.000001,0.000001,0.000001]) ) 
        new_W = self.W
        new_cost = self.cost_stochastic(index)
        self.W_arr.append(new_W)
        self.cost_arr.append(new_cost)
        # print(new_W)
        # print("\n")
        # print(new_cost)
        # print("\n")
        return new_cost
    
    def train_stochastic(self):
        print(len(self.X1_train))
        for i in range(1):
            for j in range(len(self.X1_train)):
                new_cost = self.update_stochastic(j)
                if((j%100000)== 0):
                    print(new_cost)
                    print(self.W)

if __name__=="__main__":

    Batch_grad = Gradient_descent()
    Batch_grad.pre_process()
    Batch_grad.train()
    print("---------")
    Batch_grad.W = [random.random()*10 ,random.random()*10 ,random.random()*10 ]
    Batch_grad.train_stochastic()

# 22.18540477  2.79980012 -3.54909791
# 13.75318238  4.06626798  4.53209034

# 434874
# 265.8395282381314
# [9.4245259  7.39897171 9.22322265]
# 0.38770149779714336
# [10.65604562  6.41966115  7.91690014]
# 375.1980304377401
# [11.85655621  5.47234395  6.59658091]
# 25.235684994389615
# [12.82882109  4.71381365  5.48929903]
# 1.5156124871766257
# [13.75318238  4.06626798  4.53209034]
# 54.43234417789088
# [13.90410593  3.79236234  4.20100673]
# 0.7999257531311169
# [14.72400847  3.37611017  3.5337419 ]
# 186.0860607609685
# [15.50478976  2.96227118  2.78381896]
# 78.30350938678473
# [16.12547756  2.6173735   2.17863233]
# 26.44206869743409
# [16.7346505   2.33727557  1.63529572]