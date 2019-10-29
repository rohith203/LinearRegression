import pandas as pd
import numpy as np
import random

class Normal_equation:
    def __init__(self):
        self.W = []
        self.X_train =[]
        self.Y_train = []
        self.X1_train = []
        self.X2_train = []


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

        array_1 = np.ones(len(self.X1_train))
        self.X_train = np.column_stack((array_1,self.X1_train,self.X2_train))

    # get w directly instead of getting it sequentially in gradient descent or vectorized linear regrassion
    # W = inverse(X*Xt)* (Xt*y)

    def get_W(self):
        X = self.X_train
        Y = self.Y_train
        self.W = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),Y))
        print(self.W)
        
if __name__ == "__main__":
    grad_normal = Normal_equation()
    grad_normal.pre_process()
    grad_normal.get_W()