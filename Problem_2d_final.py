import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from numpy import linalg as LA


def sigmoid(w,x,b,rows_x,col_w):
    temp = np.add(np.dot(x,w),np.full((rows_x, col_w), b))  #x.w+b
    out = 1/(1 + np.exp(-temp))
    return out

threshold = 0.0001
NoofFeatures = 2 #This should manually define
# Initializing Values
# Weight_init = np.random.randn(1,NoofFeatures+1)
Weight_init = np.random.uniform(0,1,size = (1,3))
W = Weight_init[0,0:NoofFeatures]
b= Weight_init[0,NoofFeatures]
W=W.reshape(NoofFeatures,1)

# Hyperparameters
lemda = 1
alpha = 0.01
#Reads Input Data file (Assuming input file has ONLY two output classes and NoofFeatures parameter set accordingly)
result = np.array(list(csv.reader(open("KidsHeightData.csv", "rb"), delimiter=","))).astype("float")
X = result[:,0:NoofFeatures]
[rows_x,col] = np.shape(X)
Y = result[:,NoofFeatures].reshape(rows_x,1)
[rows_y,col_y] = np.shape(Y)
[rows2,col_w] = np.shape(W)

combine_weights = np.append(b.reshape(1,1), W, axis=0) #Combined bias weight to weight matrix inorder to compute gradient easily
combine_bias = np.full((rows_x, 1), 1)

W1_calc = X[ : , 0]
W2_calc = X[ : , 1]

noofepoch = 500000
print("Init Weights")
print(W)
print(b)
marix_norm = [1]
combine_norm =[]
for epoch in range(noofepoch):
    if marix_norm < threshold:
        print("Learned Weights : ",combine_weights)
        print("Stopped Epoch :",epoch)
        break
    else:
        sig_return = sigmoid(W,X,b,rows_x,col_w)
        X_transpose_m = combine_bias.transpose()
        minus = Y - sig_return
        final_gradient_w1 = -(np.dot(W1_calc.transpose(),minus))/rows_x
        final_gradient_w2 = -(np.dot(W2_calc.transpose(),minus))/rows_x
        final_gradient_B = -(np.dot(combine_bias.transpose(),minus))/rows_x

        combine_norm = final_gradient_w1*final_gradient_w1 + final_gradient_w2 * final_gradient_w2 + final_gradient_B * final_gradient_B
        marix_norm = np.sqrt(combine_norm)
        W[0,0] = W[0,0] - (alpha*final_gradient_w1)
        W[1,0] = W[1,0] - (alpha*final_gradient_w2)
        b = b - (alpha*final_gradient_B)


print("Combine Norm: ",combine_norm)
print(marix_norm)
print("Learned Weights Final: ",W,b)

sample_data = pd.read_csv('KidsHeightData_visual.csv')
class_0 = sample_data[sample_data.output_cl == 0]
class_1 = sample_data[sample_data.output_cl == 1]
plt.plot(class_0.shoe_size,class_0.average_height,'ro')
plt.plot(class_1.shoe_size,class_1.average_height,'g+')


x = np.linspace(3,6,10).reshape(10,1)
m = np.array(W[1,0]/W[0,0])
bi = b/W[0,0]

y = -m*x-bi


plt.plot(x,y,'-r')
plt.show()
