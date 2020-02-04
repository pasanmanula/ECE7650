# Coded by : Pasan Manula Bandara - UofM
# Date : 31/01/2020
# Deep Learning Assingment 1 - Question 3 - Part A
import numpy as np
import cPickle
from numpy import linalg as LA

# Global Variable
Input_Training_Rows = 10000

def initialize_weights():
    Weight_init = np.random.uniform(0,1,size = (1,1025))
    print("Init : ",np.shape(Weight_init))
    W = Weight_init[0,0:1024]
    b= Weight_init[0,1024]
    W=W.reshape(1024,1)
    return W, b

def sigmoid(w_sig,x_sig,b_sig,rows_x_sig,col_w_sig):
    temp = np.array(np.add(np.dot(x_sig,w_sig),np.full((rows_x_sig, col_w_sig), b_sig)))  #x.w+b
    out = 1/(1 + np.exp(-temp))
    return out

def unpickle(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = cPickle.load(file)

    features = batch['data']
    labels = batch['labels']
    return features, labels

def unpickle_test(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = cPickle.load(file)
    features = batch['data']
    labels = batch['labels']
    return features, labels

def reconstruct_class(lables,priority):
    for check_class in range(Input_Training_Rows):
        if lables[check_class] != priority: #e.g., If the image is an airplane --> class is 1 otherwise 0
            lables[check_class] = 0
        else:
            lables[check_class] = 1 #Airplane
    return lables

def rgb2gray(rgb_img):
    red_channel = rgb_img[:,0:1024]
    green_channel = rgb_img[:,1024:2048]
    blue_channel = rgb_img[:,2048:3072]
    gray_img =np.dot(0.2125,red_channel) + np.dot(0.7154,green_channel) + np.dot(0.0721,blue_channel)
    return gray_img

def run_learn(X,Y,W,b):
    alpha = 0.001
    threshold = 0.0001
    noofepoch = 1000
    marix_norm = [1]
    combine_bias = np.full((1, Input_Training_Rows), 1)
    for epoch in range(noofepoch):
        if marix_norm < threshold:
            break
        else:
            sig_return = sigmoid(W,X,b,Input_Training_Rows,1)
            minus = Y.transpose() - sig_return
            final_gradient = -(np.dot(X.transpose(),minus))/Input_Training_Rows
            final_gradient_B = -(np.dot(combine_bias,minus))/Input_Training_Rows
            norm_calc = np.append(final_gradient,final_gradient_B,axis =0)
            marix_norm = LA.norm(norm_calc, axis=0)
            W = W - (alpha*final_gradient)
            b = b - (alpha*final_gradient_B)
    return W, b

def main():
  file_path = '/home/pasan/Documents/PythonCode/cifar-10-python/cifar-10-batches-py'
  # airplane : 0 automobile : 1 bird : 2 cat : 3 deer : 4 dog : 5 frog : 6 horse : 7 ship : 8 truck : 9
  Input_Test_Rows = 10000
  classes = [0,1,2,3,4,5,6,7,8,9]
  batch_number = [1,2,3,4,5]
  weight_tensor = np.zeros((len(classes),1024,1))
  bias_tensor = np.zeros((len(classes),1,1))
  W,b = initialize_weights()
  for class_id in classes:
      for batch_id in batch_number:
          print("Serving Now : Class ID :"+ str(class_id) + " Batch ID :" + str(batch_id))
          batch_features,batch_class = unpickle(file_path,batch_id)
          reconstruct_class_re = reconstruct_class(batch_class,class_id)
          Gray = rgb2gray(batch_features)
          X = np.divide(Gray, 255) #Image Downscale
          Y = np.matrix(reconstruct_class_re)
          W,b = run_learn(X,Y,W,b)
      weight_tensor[class_id][:][:] = W
      bias_tensor[class_id][:][:] = b
  print("All Tensors DIM: Weight Tensor")
  print(np.shape(weight_tensor))
  print("All Tensors DIM: Bias Tensor")
  print(np.shape(bias_tensor))

  print("***Testing has been started!***")

  #Forward Pass
  for classifier in classes:
      trained_w = weight_tensor[classifier][:][:]
      trained_b = bias_tensor[classifier][:][:]
      batch_features,batch_class = unpickle_test(file_path)
      reconstruct_class_re = reconstruct_class(batch_class,classifier)
      Gray = rgb2gray(batch_features)
      X_input = np.divide(Gray, 255) #Image Downscale
      Y_real = np.matrix(reconstruct_class_re)
      Y_hat = sigmoid(trained_w,X_input,trained_b,Input_Test_Rows,1)
      Y_hat_reconstruct = (Y_hat > 0.5).astype(int)
      difference = Y_real.transpose() - Y_hat_reconstruct
      error = np.count_nonzero(difference)
      diff = Input_Test_Rows - error
      accuracy = np.true_divide(diff, Input_Test_Rows)*100
      print("Classification Accuracy : " + str(accuracy)+"% for classifier ID : "+str(classifier))
  print("***Testing has been finished!***")

if __name__== "__main__":
    main()
