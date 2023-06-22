# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:32:24 2023

@author: lqluan
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf


#using softmax regression 

def normalize(train_x, test_x):
    """
    train_x: train samples with shape = (num_samples, num_feat)    
    test_x: testing samples with test_x.shape = (num_samples, num_feat)
    """
    # YOUR CODE HERE        
    if train_x.shape[0] < 2 or test_x.shape[0] < 2:
        raise NotImplementedError("Number of samples is at least 2")
    
    train_x_mean = np.mean(train_x, axis = (0,1))
    train_x_sigma = np.sqrt(np.mean((train_x - train_x_mean)**2, axis = (0,1)))
    norm_train_x = (train_x - train_x_mean)/train_x_sigma
    
    
    # val_x_sigma = np.sqrt(np.mean((val_x - train_x_mean)**2, axis = (0,1)))
    # norm_val_x = (val_x - train_x_mean)/train_x_sigma

    
    test_x_sigma = np.sqrt(np.mean((test_x - train_x_mean)**2, axis = (0,1)))
    norm_test_x = (test_x - train_x_mean)/train_x_sigma

    return norm_train_x, norm_test_x

class SoftmaxClassifier(object):
    def __init__(self, w_shape):
        self.w = np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape)

    def softmax(self, x):
        """
        Calculate softmax of a random matrix 
        """
        #max of z along all features
        xmax = np.max(x, axis = 1)
        #convert to shape (n_sample, 1)
        xmax = xmax.reshape(xmax.shape[0],1)
        new_x = x - xmax
        x_exp = np.exp(new_x)
        s_sum = np.sum(x_exp, axis = 1)
        #reshape from (n,) to (n,1)
        s_sum = s_sum.reshape(s_sum.shape[0],1)
        result = x_exp/s_sum
        
        return result

    def feed_forward(self, x):
        """
        x: input data with x.shape = (num_samples, num_feat) where num_feat = image_height*image_width
        w: weights
        """
        # YOUR CODE HERE
        if self.w.shape[0] != x.shape[1]:
            raise NotImplementedError("Number of rows w needs to be equal to number of columns x")

        new_z = np.dot(x,self.w)        
        y_hat = self.softmax(new_z)        
      
        return y_hat

    def compute_loss(self, y, y_hat):
        # YOUR CODE HERE
        if y.shape[0] != y_hat.shape[0]:
            raise NotImplementedError("Number of rows needs to be the same")
        if y.shape[1] != y_hat.shape[1]:
            raise NotImplementedError("Number of columns needs to be the same")

        loss = np.sum(y*-np.log(y_hat), axis = 1)
        loss = np.mean(loss)
        return loss

    def get_grad(self, x, y, y_hat):
        # YOUR CODE HERE
        if x.shape[0] != y.shape[0]:
            raise NotImplementedError("Number of rows in x and y need to be equal")
        w_grad = np.dot(np.transpose(x), y_hat - y)/y.shape[0]
        return w_grad

    def update_weight(self, grad, learning_rate):
        """Update weight in the model at each step  

        Input:
            - w_grad: numpy array with w_grad.shape = (num_feat,1). 
            it is the gradient value of w, corresponding to x, y_hat and y.
            - learning_rate: hyperparameter of learning rate
            affect updating at each step
        Output:
            - self.w: numpy array, self.w.shape = (num_feat,1). 
            the value of weight after training the model 
        """
        self.w = self.w - learning_rate*grad
        return self.w

    def numerical_check(self, x, y, grad):
        i = 3
        j = 0
        eps = 0.000005
        w_test0 = np.copy(self.w)
        w_test1 = np.copy(self.w)
        w_test0[i,j] = w_test0[i,j] - eps
        w_test1[i,j] = w_test1[i,j] + eps

        y_hat0 = np.dot(x, w_test0)
        y_hat0 = self.softmax(y_hat0)
        loss0 = self.compute_loss(y, y_hat0)

        y_hat1 = np.dot(x, w_test1)
        y_hat1 = self.softmax(y_hat1)
        loss1 = self.compute_loss(y, y_hat1)

        numerical_grad = (loss1 - loss0)/(2*eps)
        print(numerical_grad)
        print(grad[i,j])

def is_stop_training(all_val_loss, n=5):
    # YOUR CODE HERE
    if len(all_val_loss) <= 0:
        raise NotImplementedError("Phải có ít nhất một phần tử")
    if len(all_val_loss) <= n:
        is_stopped = False
    else:
        n_count = 0
        for i in range(len(all_val_loss)-1):
            if all_val_loss[i] < all_val_loss[i+1]:
                n_count = n_count + 1                           
            else:
                #reset to 0
                n_count = 0
            #threshold value
            if n_count == n:
                is_stopped = True
                break
            else: 
                is_stopped = False
            
    return is_stopped

def softmax_eval(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    y = np.argmax(y, axis=1)
    
    # YOUR CODE HERE
    class_type = np.unique(y) #type of class available
    n_class = len(class_type) #number of class
    confusion_matrix = np.zeros((n_class,n_class))
    for k in class_type:
        #number of sample belong to class k        
        N_k_index = np.where(y == k)[0]
        N_k = len(N_k_index)
        
        for val in N_k_index:            
            #find label of the index in N_k_index in y_hat 
            col = y_hat[val]
            confusion_matrix[k,col] = confusion_matrix[k,col] + 1
        confusion_matrix[k,:] = confusion_matrix[k,:]/N_k
    
    return confusion_matrix


def create_one_hot(y, num_class):
    # YOUR CODE HERE
    if len(y) < 2:
        raise NotImplementedError("Number of samples is at least 2")
    eye_mat = np.zeros((len(y), num_class))
    for i, y_val in enumerate(y):
        eye_mat[i,y_val] = 1
    return eye_mat

def reshape2D(tensor):
    # YOUR CODE HERE
    n_sample = tensor.shape[0]
    image_height = tensor.shape[1]
    image_width = tensor.shape[2]
    if image_height < 0 or image_width < 0:
        raise NotImplementedError("Non-negative only")
    result = tensor.reshape(n_sample, image_height*image_width)
    
    return result

def add_one(tensor):
    # YOUR CODE HERE
    n_sample = tensor.shape[0]
    if n_sample < 1:
        raise NotImplementedError("Số mẫu ít nhất là 2")    
    result = np.concatenate((tensor, np.ones((n_sample,1))), axis = 1)
    return result


# Parameters
num_epoch = 10000 
learning_rate = 0.01

epochs_to_draw = 100 
np.random.seed(2020)
num_classes = 10

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images[0:2500]
train_labels = train_labels[0:2500]
test_images = test_images[0:2500]
test_labels = test_labels[0:2500]


imgplot = plt.imshow(train_images[0])
plt.show()

#reshape
train_images.shape
train_images = reshape2D(train_images)
test_images = reshape2D(test_images)

#change to one-hot vector
train_y = create_one_hot(train_labels, num_classes)
test_y = create_one_hot(test_labels, num_classes)
train_y.shape
#normalize data
train_images, test_images = normalize(train_images, test_images)

#add 1 to the end of feature
train_images = add_one(train_images)
test_images = add_one(test_images)

#create Classifier
num_feature = train_images.shape[1]
mnist_classifier = SoftmaxClassifier((num_feature, num_classes))
momentum = np.zeros_like(mnist_classifier.w)

all_train_loss = []

for e in range(num_epoch):    
    # predicted output
    train_labels_hat = mnist_classifier.feed_forward(train_images)

    # compute loss
    train_loss = mnist_classifier.compute_loss(train_y, train_labels_hat)

    # Tính giá trị gradient cho trọng số
    grad = mnist_classifier.get_grad(train_images, train_y, train_labels_hat)

    # Cập nhật trọng số weight
    mnist_classifier.update_weight(grad, learning_rate)

    all_train_loss.append(train_loss)    

    # if (e % epochs_to_draw == epochs_to_draw-1):
    #     from IPython.display import clear_output
    #     clear_output(wait=True)
    #     plot_softmax_loss(all_train_loss)
    #     plt.show()
    #     plt.pause(0.1)
    #     print("Epoch %d: train loss: %.5f || val loss: %.5f" % (e+1, train_loss))

test_labels_hat = mnist_classifier.feed_forward(test_images)
np.set_printoptions(precision=2)
confusion_mat = softmax_eval(test_labels_hat, test_y)

print('Confusion matrix:')
print(confusion_mat)
print('Diagonal values:')
print(confusion_mat.flatten()[0::11])

