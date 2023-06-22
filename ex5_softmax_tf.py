import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

class SoftmaxRegressionTF(tf.keras.Model):
    def __init__(self, num_class):
        super(SoftmaxRegressionTF, self).__init__()
        """ using tf.keras.layers.dense
         https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

        Input:
            - num_class: number of class in the data
        """
    
        if num_class <= 0:
            raise NotImplementedError("Number of class is at least 1")
        # initialize the weight
        self.dense = tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.RandomNormal(seed=2020))

    def call(self, inputs, training=None, mask=None):
        """ 
        Calculate output of the model
        """
                
        output = self.dense(inputs)
        try:
            output = tf.nn.softmax(output)
        except:  # if softmax op does not exist on the gpu
            with tf.device('/cpu:0'):
                output = tf.nn.softmax(output) 

        return output
    

def normalize(train_x, test_x, val_x):
    """
    train_x: train samples with shape = (num_samples, num_feat)    
    test_x: testing samples with test_x.shape = (num_samples, num_feat)
    val_x: validating samples 
    """
    # YOUR CODE HERE        
    if train_x.shape[0] < 2 or test_x.shape[0] < 2:
        raise NotImplementedError("Number of samples is at least 2")
    
    train_x_mean = np.mean(train_x, axis = (0,1))
    train_x_sigma = np.sqrt(np.mean((train_x - train_x_mean)**2, axis = (0,1)))
    norm_train_x = (train_x - train_x_mean)/train_x_sigma
    
    
    val_x_sigma = np.sqrt(np.mean((val_x - train_x_mean)**2, axis = (0,1)))
    norm_val_x = (val_x - train_x_mean)/train_x_sigma

    
    test_x_sigma = np.sqrt(np.mean((test_x - train_x_mean)**2, axis = (0,1)))
    norm_test_x = (test_x - train_x_mean)/train_x_sigma

    return norm_train_x, norm_test_x, norm_val_x

def reshape2D(tensor):
    # YOUR CODE HERE
    n_sample = tensor.shape[0]
    image_height = tensor.shape[1]
    image_width = tensor.shape[2]
    if image_height < 0 or image_width < 0:
        raise NotImplementedError("Non-negative only")
    result = tensor.reshape(n_sample, image_height*image_width)
    
    return result

def create_one_hot(y, num_class):
    # YOUR CODE HERE
    if len(y) < 2:
        raise NotImplementedError("Number of samples is at least 2")
    eye_mat = np.zeros((len(y), num_class))
    for i, y_val in enumerate(y):
        eye_mat[i,y_val] = 1
    return eye_mat

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


num_epoch = 100
learning_rate = 0.001

batch_size = 32
num_classes = 10
np.random.seed(2020)

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

val_images =  test_images[3000:3500]
val_labels =  test_labels[3000:3500]


train_images = train_images[0:2500]
train_labels = train_labels[0:2500]
test_images = test_images[0:2500]
test_labels = test_labels[0:2500]


#reshape
train_images.shape
train_images = reshape2D(train_images)
test_images = reshape2D(test_images)
val_images = reshape2D(val_images)

#change to one-hot vector
train_y = create_one_hot(train_labels, num_classes)
test_y = create_one_hot(test_labels, num_classes)
train_y.shape
val_y = create_one_hot(val_labels, num_classes)
#normalize data
train_images, test_images, val_images = normalize(train_images, test_images, val_images)


# using cpu for gup
device = '/cpu:0' if len(tf.config.experimental.list_physical_devices('GPU')) == 0 else '/gpu:0'

with tf.device(device):
    #build model and optimizer
    model = SoftmaxRegressionTF(num_classes)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # training model
    model.fit(train_images, train_y, batch_size=batch_size, epochs=num_epoch, validation_data=(val_images, val_y), verbose=2)

    # validate model prediction on test set
    scores = model.evaluate(test_images, test_y, 32, verbose=2)
    y_hat = model.predict(test_images)
    confusion_mat = softmax_eval(y_hat, test_y)

    print('Confusion matrix:')
    print(confusion_mat)
    print('Diagonal values:')
    print(confusion_mat.flatten()[0::11])