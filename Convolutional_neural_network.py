# Eric Tran wangermote 180899350
# Maheep Jain maheepjain 203386460
# For CP468 Project
# https://github.com/wangermote/CP468-Image-classification 
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import sys

# load the test data
def load_cifar_testdata():
    # loading the  dataset
    #train images, train labels, test images, test labels
    (train_x, train_y), (test_x, test_y) = datasets.cifar10.load_data()
    # divde by 255 in order to make the data useable later
    train_x = train_x/255
    test_x = test_x/255
    # return all variables
    return train_x, train_y, test_x, test_y


# define cnn model
def cnn_model():
  # make a sequential model
    model = models.Sequential()
    # add the convolution layer, and then add the pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # dropout normalization rate is 20%
    model.add(layers.Dropout(0.2))
    # repeat adding convolution, pooling, and dropout 2 more times
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # flatten 
    model.add(layers.Flatten())
    # add 2 more dense layers on top
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    # print the summary of the model
    model.summary()
    return model

# plotting curves
def save_and_plot(model):

    # plotting entropy on a 3x1 grid, first subplot
    plt.subplot(3,1,1)
    # set title to entropy
    plt.title('Entropy')
    # plotting entropy train vs test (train red test is blue)
    plt.plot(model.history['loss'], color='red', label='train')
    plt.plot(model.history['val_loss'], color='blue', label='test')
    #plot x and y labels on graph
    plt.xlabel('Epoch')
    plt.ylabel('Entropy loss')
    plt.legend(loc='best')

    # plotting accuracy on a 3x1 grid, second subplot
    plt.subplot(3,1,3)
    # set title to accuracy
    plt.title('Accuracy')
    # plotting accuracy train vs test (train red test is blue)
    plt.plot(model.history['accuracy'], color='red', label='train')
    plt.plot(model.history['val_accuracy'], color='blue', label='test')
    #plot x and y labels on graph
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # save the plot to a file
    filename = sys.argv[0].split('/')[-1].split(".")[0]
    #the plot will become a png
    plt.legend(loc='best')
    plt.savefig(filename + '.png')
    #close the plot
    plt.close()
    return 
#run the convolutional neural network
def run_neural_network():
    # load dataset
    train_x, train_y, test_x, test_y = load_cifar_testdata()
    # create the model
    model = cnn_model()
    #compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    #training model
    history = model.fit(train_x, train_y, epochs=24, 
                        validation_data=(test_x, test_y))
    # evaluate the model
    test_loss, test_acc = model.evaluate(train_x, train_y, verbose=4)
    #print for accuracy
    print("accuracy {}".format(test_acc))
    #print for loss
    print("loss  {}".format(test_loss))
    # print accuracy and entropy
    save_and_plot(history)
    return

run_neural_network()