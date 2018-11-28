import numpy as np 
import cv2
import gzip
import keras as kr
import sklearn.preprocessing as pre

def Reshape() :
    #import the image
    img = cv2.imread("5.jpg")

    #Convert the image to grey scale(needs to be done as Mnist images are in greyscale)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Check to see if the image has been converted
    print(type(gray_image))

    #Resize the image into a 26 by 26 array so it is the same format as our NN data
    Correct_size_img = cv2.resize(gray_image,(26,26))
   

    #print(Correct_size_img.shape)


def NeuralNetwork() :
    #create a sequential model of our neural network
    model = kr.models.Sequential()


    model.add = (kr.layers.Dense(units = 512, activation = 'relu', input_dim = 784))
    model.add = (kr.layers.Dense(units = 512, activation = 'relu'))
    model.add = (kr.layers.Dense(units = 10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.build()




    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()
    

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()
    
    test_images = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
    test_labels =  np.array(list(test_lbl[ 8:])).astype(np.uint8)


    (encoder.inverse_transform(model.predict(test_images)) == test_labels).sum()

def readingFiles() :
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_imgs = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_lbl = f.read()
    
    train_imgs = ~np.array(list(train_imgs[16:])).reshape(60000, 28, 28).astype(np.uint8) /255.0
    train_lbl = np.array(list(train_lbl [8:])).astype(np.uint8)


def Menu() :
    print("Please enter the image you want to check")

def main() :

    readingFiles()



if __name__ == '__main__':
    main()

