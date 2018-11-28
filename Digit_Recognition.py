import numpy as np 
import cv2
import gzip
import keras as kr
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os.path

def NeuralNetwork(userImage) :
    model = kr.models.Sequential()

    model.add(kr.layers.Dense(units = 512, activation = 'relu', input_dim = 784))
    model.add(kr.layers.Dense(units = 512, activation = 'relu'))
    model.add(kr.layers.Dense(units=10, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


    with gzip.open('data/train-images-idx3-ubyte.gz') as f:
       train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz') as f:
       train_lbl = f.read()

    with gzip.open('data/t10k-images-idx3.ubyte.gz' 'rb') as f:
        test_img = f.read()

    with gzip.open('data/t10k-labels-idx3.ubyte.gz' 'rb') as f:
        test_lbl = f.read()

    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28,28).astype(np.uint8) / 255.0
    train_lbl = np.array(list(train_lbl[8:])).astype(np.uint8)

    inputs = train_img.reshape(6000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    model.fit(inputs, outputs, epoch = 2, batch_size = 100)
   
    print(encoder.inverse_transform(model.predict(userImage)))


def userImage(userImg):
    #read in the image in greyscale
    image = cv2.imread(userImg, 0)
    print(image.shape)
    #resize the image to 28 by 28 format as this is what our neural network understands
    correct_img = cv2.resize(image,(28,28))
    print(correct_img.shape)
    img = prepareImage(correct_img)
    NeuralNetwork(img)

def prepareImage(correct_image):
    #get all of the pixel values for the data
    imageData = list(correct_image.get_data())
    #next need to normalise the data as our Neural network has also been normalized multiplying by (1/255) should do
    imageData = [(255 - i) * 1/255.0 for i in imageData]

    #finally rebuild the image
    imageData.np.array(list(imageData)).reshape(1, 784)
    #return the image
    return imageData

#main function    
def main():
   userImage('5.jpg')

if __name__ == '__main__':
    main()