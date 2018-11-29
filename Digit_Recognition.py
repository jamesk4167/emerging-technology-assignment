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
    #create a sequential model with keras
    model = kr.models.Sequential()

    model.add(kr.layers.Dense(units = 512, activation = 'relu', input_dim = 784))
    model.add(kr.layers.Dense(units = 512, activation = 'relu'))
    model.add(kr.layers.Dense(units=10, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


    with gzip.open('data/train-images-idx3-ubyte.gz') as f:
       train_img = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz') as f:
       train_lbl = f.read()

    train_img = ~np.array(list(train_img[16:])).reshape(60000, 28,28).astype(np.uint8) / 255.0
    train_lbl = np.array(list(train_lbl[8:])).astype(np.uint8)

    inputs = train_img.reshape(60000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)
    print("training Neural Network")
    model.fit(inputs, outputs, epoch = 10, batch_size = 100)
   
    print(encoder.inverse_transform(model.predict(userImage)))


def userImage(userImg):
    #read in the image in greyscale
    image = Image.open(userImg)
    greyImage = image.convert('L')
    #imshow
    #print(image.shape)
    #resize the image to 28 by 28 format as this is what our neural network understands
    ResizedImage = greyImage.resize((28,28),Image.BICUBIC)
    #print(correct_img.shape)

    preparedImage = prepareImage(ResizedImage)
    NeuralNetwork(preparedImage)

def prepareImage(img):
    #get all of the pixel values for the data
    imageData = list(img.getdata())
    #next need to normalise the data as our Neural network has also been normalized multiplying by (1/255) should do
    #for i in imageData:
    imageData = [(255 - i) * 1.0 / 255.0 for i in imageData]
    

    #finally rebuild the image
    RebuiltImg = np.array(list(imageData)).reshape(1,784)
    #used this print statement to make sure the image was resized properly
    #print(RebuiltImg)
    ## Return the image data
    return RebuiltImg


def Menu():
    imageDir = input("please enter the path to the image you want to check")
    userImage(imageDir)

#main function    
def main():
   Menu()

if __name__ == '__main__':
    main()