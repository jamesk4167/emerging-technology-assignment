import numpy as np 

import gzip
import keras as kr
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import os.path
from keras.models import model_from_json

def NeuralNetwork() :
    global encoder
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
    #convert to a numpy array and divide by 255 as RGB values are between 255 and 0
    train_img = np.array(list(train_img[16:])).reshape(60000, 28,28).astype(np.uint8) / 255.0
    train_lbl = np.array(list(train_lbl[8:])).astype(np.uint8)

    inputs = train_img.reshape(60000, 784)
    #Binrize labels
    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)
    #print(train_lbl[0], outputs[0])

    model.fit(inputs, outputs, epochs=10, batch_size = 100)
    #save the model to a json file
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
       json_file.write(model_json)
   
    #save the weights 
    model.save_weights("model.h5")
    print("Saved the model to disk")
    #test if the neural network works
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:

        test_img = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()
    
    test_img = np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
    test_lbl = np.array(list(test_lbl[ 8:])).astype(np.uint8)

    #encoder.inverse_transform(model.predict(test_img)) == test_lbl.sum()

    #print(model.predict_classes(userImg))

    
    plt.imshow(test_img[5].reshape(28, 28), cmap='gray')
    Menu()


def TestAgainstUserImage(userImg):
    if(os.path.isfile("model.json")) :
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights into the new model
        loaded_model.load_weights("model.h5")
        print("loaded model from disk")
        model = loaded_model
        #loaded_model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        prediction = model.predict(np.array(userImg, dtype=float))

        print("Predicted: ", prediction)
        print("The number you have entered is :")
        #encoder = pre.LabelBinarizer()
        #print(encoder.inverse_transform(loaded_model.predict_classes(userImg)))
        print(str(loaded_model.predict_classes(userImg)))
    else :
        NeuralNetwork()
    

def userImage(userImg):
    #read in the image
    image = Image.open(userImg)
    #Convert to greyscale
    greyImage = image.convert('L')
    
    
    #resize the image to 28 by 28 format as this is what our neural network understands
    ResizedImage = greyImage.resize((28,28),Image.BICUBIC)
    #print(correct_img.shape)

    preparedImage = prepareImage(ResizedImage)
    TestAgainstUserImage(preparedImage)

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
    #for this program to work I have provided some sample images inside of the folder that can be used eg 5.jpg
    imageDir = input("please enter the path to the image you want to check: ")
    try:
       userImage(imageDir) 
    except FileNotFoundError:
        print("Sorry this file does not exist, try using 5.jpg or 4.jpg which I have provided for you")
        print("=========================================================")
        Menu()

#main function    
def main():
   Menu()

if __name__ == '__main__':
    main()