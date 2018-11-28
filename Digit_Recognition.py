import numpy as np 
import cv2
import gzip
import keras as kr
import sklearn.preprocessing as pre
import matplotlib.pyplot as plt

def NeuralNetwork() :
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

    inputs = train_img.reshape(6000, 784)

    encoder = pre.LabelBinarizer()
    encoder.fit(train_lbl)
    outputs = encoder.transform(train_lbl)

    model.fit(inputs, outputs, epoch = 2, batch_size = 100)


    with gzip.open('data/t10k-images-idx3.ubyte.gz' 'rb') as f:
        test_img = f.read()

    with gzip.open('data/t10k-labels-idx3.ubyte.gz' 'rb') as f:
        test_lbl = f.read()


    test_img = ~np.array(list(test_img[16:])).reshape(60000, 28,28).astype(np.uint8) / 255.0
    test_lbl = np.array(list(test_lbl[8:])).astype(np.uint8)

    (encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()

    model.predict(test_img[5:6])

    plt.imshow(test_img[5].reshape(28,28), cmap='gray')
