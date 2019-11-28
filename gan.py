#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 01:19:48 2019

@author: adarsh
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input, Add
from keras.models import Model, Sequential
from keras.layers import Reshape, Dense, Dropout, Flatten,Convolution2D, UpSampling2D, LeakyReLU, BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('th')


rand_dim = 100

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train.reshape(60000, 784)


adam = Adam(lr=0.0002, beta_1=0.5)

""" Generator Model (Residual Network)"""
gen_input= Input((rand_dim,))
g_layer1=Dense(256, kernel_initializer=initializers.RandomNormal(stddev=0.02))(gen_input)
g_layer1=LeakyReLU(0.2)(g_layer1)
g_layer2=Dense(512)(g_layer1)
g_layer2=LeakyReLU(0.2)(g_layer2)
g_layer3=Dense(1024)(g_layer2)
g_residual=Dense(1024)(gen_input)
g_layer3=Add()([g_layer3, g_residual])
g_layer3=LeakyReLU(0.2)(g_layer3)
Image_layer=Dense(784, activation='tanh')(g_layer3)
generator=Model(inputs=gen_input, outputs=Image_layer)
generator.compile(loss='binary_crossentropy', optimizer=adam)



""" Discriminator Model (Residual Network)""" 
dis_input=Input(shape=(784,))
d_layer1=Dense(1024, kernel_initializer=initializers.RandomNormal(stddev=0.02))(dis_input)
d_layer1=LeakyReLU((0.3))(d_layer1)
d_layer1=Dropout(0.3)(d_layer1)
d_layer2=Dense(512)(d_layer1)
d_layer2=LeakyReLU((0.3))(d_layer2)
d_layer2=Dropout(0.3)(d_layer2)
d_layer3=Dense(256)(d_layer2)
residual=Dense(256)(dis_input)
d_layer3=Add()([d_layer3,residual])
d_layer3=LeakyReLU((0.3))(d_layer3)
d_layer3=Dropout(0.3)(d_layer3)
Final_layer=Dense(1, activation='sigmoid')(d_layer3)
discriminator=Model(inputs=dis_input, outputs=Final_layer)
discriminator.compile(loss='binary_crossentropy', optimizer=adam)




# Combined network
discriminator.trainable = False
ganInput = Input(shape=(rand_dim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, rand_dim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print( 'Epochs:', epochs)
    print ('Batch size:', batchSize)
    print ('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print( '-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(int(batchCount))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, rand_dim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            y_lab = np.zeros(2*batchSize)
            # One-sided label smoothing
            y_lab[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, y_lab)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, rand_dim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 5 == 0:
            plotGeneratedImages(e)
            saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)

if __name__ == '__main__':
    train(200, 128)

