import torch
import mnist

from tools import nn, chart

train_data = mnist.load('fashion')
test_data = mnist.load('fashion', train=False)
image, label = mnist.random_img('fashion')

model = nn.FashionModel()
classifier = nn.Classifier(model, learning_rate=0.01)
epochs = 15

## Pre-training
predictions = classifier.predict(image)
chart.full(image, label, predictions[0], mnist.fashion_classes, title=f'Prediction for {label} (pre-training)')

## Training
state_dict, train_loss, test_loss = classifier.train(train_data, test_data, epochs, to_file='fashion')

## Post-training
predictions = classifier.predict(image)
chart.full(
    image, label, predictions[0], 
    mnist.fashion_classes,
    epochs=epochs, train_loss=train_loss, test_loss=test_loss
)