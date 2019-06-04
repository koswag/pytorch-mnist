import torch
import mnist

from tools import nn, chart

print('start')

train_data = mnist.load('digits')
test_data = mnist.load('digits', train=False)
image, label = mnist.random_img('digits')

model = nn.DigitModel()
classifier = nn.Classifier(model, learning_rate=0.01)
epochs = 5

## Pre-training
predictions = classifier.predict(image)
chart.full(image, label, predictions[0], range(10), title=f'Prediction for {label} (pre-training)')

## Training
state_dict, train_loss, test_loss = classifier.train(train_data, test_data, epochs, to_file='digits')

## Post-training
predictions = classifier.predict(image)
chart.full(image, label, predictions[0], range(10), epochs, train_loss, test_loss)