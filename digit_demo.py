import torch
import mnist

from tools import nn, chart

state_dict = torch.load('MNIST_model/digits.pth')

model = nn.DigitModel()
model.load_state_dict(state_dict)

classifier = nn.Classifier(model)

for _ in range(1):
    img, label = mnist.random_img('digits')
    predictions = classifier.predict(img)

    chart.full(img, label, predictions[0], range(10), title=f'Prediction for {label}')