import torch
from tools import nn, chart
import mnist

state_dict = torch.load('mnist/MNIST_model/fashion.pth')

model = nn.FashionModel()
model.load_state_dict(state_dict)

classifier = nn.Classifier(model)

for _ in range(5):
    img, label = mnist.random_img('fashion')
    predictions = classifier.predict(img)

    chart.full(img, label, predictions[0], mnist.fashion_classes, title=f'Prediction for {mnist.fashion_classes[label]}')
