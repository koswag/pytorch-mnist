import torch
import torch.nn.functional as F
import math

class Classifier:
    def __init__(self, model, learning_rate=0.01):
        self.model = model.cuda()
        self.criterion = torch.nn.NLLLoss()

        train_params = self.model.parameters()

        self.optimizer = torch.optim.Adam(train_params, lr=learning_rate)
    
    def train(self, train_data, test_data, epochs, to_file='model') -> (dict, list, list):
        """Train and validate classifier on given data

        Params:
        - train_data : train dataloader
        - test_data : test dataloader
        - epochs : number of training epochs
        - save_dir : directory name to save trained model
        Return:
        - trained model state_dict
        - train loss values
        - test loss values
        """
        min_loss = math.inf
        train_losses = []
        test_losses = []
        for e in range(epochs):
            self.model.train()
            train_loss = 0
            for feats, labels in train_data:
                feats, labels = feats.cuda(), labels.cuda()
                feats = flatten(feats)

                self.optimizer.zero_grad()

                output = self.model(feats)
                loss = self.criterion(output, labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            else:
                test_loss, accuracy = self.test(test_data)

                if test_loss < min_loss:
                    torch.save(self.model.state_dict(), f'MNIST_model/{to_file}.pth')
                    min_loss = test_loss

                train_losses.append(train_loss/len(train_data))
                test_losses.append(test_loss/len(test_data))

                print('Epoch {}: {:.3f}%'.format(e + 1, accuracy*100))
        return self.model.state_dict(), train_losses, test_losses

    def test(self, data):
        """Test classifier's accuracy on test data
        Params:
        - data : test dataloader
        Return:
        - test running loss
        - avarage accuracy
        """
        loss = .0
        accuracy = .0

        with torch.no_grad():
            self.model.eval()
            for feats, labels in data:
                feats, labels = feats.cuda(), labels.cuda()
                feats = flatten(feats)

                logps = self.model(feats)
                ps = torch.exp(logps)

                top_class = top_index(ps)
                equals = top_class == labels.view(*top_class.shape)

                loss += self.criterion(logps, labels)
                accuracy += torch.mean(to_float(equals))
        return loss, accuracy/len(data)

    def predict(self, feats):
        """Predict image class
        Params:
        - image tensor
        Return:
        - class probability distribution tensor
        """
        feats = feats.cuda()
        feats = flatten(feats)

        with torch.no_grad():
            self.model.eval()
            output = self.model(feats)
        return F.softmax(output, dim=1).cpu()

class DigitModel(torch.nn.Module):
    """Example nn model for MNIST Digits"""
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

        self.dropout = torch.nn.Dropout(p=0.2)
    
    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        h = self.dropout(F.relu(self.fc2(h)))

        y = F.log_softmax(self.fc3(h), dim=1)
        return y

class FashionModel(torch.nn.Module):
    """Example nn model for MNIST Fashion"""
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 10)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))

        y = F.log_softmax(self.fc4(h), dim=1)
        return y

def flatten(tensor):
    """Flattens tensor to 1d"""
    return tensor.view(tensor.shape[0], -1)

def top_index(tensor):
    """Index of tensor's top value"""
    _, top_idx = tensor.topk(1, dim=1)
    return top_idx

def to_float(tensor):
    """Convert tensor to FloatTensor"""
    return tensor.type(torch.FloatTensor)