import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #change the weights of the neural network over time and chage the accuracy(BACKPROPAGATION)
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor #transform image into numbers
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, num_classes=10):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 32, 3) ## 1 grey layer  network  output32 filters of 3by 3 size
        self.bn1 = nn.BatchNorm2d(32) #output from other layer
        self.conv2 = nn.Conv2d(32, 32, 3) #input,output,win
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2) # max pooling to lower the samplesize by factor of 2
        self.conv4 = nn.Conv2d(32, 64, 3) # input,output,win
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, self.num_classes)#take the in put dim into linear layer and give the probability of which classes it belong

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr) # optimize the paramenter inhereted from nn.module at the backend. and lr

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.get_data()

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, 28, 28)) #batch size of 1, with 1 channel, 4 dimentional array of zeros tesors
        batch_data = self.conv1(batch_data) #feeding the zeros to the network to count the dimen
        #batch_data = self.bn1(batch_data)
        batch_data = self.conv2(batch_data)
        #batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)

        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size())) #number of element in batch size of 1

    def forward(self, batch_data):
        batch_data = T.tensor(batch_data).to(self.device) ##calculating the forword passes Passing batch data from mnest dataset to gpu

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1) #flatening 0th element by -1 which will give us 2 dimentional sqaure array#get input size funtion

        classes = self.fc1(batch_data)

        return classes


    def get_data(self):
        mnist_train_data = MNIST('mnist', train=True,
                                 download=True, transform=ToTensor())
        self.train_data_loader = T.utils.data.DataLoader(mnist_train_data,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=8)
        mnist_test_data = MNIST('mnist', train=False,
                                 download=True, transform=ToTensor())
        self.test_data_loader = T.utils.data.DataLoader(mnist_test_data,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=8) ##object to load the data, shuffle it,add other utility funtion ##object to load the data, shuffle it,add other utility funtion

    def _train(self): #tells the data that you are entering the training mode#explicitng telling it so that it does not update the batch norm layer
        self.train()
        for i in range(self.epochs): ##iterating over the total data(epochs) batch by batch
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input) #perform a feed forward pass to get prediction
                loss = self.loss(prediction, label)
                prediction = F.softmax(prediction, dim=1)
                classes = T.argmax(prediction, dim=1) #argmax value of a target function, finding the class with the largest
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                loss.backward() #backpropagation of loss so the loss is less overtime#vvq
                self.optimizer.step() #accuracy will go up over time
            print('Finish epoch ', i, 'total loss %.3f' % ep_loss,
                    'accuracy %.3f' % np.mean(ep_acc))
            self.loss_history.append(ep_loss)

    def _test(self):
        self.eval()

        ep_loss = 0
        ep_acc = []
        for j, (input, label) in enumerate(self.test_data_loader):
            label = label.to(self.device)
            prediction = self.forward(input)
            loss = self.loss(prediction, label)
            prediction = F.softmax(prediction, dim=1)
            classes = T.argmax(prediction, dim=1)
            wrong = T.where(classes != label,
                            T.tensor([1.]).to(self.device),
                            T.tensor([0.]).to(self.device))
            acc = 1 - T.sum(wrong) / self.batch_size

            ep_acc.append(acc.item())

            ep_loss += loss.item()

        print('total loss %.3f' % ep_loss,
                'accuracy %.3f' % np.mean(ep_acc))

if __name__ == '__main__':
    network = CNN(lr=0.001, batch_size=128, epochs=25)
    network._train()
    plt.plot(network.loss_history)
    plt.show()
    plt.plot(network.acc_history)
    plt.show()
    network._test()