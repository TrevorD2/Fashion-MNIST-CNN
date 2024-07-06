import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, MaxPool2d, Dropout2d, Linear
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Get data and normalize to the range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

#Batch data and allow data to be shuffled
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#CNN Model
class pytorch_CNN(Module):
    #Declare and assign layers in the network
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 3, stride = 2)
        self.pool1 = MaxPool2d((2, 2))
        self.drop1 = Dropout2d()
        self.conv2 = Conv2d(32, 64, 3, stride = 2)
        self.pool2 = MaxPool2d((2, 2))
        self.drop2 = Dropout2d()
        self.l1 = None #Assign this layer in forward method
        self.l2 = Linear(32, 10)

    #Process input into final logits
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        #Flatten the output for the linear layer
        x = torch.flatten(x, 1)
        
        if self.l1 is None:
            # Initialize self.l1 with the computed input size
            self.l1 = Linear(x.size(1), 32)
        
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    
#Instantiate model
model = pytorch_CNN()

#Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

#Train and evaluate the model
for epoch in range(EPOCHS):
    #Set mode to train
    model.train()
    running_loss = 0.0

    #Train model
    for i, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            running_loss = 0.0

    #Set mode to eval
    model.eval()
    correct = 0
    total = 0

    #Evaluate model
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #Print relevant information
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {running_loss*10 / len(train_loader):0.2f}, '  #Average training loss for the epoch
        f'Accuracy: {(correct / total) * 100:0.2f}, '       #Test accuracy for the epoch
        f'Test Loss: {loss_func(outputs, labels):0.2f}, '   #Test loss for the epoch
        f'Test Accuracy: {(correct / total) * 100:0.2f}'    #Test accuracy for the epoch
    )
