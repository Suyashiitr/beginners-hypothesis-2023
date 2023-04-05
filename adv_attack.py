import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from zipfile import ZipFile
from tqdm.auto import tqdm
import os
import argparse

import multiprocessing
from multiprocessing import freeze_support


parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
args = parser.parse_args()

def fgsm_attack(image, epsilon, data_grad): 
    data_grad = image.grad.data
    grad_sign = data_grad.sign() #Finding sign of the gradient 
    perturbed_image = image + (epsilon*grad_sign)
    perturbed_image = torch.clamp(perturbed_image,0.2924,0.6878 ) 
    return perturbed_image

def train(net, criterion, optimizer, dataloader, alpha, num_epochs=20):
    with tqdm(range(num_epochs), desc='training') as training_bar:
        for epoch in training_bar:
            example_loss = 0.0
            epoch_bar = tqdm(enumerate(dataloader), desc=f'epoch {epoch + 1}')
            for i, data in epoch_bar:
                inputs, labels = data
                perturbed_labels = np.full(len(perturbed_inputs),'not_cat')
                inputs.requires_grad = True
                data_grad = inputs.grad.data
                perturbed_inputs = fgsm_attack(inputs, alpha, data_grad)
                print('alpha : ' , alpha)
                train_inputs = torch.cat((inputs, perturbed_inputs), 0) #concatenating original images with perturbed images
                train_labels = torch.cat((labels, perturbed_labels), 0)
                optimizer.zero_grad()
                outputs = net(train_inputs)
                outputs.retain_grad()
                train_loss = criterion(outputs, train_labels)
                train_loss.backward()
                optimizer.step()
                example_loss += train_loss.item()
                epoch_bar.set_postfix(
                    avg_example_loss = example_loss/(i+1)
                )
       
            # Evaluate model on clean test set
            correct = 0
            total = 0
            with torch.no_grad():
                for data in dataloader:
                    inputs, labels = data
                    outputs = net(inputs)
                    predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'----> accuracy on the clean test set: {100 * correct / total:.2f}%')

            # Evaluate model on adversarial test set
            correct = 0
            total = 0
            with torch.no_grad():
                for data in dataloader:
                    inputs, labels = data
                    inputs.requires_grad = True
                    perturbed_inputs = fgsm_attack(inputs, alpha, inputs.grad.data)
                    perturbed_outputs = net(perturbed_inputs)
                    predicted = torch.max(perturbed_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'----> accuracy on the adversarial test set: {100 * correct / total:.2f}%')

    print('----> finished training')

    ##############################
    #### testing on train set ####
    ##############################

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = net(inputs)
            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'----> accuracy on the dataset: {100 * correct / total:.2f}%')

    ####################
    #### save model ####
    ####################

    print('----> saving model to adv_attack.pt')

    torch.save(net.state_dict(), 'adv_attack.pt')


def main():
    # added freeze_support() function call
    multiprocessing.freeze_support()

    ############################################################
    #### defining the network, optimizer, and loss function ####
    ############################################################

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 2)


        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    ########################
    #### unzipping data ####
    ########################

    print('----> unzipping train_data into unzipped_data')

    with ZipFile(args.train_data, 'r') as f:
        f.extractall('unzipped_data')

    ################################
    #### preparing data loaders ####
    ################################

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4901, 0.4617, 0.4061], [0.1977, 0.1956, 0.1947])
    ])

    data = torchvision.datasets.ImageFolder(
        root='unzipped_data',
        transform=transform
    )

    classes = data.classes
    dataloader = torch.utils.data.DataLoader(data, batch_size=32 , shuffle=True, num_workers=1)

#############################################
#### loading pretrained model, if exists ####
#############################################

    # if os.path.exists('adv_attack.pt'):
    #     print('----> loaded pretrained model')
    #     net.load_state_dict(torch.load('adv_attack.pt'))

###########################
#### starting training ####
###########################

    num_epochs = 20

    print(f'----> starting training for {num_epochs} epochs')
    
    
    with tqdm(range(num_epochs), desc='training') as training_bar:
        for epoch in training_bar:
            running_loss = 0.0
            epoch_bar = tqdm(enumerate(dataloader), desc=f'epoch {epoch + 1}')
            for i, data in epoch_bar:
               inputs, labels = data
               inputs.requires_grad = True
            #    outputs = net(inputs)

            #    optimizer.zero_grad()
            #    loss = criterion(outputs, labels)
            #    loss.backward()
            #    optimizer.step()
               epsilon = 0.01
               
               out = net(inputs)
               optimizer.zero_grad()
               loss = criterion(out,labels)
               loss.backward()
               optimizer.step()
               data_grad = inputs.grad.data
               perturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)
               train_inputs = torch.cat((inputs, perturbed_inputs), 0) #concatenating original images with perturbed images
               train_labels = torch.cat((labels, labels), 0)
               optimizer.zero_grad()
               outputs = net(train_inputs)
               outputs.retain_grad()
               train_loss = criterion(outputs, train_labels)
               train_loss.backward()
               optimizer.step()
               

               running_loss += train_loss.item()
               epoch_bar.set_postfix(average_running_loss=running_loss / (i + 1))


    print('----> finished training')
    torch.save(net.state_dict(), 'submission21411035.pt')
##############################
#### testing on train set

if __name__ == '__main__':
    main()
