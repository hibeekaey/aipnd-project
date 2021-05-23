import argparse
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, default="./flowers")
parser.add_argument('--save_dir', type=str, default="./checkpoint.pth")
parser.add_argument('--arch', type=str, default="resnet")
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', action="store_true", default=False)

args, _ = parser.parse_known_args()


def load_model(arch, hidden_units, dropout, learning_rate, gpu, num_labels=102):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if arch == 'resnet':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError('Unexpected network architecture', arch)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_units, num_labels)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    return model, criterion, optimizer, device


def train_model(image_datasets, save_dir, arch='resnet', hidden_units=512, dropout=0.2, learning_rate=0.001, epochs=5,
                gpu=False):
    steps = 0
    running_loss = 0
    print_every = 5

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    num_labels = len(image_datasets['train'].classes)
    model, criterion, optimizer, device = load_model(arch, hidden_units, dropout, learning_rate, gpu, num_labels)

    trainloader = dataloaders['train']

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                testloader = dataloaders['valid']

                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")

                running_loss = 0

                model.train()

    model.class_to_idx = image_datasets['train'].class_to_idx

    if save_dir:
        torch.save({'arch': arch,
                    'classifier': model.classifier,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx}, save_dir)

    return model


if args.data_dir:
    data_dir = args.data_dir
    _save_dir = args.save_dir
    _arch = args.arch
    _hidden_units = args.hidden_units
    _dropout = args.dropout
    _learning_rate = args.learning_rate
    _epochs = args.epochs
    _gpu = args.gpu

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),

        'valid': transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),

        'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    }

    _image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    train_model(_image_datasets, _save_dir, _arch, _hidden_units, _dropout, _learning_rate, _epochs, _gpu)
