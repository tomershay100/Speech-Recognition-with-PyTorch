# Tomer Shay, Roei Gida
import os
import sys
from copy import deepcopy

import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch import nn

from gcommand_dataset import GCommandLoader


class BaseModel(nn.Module):
    def __init__(self, lr):
        super(BaseModel, self).__init__()

        self.name = 'Base Model'
        self.lr = lr

        self.train_accuracies = []
        self.train_loss = []
        self.validate_accuracies = []
        self.validate_loss = []


class BestModel(BaseModel):
    def __init__(self, lr):
        super(BestModel, self).__init__(lr)

        self.name = 'Best Model'

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 30),
            nn.LogSoftmax(dim=1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class VGG11Model(BaseModel):
    def __init__(self):
        super(VGG11Model, self).__init__(LEARNING_RATE)

        self.name = 'VGG11'

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7680, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 30),
            nn.LogSoftmax(dim=1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def export_plot():
    global MODEL, DEVICE
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.title(f'{MODEL.name}')
    plt.plot(MODEL.validate_accuracies, label="validate")
    plt.plot(MODEL.train_accuracies, label="train")
    plt.ylabel("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(MODEL.validate_loss, label="validate")
    plt.plot(MODEL.train_loss, label="train")
    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.legend()
    plt.savefig(f'{MODEL.name}{DEVICE}.png')
    print(f'Model\'s plot saved to \'{MODEL.name}{DEVICE}.png\'')


def train(train_set):
    global MODEL, DEVICE

    MODEL.train()
    train_loss = 0
    correct = 0
    for _, (x, y) in tqdm.tqdm(enumerate(train_set), desc='train iterations', total=len(train_set)):
        x.to(DEVICE)
        y.to(DEVICE)

        MODEL.optimizer.zero_grad()
        output = MODEL(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        MODEL.optimizer.step()
        train_loss += float(loss.data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

    MODEL.train_accuracies.append(100 * correct / len(train_set.dataset))
    MODEL.train_loss.append(train_loss / (len(train_set.dataset) / train_set.batch_size))


def validate(validate_set):
    global MODEL, DEVICE
    MODEL.eval()
    validate_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (x, y) in tqdm.tqdm(enumerate(validate_set), desc='validation iterations', total=len(validate_set)):
            x.to(DEVICE)
            y.to(DEVICE)

            output = MODEL(x)
            loss = F.nll_loss(output, y)
            validate_loss += float(loss.data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    MODEL.validate_accuracies.append(100 * correct / len(validate_set.dataset))
    MODEL.validate_loss.append(validate_loss / (len(validate_set.dataset) / validate_set.batch_size))


def test(test_set):
    global MODEL, DEVICE, IDX_TO_CLASS, TEST_FILE_NAMES

    test_predictions = []
    MODEL.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm.tqdm(enumerate(test_set), desc='making predictions', total=len(test_set)):
            x.to(DEVICE)
            output = MODEL(x)
            pred = output.max(1, keepdim=True)[1]
            label = IDX_TO_CLASS[int(pred[0][0])]
            file_name = os.path.split(TEST_FILE_NAMES[i][0])[1]
            test_predictions.append(file_name + "," + label)

    # sort by numbers
    test_predictions.sort(key=lambda z: int(z.split('.')[0]))
    return test_predictions


def check_accuracy(vec_1, vec_2):
    count = 0
    for value_1, value_2 in zip(vec_1, vec_2):
        if value_1 == value_2:
            count += 1
    return 100 * count / len(vec_1)


def running_epochs(train_set, validate_set):
    global MODEL, EPOCHS
    best_acc_model = deepcopy(MODEL)
    best_val_acc = 0

    for i in range(EPOCHS):
        print(f'\n+-------------------- EPOCH #{i + 1} --------------------------+')

        print("")
        train(train_set)
        print("")
        t = PrettyTable(["Train Accuracy", "Train Loss"])
        t.add_row(["{:.2f}".format(MODEL.train_accuracies[-1]), "{:.2f}".format(MODEL.train_loss[-1])])
        print(t)

        print("")
        validate(validate_set)
        print("")
        t = PrettyTable(["Validate Accuracy", "Validate Loss"])
        t.add_row(["{:.2f}".format(MODEL.validate_accuracies[-1]), "{:.2f}".format(MODEL.validate_loss[-1])])
        print(t)
        print("")

        if best_val_acc < MODEL.validate_accuracies[-1]:
            best_val_acc = MODEL.validate_accuracies[-1]
            best_acc_model = deepcopy(MODEL)
            if best_val_acc > 90:
                MODEL.optimizer.param_groups[0]['lr'] = 0.0001
            print("      !!!!!!! MODEL SAVED !!!!!!")

    MODEL = best_acc_model


def load_files(folder='gcommands'):
    global BATCH_SIZE

    train_set = GCommandLoader(folder + "/train")
    validate_set = GCommandLoader(folder + "/valid")
    test_set = GCommandLoader(folder + "/test")

    train_set = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    validate_set = torch.utils.data.DataLoader(validate_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    return train_set, validate_set, test_set


# hyper parameters
if len(sys.argv) == 2 and sys.argv[1] == 'cuda':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

hyper_parameters = PrettyTable(field_names=["Device", "Batch Size", "Number Of Epochs", "Learning Rate"],
                               title='Hyper Parameters')
hyper_parameters.add_row([DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE])
print(hyper_parameters)

# load files
train_loader, validate_loader, test_loader = load_files()

class_to_idx = train_loader.dataset.class_to_idx
IDX_TO_CLASS = dict((value, key) for key, value in class_to_idx.items())
TEST_FILE_NAMES = test_loader.dataset.spects

# MODEL = VGG11Model()
MODEL = BestModel(LEARNING_RATE)

running_epochs(train_loader, validate_loader)
print("\n+--------------------------------------------------------+")
print("learn finished, BEST MODEL saved.\nexporting plot..")
export_plot()

print('\n\nThe Final Model\'s Results:')

tmp = PrettyTable(["Train Accuracy", "Train Loss", "Validate Accuracy", "Validate Loss"], title='BEST MODEL RESULTS')
tmp.add_row(["{:.2f}".format(MODEL.train_accuracies[-1]), "{:.2f}".format(MODEL.train_loss[-1]),
             "{:.2f}".format(MODEL.validate_accuracies[-1]), "{:.2f}".format(MODEL.validate_loss[-1])])
print(tmp)

print('\nExports testing set predictions to "test_y" file..')
test_prediction = test(test_loader)
f = open(f"test_y", "w")
for y_hat in test_prediction:
    f.write(str(y_hat) + "\n")
f.close()
