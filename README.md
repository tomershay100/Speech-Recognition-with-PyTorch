


# Speech Recognition with PyTorch
CNN implementation in Python with PyTorch, on audio (``.wav``) files (94+ on test).

1. [General](#General)
    - [Background](#background)
    - [Model Structure](#model-structure)
    - [Files Structure](#files-structure)
    - [Running Instructions](#running-instructions)
2. [Dependencies](#dependencies) 
3. [Installation](#installation)
4. [Footnote](#footnote)

## General

#### Contributes
* Shilat Givati
* Tomer Shay

### Background
Implementation of a neural network on the audio files. using ``gcommand_dataset.py`` that converts the ``.wav`` files into a 2D matrix (of ``161 x 101``).

The audio files in this dataset are ``~`` ``1sec`` long, and there are `` 30`` optional commands that can be heard in the files.

### Model Structure
You can see the diagram of the Neural Network:
In short, the model has ``6`` convolutional layers, with ``Batch Normalize``, ``ReLU`` and ``Max Pooling`` after each one. Then a ``Flatten`` layer and ``4`` more ``Fully Connected`` layers. The output of the neural network is ``30``.
#### Hyper Parameters

 - **Dropout:** after the first fully-connected layer ``= 0.1``
 - **Epochs ``= 15``**
 - **Batch Size  ``= 64``**
 - **Optimizer  ``= Adam``**
 - **Learning Rate  ``= 0.0001``**

![](https://github.com/tomershay100/Speech-Recognition-with-PyTorch/blob/main/CNN%20Architecture.svg)

### Files Structure

For the network to run properly, the audio files must be organized within folders as follows:
* The ``gcommands`` folder next to the ``ex5.py`` file
* A ``gcommands/train`` folder with subfolders (with the names of the ``labels``), so that inside each folder are the ``.wav`` files associated with the same label.
* A ``gcommands/validate`` folder contains subfolders (with the names of the `labels`), so that within each folder are the ``.wav` files associated with the same label.
* A ``gcommands/test`` folder contains a subfolder (its name is irrelevant), so it contains the test's ``.wav`` files.

#### About The Output Files
The program code exports a total of 2 files:
* A ``test_y`` file that contains the predictions for the test.
* The ``BestModelcpu.png`` or ``BestModelcuda.png`` file (based on the device on which the code runs), which contains a graph of the accuracy percentage and loss values of the training and the validation depending on the epochs.

### Running Instructions

The program gets one argument, that can be ``cuda``. If it is, then the program will check if ``cuda`` can be used and if so, a run. If no argument is given at all, or an argument is not ``cuda``, the program will run the code on the ``cpu``.

running example:
```
	$ python3 ex5.py cuda
```

Note that for using the dataset given in this repo, you need to download the dataset (about ``1GB``). You can also use ``google colab`` for running this program.
## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* Git
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Tqdm](https://pypi.org/project/tqdm/)
* [PyTorch](https://pytorch.org/get-started/locally/)
* [PrettyTable](https://pypi.org/project/prettytable/)
* [Soundfile](https://pypi.org/project/SoundFile/)
* [Librosa](https://pypi.org/project/Librosa/)


## Installation

1. Open the terminal.
2. Clone the project by:
	```
	$ git clone https://github.com/tomershay100/Speech-Recognition-with-PyTorch.git
	```
3. Run the ```main.py``` file:
```
	$ python3 ex5.py cuda
```

 
## Footnote:
As you can see, there are several additional files. In the files you can see a report in the Hebrew language that describes the code and the model, you can see a graph that describes the success rates in train and validate within epochs, and you can see a diagram of the network structure.
