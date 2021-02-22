# Electroencephalography Classification Second Part (Desktop Application)

This is a master project from Sushant and Vahap, master students at Uni Bremen. This project is tutored by Yarib Nevarez.

The project aims to develop and implement EEG signal classifier for detection of epileptic seizures.

The project software consists of two parts. This project is the second part (**Eeg-Classification**). The first part is to build a neural network by using python language to classify EEG signals.  

The second part implements the neural network which is built in the first part, to perform classification. It is written in C language to be installed in FPGA board. The first part of the project is **Eeg-Classification-Network-Builder** and can be found [here](https://gitlab.com/abdulvahap/eeg-classification-network-builder).

## What to do 

1. Go to [first part](https://gitlab.com/abdulvahap/eeg-classification-network-builder) .
2. Run **NeuralNetworkClass.py** to create a neural network and export network parameters together with test data.
3. Copy and paste exported csv files in [this part](https://gitlab.com/abdulvahap/eeg-classification) to perform classification on test data.
4. Run C code to perform classification on desktop application.