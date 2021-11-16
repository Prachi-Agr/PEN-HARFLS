import torch
import numpy as np
from datasets import load_data_UCIHAR
from train import train
from torch import nn
# from model import LSTMModel, Bidir_LSTMModel, Res_LSTMModel, Res_Bidir_LSTMModel, init_weights
from model_cuda import PEN, init_weights
from utils import plot, evaluate
import config as cfg
import data_file as df
import sys
from FedAvg import split_FL
import phe
from encryption import train_and_encrypt
import copy
import numpy as np
from test import test
# Data file to load X and y values

X_train_signals_paths = df.X_train_signals_paths
X_test_signals_paths = df.X_test_signals_paths

y_train_path = df.y_train_path
y_test_path = df.y_test_path

# LSTM Neural Network's internal structure

n_hidden = cfg.n_hidden
n_classes = cfg.n_classes
epochs = cfg.n_epochs
learning_rate = cfg.learning_rate
weight_decay = cfg.weight_decay
clip_val = cfg.clip_val
diag = cfg.diag

#Generate public and private key
public_key, private_key = phe.generate_paillier_keypair(n_length=128)

# Training
# check if GPU is available

#train_on_gpu = torch.cuda.is_available()
if (torch.cuda.is_available() ):
    print('Training on GPU')
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')


def main():


    X_train, y_train, X_test, y_test = load_data_UCIHAR()

    # Input Data-UCI_HAR

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each series)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

    # make splits for federate learning setup
    client_1, client_2, client_3 = split_FL(3,X_train,y_train)
    # Some debugging info

    # print("Some useful info to get an insight on dataset's shape and normalisation:")
    # print("(X shape, y shape, every X's mean, every X's standard deviation)")
    # print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))

    for lr in learning_rate: #for experimentation with different learning rates
        arch = cfg.arch
        if arch['name']=='PEN':
            net= PEN()
        else:
            print("Incorrect architecture chosen. Please check architecture given in config.py. Program will exit now! :( ")
            sys.exit()
        net.apply(init_weights)
        print(diag)
        # opt = torch.optim.Adam(net.parameters(), lr=lr)
        opt=torch.optim.SGD(net.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        net = net.float()
        #initalized to maintain aggregate
        params, opt_modified = train(net, client_1[0], client_1[1], X_test, y_test, opt=opt, criterion=criterion, epochs=1, clip_val=clip_val)   
        model = params['best_model']
        # opt = opt_modified
        # for name, param in model.named_parameters():
        #     print('name: ', name)
        #     print(type(param))
        #     print('param.shape: ', param.shape)
        #     print('param.requires_grad: ', param.requires_grad)
        #     print('=====')
        for i in range(0,epochs):
            # k=0
            # model_list=list()
            # for param_tensor in model.parameters():
            #     for val in param_tensor.flatten():
            #         model_list.append(val.detach().numpy().item())
            #         k=k+1
            #         if(k>10):
            #             break
            # print("model params:", model_list)
            
            opt=torch.optim.SGD(model.parameters(), lr)
            #client 1
            client_model1 = copy.deepcopy(model)
            encrypted_client_1 = train_and_encrypt(client_model1, client_1[0], client_1[1], X_test, y_test, torch.optim.SGD(client_model1.parameters(), lr), criterion,  public_key)
            #client 2
            client_model2 = copy.deepcopy(model)
            encrypted_client_2 = train_and_encrypt(client_model2, client_2[0], client_2[1], X_test, y_test, torch.optim.SGD(client_model2.parameters(), lr), criterion,  public_key)
            #client 3
            client_model3 = copy.deepcopy(model)
            encrypted_client_3 = train_and_encrypt(client_model3, client_3[0], client_3[1], X_test, y_test, torch.optim.SGD(client_model3.parameters(), lr), criterion,  public_key)

            # print("client 1:", encrypted_client_1[0:9])
            # print("Shape of client 2:", encrypted_client_2[0:9])
            # print("Shape of client 3:", encrypted_client_3[0:9])

            aggregated_model = np.add(encrypted_client_1, encrypted_client_2)
            aggregated_model = np.add(aggregated_model, encrypted_client_3)
            # print("Aggregate Model shape:", aggregated_model[0:9])

            raw_values = list()
            for val in aggregated_model:
                raw_values.append(private_key.decrypt(val))
            # new = np.array(raw_values).reshape(model.weight.data.shape)/3
            # model.weight.data = new
            # print("Raw values:", raw_values[0:9])
            new = np.array(raw_values)/3
            # print("\t% new array: ",new[0:9])
            # new -> model (iterate)
            j=0
            for param_tensor in model.parameters():
                for val in param_tensor.flatten():
                    val.data.copy_(torch.from_numpy(np.array(new[j])))
                    j=j+1
            
            best_accuracy=0.0
            best_model= None
            # Testing on aggregate model  (decrypted)
            print("\t% Aggregate performance on Test Set: " + \
              str(test(model, X_test, y_test, criterion, best_accuracy, best_model, test_batch=64)))




        # params = train(net, X_train, y_train, X_test, y_test, opt=opt, criterion=criterion, epochs=epochs, clip_val=clip_val)
        # evaluate(params['best_model'], X_test, y_test, criterion)
        # plot(params['epochs'], params['train_loss'], params['test_loss'], 'loss', lr)
        # plot(params['epochs'], params['train_accuracy'], params['test_accuracy'], 'accuracy', lr)

        #plot(params['lr'], params['train_loss'], params['test_loss'], 'loss_lr', lr)


main()