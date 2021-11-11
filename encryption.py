'''
Homomorphic Encryption on model weights before sending to server

Procedure Implemented:
Central server sends public key to each client so that they can encrypt their weights.
Aggregate the weights and send to central server.  Central server uses private key to decrypt the weights.

'''
import phe
import copy
from train import train
import config as cfg
import torch
import numpy as np
# note that in production the n_length should be at least 1024
# public_key, private_key = phe.generate_paillier_keypair(n_length=128)

def train_and_encrypt(model, input, target, x_test, y_test, opt, criterion, pubkey):
    new_model_params, opt_mod = train(model , input, target, x_test, y_test, opt=opt, criterion=criterion, epochs=3, clip_val=cfg.clip_val)
    new_model = new_model_params['best_model']
    encrypted_weights = list()
    i=0
    print ("client")
    for param_tensor in new_model.parameters():
        # print(type(param_tensor))
        for val in param_tensor.flatten():
            val=val.detach().numpy().item()
            # if(i<10):
            #     print(val)
            # i=i+1
            encrypted_weights.append(pubkey.encrypt(val))

    # for val in new_model.weight.data[:,0]:
    #     encrypted_weights.append(public_key.encrypt(val))
    # ew = np.array(encrypted_weights).reshape(new_model.parameters().shape)
    ew= np.array(encrypted_weights)

    return ew

# for i in range(3):
#     print("\nStarting Training Round...")
#     print("\tStep 1: send the model to Bob")
#     bob_encrypted_model = train_and_encrypt(copy.deepcopy(model),
#                                             bob[0], bob[1], public_key)

#     print("\n\tStep 2: send the model to Alice")
#     alice_encrypted_model=train_and_encrypt(copy.deepcopy(model),
#                                             alice[0],alice[1],public_key)

#     print("\n\tStep 3: Send the model to Sue")
#     sue_encrypted_model = train_and_encrypt(copy.deepcopy(model),
#                                             sue[0], sue[1], public_key)

#     print("\n\tStep 4: Bob, Alice, and Sue send their")
#     print("\tencrypted models to each other.")
#     aggregated_model = bob_encrypted_model + \
#                        alice_encrypted_model + \
#                        sue_encrypted_model

#     print("\n\tStep 5: only the aggregated model")
#     print("\tis sent back to the model owner who")
#     print("\t can decrypt it.")
#     raw_values = list()
#     for val in sue_encrypted_model.flatten():
#         raw_values.append(private_key.decrypt(val))
#     new = np.array(raw_values).reshape(model.weight.data.shape)/3
#     model.weight.data = new

#     print("\t% Correct on Test Set: " + \
#               str(test(model, test_data, test_target)*100))
# # import syft as sy  # import the Pysyft library

# # # hook PyTorch to add extra functionalities like Federated and Encrypted Learning
# # hook = sy.TorchHook(torch) 

# # # simulation functions
# # from future import connect_to_workers, connect_to_crypto_provider

# # workers = connect_to_workers(n_workers=2)   
# # crypto_provider = connect_to_crypto_provider()

# # '''
# # model = Net()
# # model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)

# # optimizer = optim.SGD(model.parameters(), lr=args.lr)
# # optimizer = optimizer.fix_precision() 

# # '''