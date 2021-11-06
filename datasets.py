'''
Datasets Used:
WISDM, UCI_HAR2012, PAMAP2, OPPORTUNITY
'''
import numpy as np
import pandas as pd
import os
'''
UCI_HAR 2012
'''
def load_data_UCIHAR():
    INPUT_COLUMNS = ["body_acc_x_","body_acc_y_","body_acc_z_","body_gyro_x_","body_gyro_y_","body_gyro_z_","total_acc_x_","total_acc_y_",
                    "total_acc_z_"]
    LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

    DATA_DIR='UCI HAR Dataset/UCI HAR Dataset/'
    TRAIN='train/Inertial Signals/'
    TEST='test/Inertial Signals/'
    X_TRAIN_PATHS=[DATA_DIR+TRAIN+col+'train.txt' for col in INPUT_COLUMNS]
    X_TEST_PATHS=[DATA_DIR+TEST+col+'test.txt' for col in INPUT_COLUMNS]
    #appending all 9 signals
    X_Train = []
    for path in X_TRAIN_PATHS:
        file = open(path, 'r')
        X_Train.append([np.array(s, dtype=np.float32) for s in [row.replace('  ', ' ').strip().split(' ') for row in file]])
        file.close()

    X_Train=np.transpose(np.array(X_Train), (1, 0,2))

    X_Test = []

    for path in X_TEST_PATHS:
        file = open(path, 'r')
        X_Test.append([np.array(s, dtype=np.float32) for s in [row.replace('  ', ' ').strip().split(' ') for row in file]])
        file.close()

    X_Test=np.transpose(np.array(X_Test), (1, 0, 2))

    Y_TRAIN_PATH=DATA_DIR+'train/y_train.txt'
    Y_TEST_PATH=DATA_DIR+'test/y_test.txt'
    #labels for training and testing data 
    y_Train=[]
    file = open(Y_TRAIN_PATH, 'r')
    y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
    file.close()
    y_Train= y_ - 1    #shifts class values from [1,6] to [0,5]

    y_Test=[]
    file = open(Y_TEST_PATH, 'r')
    y_ = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.int32)
    file.close()
    y_Test= y_ - 1

    return X_Train,y_Train,X_Test,y_Test




