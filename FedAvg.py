def split_FL(num_splits, x_train, y_train):
    split1= (x_train[0:2500], y_train[0:2500])
    split2= (x_train[2500:5000], y_train[2500:5000])
    split3= (x_train[5000:7300], y_train[5000:7300])
    return split1, split2, split3