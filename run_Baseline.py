from __future__ import print_function

import os
import sys
import time
import json
import argparse
import Baseline as Baseline
import h5py
import numpy as np
import keras.backend as K

#from keras.datasets import cifar100
from load_cifar100 import load_cifar100
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

"""
runnung 427s per epoch
"""
def run_cifar100(batch_size,
                nb_epoch,
                depth,
                nb_dense_block,
                nb_filter,
                growth_rate,
                dropout_rate,
                learning_rate,
                weight_decay,
                plot_architecture,
                compression=0.5,
                init_from_epoch=0):
    """ Run CIFAR100 experiments
    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
    :param plot_architecture: bool -- whether to plot network architecture
    """

    ###################
    # Data processing #
    ###################

    # the data, shuffled and split between train and test sets
    #(X_train, y_train), (X_test, y_test) = cifar100.load_data()
    (X_train, y_train), (X_test, y_test) = load_cifar100()
	
    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    if K.image_dim_ordering() == "th":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_dim_ordering() == "th":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_dim_ordering() == "tf":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    print("X_train shape:{}".format(X_train.shape))

    ###################
    # Construct model #
    ###################

    model = Baseline.Baseline(nb_classes,
                              img_dim,
                              depth,
                              nb_dense_block,
                              growth_rate,
                              nb_filter,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay,
                              compression=0.5)
    # Model output
    model.summary()

    # Build optimizer
    #opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = SGD(lr = learning_rate, momentum = 0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    if plot_architecture:
        from keras.utils.visualize_util import plot
        plot(model, to_file='./figures/Baseline_sum_archi.png', show_shapes=True)

    ####################
    # Network training #
    ####################
    print("Training")

    list_train_loss = []
    list_test_loss = []
    list_learning_rate = []
    loglog = [0]
    lr=learning_rate
    if init_from_epoch != 0:
        model_path = 'weights/Baseline_sum-cifar100-40-tf-'+str(init_from_epoch)+'.h5'
        print('loading wights from %s'%model_path)
        model.load_weights(model_path)
    
    print('traing on batch from epoch %d'%init_from_epoch)
    for e in range(init_from_epoch,nb_epoch):

        if e == int(0.5 * nb_epoch):
           K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.75 * nb_epoch):
           K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        split_size = batch_size
        num_splits = X_train.shape[0] / split_size
        arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)

        l_train_loss = []
        start = time.time()

        for batch_nm,batch_idx in enumerate(arr_splits):

            X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

            l_train_loss.append([train_logloss, train_acc])
            sys.stdout.write("\rEpoch{},Batch {}/{}:Training logloss:{:.4f}, training accuracy:{:.4f}%"\
                    .format(e,batch_nm,num_splits,train_logloss,train_acc*100))
        test_logloss, test_acc = model.evaluate(X_test,
                                                Y_test,
                                                verbose=1,
                                                batch_size=64)


        #EarlyStopping

        # loglog.append(np.mean(np.array(l_train_loss), 0)[0])
        # if len(loglog) >= 20 :
        #     if loglog[-1] - loglog[1] >= -0.01:
        #         print("\n\n\nreduce LR\n\n\n")
        #         lr=np.float32(lr / 10.)
        #         print(lr)
        #         K.set_value(model.optimizer.lr, lr)
        #         loglog = [0]
        #     else:
        #         loglog = [0]
        
        # print("\n\nNOTICE{}\n\n".format(loglog))

        
        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_test_loss.append([test_logloss, test_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
        print('\nEpoch %s/%s, training logloss:%4f test_logloss: %4f, test acc: %4f%% Time: %s' \
              % (e + 1, nb_epoch, np.mean(np.array(l_train_loss), 0)[0], test_logloss, test_acc*100, time.time() - start))

        weights_file = 'weights/Baseline-cifar100-40-12-tf-'+str(e)+'.h5'
        if e%5 ==0:
            model.save_weights(weights_file)

        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["optimizer"] = opt.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["test_loss"] = list_test_loss
        d_log["learning_rate"] = list_learning_rate

        json_file = os.path.join('./log/Baseline_log_cifar100.json')
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run CIFAR100 experiment')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=250, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=22,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=3,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')
    parser.add_argument('--compression', type=float, default=0.5,
                        help='')
    parser.add_argument('--init_from_epoch', type=int, default=0,
                        help='')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_cifar100(args.batch_size,
                args.nb_epoch,
                args.depth,
                args.nb_dense_block,
                args.nb_filter,
                args.growth_rate,
                args.dropout_rate,
                args.learning_rate,
                args.weight_decay,
                args.plot_architecture,
                args.compression,
                args.init_from_epoch)
