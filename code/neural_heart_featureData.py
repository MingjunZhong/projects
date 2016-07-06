# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:38:33 2016

@author: mzhong
"""
import pandas as pd
import csv
import numpy as np
from pandas import HDFStore
from sklearn.cross_validation import train_test_split
from scipy.io import wavfile
import random
import json
import glob
import scipy.io

import sys
import os
import time
import theano
import theano.tensor as T
import lasagne

### Some global variables
windowlength = 1 # in seconds
frequency = 318 # sampling rate
nosOfTestSamples = 500

### It is going to read the heart signals for analysis
def read_heartwave():
#    direct = ['training-a','training-b','training-c','training-d','training-e']
#    windowlist = []
#    labellist = []
#    for dirname in direct:
#        dir_wavfile = '../data/training/'+dirname+'/'
#        exlfile = pd.read_excel(dir_wavfile+'REFERENCE.xlsx')        
#        for i in range(exlfile.shape[0]):
#            wavfilename = dir_wavfile+str(exlfile.iloc[i].filename)+'.wav'
#            #print(wavfilename)
#            reading = wavfile.read(wavfilename)[1].astype('float32')
#            nosOfSeconds = len(reading)/frequency
#            wavlabel = np.uint8((1+exlfile.iloc[i].label)/2)
#            
#            # now we want to split the signal to windows
#            for j in range(int(nosOfSeconds-windowlength)):
#                window = reading[j*frequency:(j+windowlength)*frequency]
#                windowlist.append(window)
#                labellist.append(wavlabel)
    
    mat = scipy.io.loadmat('../data/tangMatlabData/features318_labels_3427records_filtered40_120Hz.mat')
    data=mat['features_labels']
    windowlist = []
    labellist = []
    for i in range(len(data)):
        datai = data[i][0]
        print(len(datai))
        for j in range(len(datai)):
            fd = datai[j]
            windowlist.append(fd[0][0].flatten().astype('float32'))
            labellist.append(np.uint8((1+fd[0][1][0][0])/2))
            
    # TODO : this looks stupid
    # This is to extract the samples of the two classes
    find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
    elem = [0]
    label0 = find(labellist,elem)[0]
    random.shuffle(label0)
    label0 = np.array(label0)
    windowlist0 = list(np.array(windowlist)[label0])
    labellist0 = list(np.array(labellist)[label0])
    
    elem = [1]
    label1 = find(labellist,elem)[0]
    random.shuffle(label1)
    label1 = np.array(label1)
    windowlist1 = list(np.array(windowlist)[label1])
    labellist1 = list(np.array(labellist)[label1])
    
    # We want to make the number of samples of two classes to be balanced
    nosOfSample = min(len(labellist0),len(labellist1))
    win = windowlist0[0:nosOfSample]
    win.extend(windowlist1[0:nosOfSample])
    lab = labellist0[0:nosOfSample]
    lab.extend(labellist1[0:nosOfSample])
    
    windowlist = np.array(win)
    labellist = np.array(lab)
    
    # shuffle the samples
    indices = np.arange(len(labellist))
    np.random.shuffle(indices)
    windowlist = windowlist[indices]
    labellist = labellist[indices]
    
    # Generate training and test data
    Xtest = windowlist[0:nosOfTestSamples]
    Ytest = labellist[0:nosOfTestSamples]
    Xtrain = windowlist[nosOfTestSamples+1:]
    Ytrain = labellist[nosOfTestSamples+1:]
    return Xtrain, Ytrain, Xtest, Ytest

# We want to read the validation data
def read_valheartwave():
    # directory of validation data
    dir_wavfile = '../data/validation/'
    exlfile = pd.read_excel(dir_wavfile+'REFERENCE.xlsx')
    stride = frequency
    windowlist = []
    data = {}
    for i in range(exlfile.shape[0]):
        wavfilename = dir_wavfile+str(exlfile.iloc[i].filename)+'.wav'
        #print(wavfilename)
        reading = wavfile.read(wavfilename)[1].astype('float32')
        nosOfSeconds = len(reading)/frequency
        wavlabel = np.uint8((1+exlfile.iloc[i].label)/2)
        datai = {}
        datai['label'] = wavlabel
        # now we want to split the signal to windows
        start = 0
        while start + frequency < len(reading):
            window = reading[start:start+frequency]
            windowlist.append(window)
            start = start + stride
        datai['window'] = windowlist
        data[str(exlfile.iloc[i].filename)] = datai
    return data
   
# ##################### Build the neural network model #######################
# This script defines a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.
   
def build_cnn(input_var=None, lengthOfInputVector=None):
    # We'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, lengthOfInputVector),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers
    print(network.output_shape)
    
    #network = lasagne.layers.DropoutLayer(network, p=0.2)
    #print(network.output_shape)
    
    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(1,5),
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
    print(network.output_shape)    
    
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print network.output_shape    
    # Max-pooling layer of factor 2 in both dimensions:
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2))
#    print network.output_shape
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=256, filter_size=(1,3),
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print(network.output_shape)
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2))
#    print network.output_shape#
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print network.output_shape
    # A fully-connected layer of 256 units with 50 dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.3),
#            num_units=lengthOfInputVector*10,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print network.output_shape        
#    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.3),
#            num_units=lengthOfInputVector*8,
#            nonlinearity=lasagne.nonlinearities.softmax)
#    print network.output_shape       
    
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.0),
#            num_units=lengthOfInputVector*10,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print network.output_shape    
#    
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print network.output_shape
    
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.0),
#            num_units=lengthOfInputVector*5,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print(network.output_shape)    
#    
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print(network.output_shape)
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=lengthOfInputVector*2,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(network.output_shape)    
    
    #network = lasagne.layers.DropoutLayer(network, p=0.5)
    #print(network.output_shape)
    
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)
    print(network.output_shape)       
    return network

def build_cnn_mnist(input_var=None, dropout=True):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    if dropout:
        p = 0.5
    else:
        p = 0.0

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, frequency),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.
    print(network.output_shape)                                    
    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=5, filter_size=(1, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
    print(network.output_shape)
    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))
    #print(network.output_shape)
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=5, filter_size=(1, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print(network.output_shape)
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=p),
            num_units=frequency*2,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(network.output_shape)
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=p),
            num_units=frequency,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(network.output_shape)
    
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=p),
#            num_units=frequency,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print(network.output_shape)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#            num_units=2,
#            nonlinearity=lasagne.nonlinearities.softmax)
            
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=p),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)
    print(network.output_shape)
    return network
               
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        #print('start_idx={}'.format(start_idx))
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        #print('excerpt={}'.format(excerpt))
        #X = pd.DataFrame(inputs[excerpt]).values.astype(np.float32)
        #y = pd.DataFrame(targets[excerpt]).values.astype(np.uint8)
        X = inputs[excerpt]
        y = targets[excerpt]
        yield X, y
        
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='cnn_mnist', num_epochs=500):
    # load the dataset
    print("loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()   
    X_train, y_train, X_test, y_test = read_heartwave()
    print(X_train.shape)
    
    # this function is for computing predictioin error
    find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('input')
    target_var = T.ivector('targets')
    lengthOfInputVector = len(X_train[0])
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'cnn':
        network = build_cnn(input_var, lengthOfInputVector=lengthOfInputVector)
    elif model == 'cnn_mnist':
        network = build_cnn_mnist(input_var,dropout=True)
    else:
        print("Unrecognized model type {}".format(model))
        
    # Create a loss expression for traing, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #loss = lasagne.objectives.binary_crossentropy(prediction, target_var.T)     
    loss = loss.mean()
    # loss = T.mean((prediction - target_var)**2)
    # we could add some weight decay as well here, see lasagne.regularization.
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
    #       loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adam(loss, params)        
    # Create a loss expression for validation/testing.  The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
    #                                                   target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                 dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var.T))

    # Compile a function performing a training step on mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    #prediction_for_gene_expres = theano.function([input_var],prediction) 
    
    # for prediction
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    pred_class = T.argmax(test_prediction, axis=1)
    test_fn = theano.function([input_var],pred_class)
        
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    minibatch_size = 64
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):
            inputs, targets = batch
            inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
            train_err += train_fn(inputs, targets)
            train_batches += 1
            
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
            inputs, targets = batch
            inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        # compute the prediction error        
        inputs = X_test
        inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
        pred_class_label = test_fn(inputs)
        # now compute error
        # for 0
        elem = [0]
        label0 = find(y_test,elem)[0]
        pred_true_neg = len(find(np.array(pred_class_label)[label0],elem)[0])
        acc0 = 1.0*pred_true_neg/len(label0)
        
        # for 1
        elem = [1]
        label1 = find(y_test,elem)[0]
        pred_true_pos = len(find(np.array(pred_class_label)[label1],elem)[0])
        acc1 = 1.0*pred_true_pos/len(label1)
        
        acc_test = 0.5*(acc0+acc1)
            
        #print predicted_gene_expres
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        print("  test accuracy:\t\t{:.2f} %".format(acc_test*100))
        
    # store data in HDFStore
#    inputs = pd.DataFrame(X_val).values.astype(np.float32)    
#    inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
#    predicted_gene_expres = prediction_for_gene_expres(inputs)
#    
#    
#    geneStore[key+'/prediction'] = pd.DataFrame(np.array(predicted_gene_expres).flatten())
#    geneStore[key+'/true_expres'] = pd.DataFrame(np.array(y_val).flatten())
#    geneStore[key+'/X'] = pd.DataFrame(X_val)
#    # After training, we compute and print the test error:
#    test_err = 0
#    test_batches = 0
#    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
#        inputs, targets = batch
#        inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
#        err, test_prediction = val_fn(inputs, targets)
#        test_err += err
#        test_batches += 1
#    print("Final results:")
#    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

#    return predicted_gene_expres, y_val
    # Optionally, you could now dump the network weights to a file like this:
    np.savez('heart_feature_model.npz', *lasagne.layers.get_all_param_values(network))
    
    # network for prediction
    print("Building model and compiling functions...")
    if model == 'cnn':
        prednetwork = build_cnn(input_var, lengthOfInputVector=lengthOfInputVector)
    elif model == 'cnn_mnist':
        prednetwork = build_cnn_mnist(input_var, dropout=False)
    else:
        print("Unrecognized model type {}".format(model))
    
    # And load them again later on like this:
    with np.load('heart_feature_model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(prednetwork, param_values)
        
    # Compute the accuracy    
    test_prediction = lasagne.layers.get_output(prednetwork, deterministic=True)
    pred_class = T.argmax(test_prediction, axis=1)
    test_fn = theano.function([input_var],pred_class)
    inputs = X_test
    inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
    pred_class_label = test_fn(inputs)
    
    # compute the prediction error
    # now compute error
    # for 0
    elem = [0]
    label0 = find(y_test,elem)[0]
    pred_true_neg = len(find(np.array(pred_class_label)[label0],elem)[0])
    acc0 = 1.0*pred_true_neg/len(label0)
    
    # for 1
    elem = [1]
    label1 = find(y_test,elem)[0]
    pred_true_pos = len(find(np.array(pred_class_label)[label1],elem)[0])
    acc1 = 1.0*pred_true_pos/len(label1)
    
    acc_test = 0.5*(acc0+acc1)
    #print(pred_class_label)
    print("  test accuracy:\t\t{:.2f} %".format(acc_test*100))

def prediction(model='cnn_mnist'):
    # load the test data for prediction
    test_data = read_valheartwave()
    
    # load the trained model for prediction
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('input')
    #target_var = T.ivector('targets')
    lengthOfInputVector = frequency
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'cnn':
        network = build_cnn(input_var, lengthOfInputVector=lengthOfInputVector)
    elif model == 'cnn_mnist':
        network = build_cnn_mnist(input_var)
    else:
        print("Unrecognized model type {}".format(model))
        
    with np.load('heart_feature_model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    # make the prediction
    output = lasagne.layers.get_output(network)
    prediction = theano.function([input_var], output)
    
    # Now we are ready to predict the signal
    for sample in test_data:
        #print(sample)
        data = test_data[sample]
        results_samp = {}
        results_samp['truelabel'] = int(data['label'])
        pred_label = []
        window = data['window']
        for onewindow in window:
            onewindow = np.reshape(onewindow,(1,1,1,frequency))
            predicted_win = prediction(onewindow)            
            pred_label.append(list(np.array(predicted_win).flatten()).index(np.array(predicted_win).max()))
        results_samp['predlabel'] = pred_label
        #pred_results[sample] = results_samp
        #print(results_samp['truelabel'])
        #print(results_samp['predlabel'][0])
        with open(sample+'.json', 'w') as f:
            json.dump(results_samp, f)

    #return pred_results
def accuracy():
    # list all the filenames with extention .json
    filenamelist = glob.glob("*.json")
    acc = []
    for filename in filenamelist:    
        with open(filename, 'r') as f:
            print(filename)
            jsondata = json.load(f)
            probability = [1.0*(len(jsondata['predlabel'])-sum(jsondata['predlabel']))/len(jsondata['predlabel']),
                           1.0*sum(jsondata['predlabel'])/len(jsondata['predlabel'])]
            predlabel = probability.index(np.array(probability).max())
            truelabel = jsondata['truelabel']
            if predlabel == truelabel:
                acc.append(1)
            else:
                acc.append(0)
    print('prediction accuracy:{}'.format(1.0*sum(acc)/len(acc)))
            
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on gene data using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
#        prediction()
#        accuracy()
        
        #X_train, y_train, X_test, y_test = read_heartwave()