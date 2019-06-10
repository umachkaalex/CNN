from functions import *

'''
model_params['accuracy_batch'] - here you can define in which size divide train and test sets
                                 when calculating accuracies. Helpful for old GPU cards. 
                                 If set to '0', whole datasets are sent to calc accuracy.

model_params['plot_from_epoch'] - when print final training losses graph, you can plot from the n-epoch 
                                  to see better what happened in last ones                     
'''

np.random.seed(1)
fldr = 'D:/Deep_Learning/MyProjects/CNN_Stocks/img_data/numbers/'
dataset = PrepareData(fldr).run()

model_params = dict({'learning_rate': 0.005,
                     'num_epochs': 20,
                     'minibatch_size': 256,
                     'accuracy_batch': 100,
                     'print_cost': True,
                     'plot_from_epoch': 5,
                     'save_weights': True,
                     })


layers_params = dict({'layer1': dict({'filter': [32, 32, 1, 32],
                                      'strides': [1, 1, 1, 1],
                                      'padding': 'SAME',
                                      'mp_ksize': [1, 4, 4, 1],
                                      'mp_strides': [1, 4, 4, 1],
                                      'mp_padding': 'SAME'}),
                      'layer2': dict({'filter': [1, 1, 32, 64],
                                      'strides': [1, 1, 1, 1],
                                      'padding': 'SAME',
                                      'mp_ksize': [1, 2, 2, 1],
                                      'mp_strides': [1, 2, 2, 1],
                                      'mp_padding': 'SAME'}),
                      })

model_params['layers'] = len(layers_params.keys())
model_params['layers_params'] = layers_params

train_accuracy, test_accuracy, parameters = model(dataset, model_params)

print("Train Accuracy:", np.mean(train_accuracy))
print("Test Accuracy:", np.mean(test_accuracy))
