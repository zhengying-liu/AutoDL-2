from automl_workflow.api import DataIngestor
from automl_workflow.api import TrainInfo
from automl_workflow.utils import get_logger

from torch.utils.data import Dataset

import logging
import tensorflow as tf
import numpy as np
import time




class MyDataIngestor(DataIngestor):

    '''
    args needed form train_info:
        logger                       @ model.py
        Model.step                   @ model.py
        Model.next_element           @ model.py
        Model.num_examples_train     @ model.py
        Model.X                      @ model.py
        Model.Y                      @ model.py
        Model.data_step              @ model.py
    '''
    def __init__(self):
        default_train_info = TrainInfo()
        default_train_info["step"] = 0
        default_train_info["X"] = []
        default_train_info["Y"] = []
        default_train_info["next_element"] = []
        default_train_info["data_step"] = 0
        default_train_info["logger"] = get_logger('INFO')
        self.default_train_info = default_train_info

        # default_train_info = TrainInfo(self.step, self.X, self.Y, self.next_element, self.data_step)


    def ingest(self, dataset, train_info=None, mode='train'):
        if train_info is None:
            train_info = TrainInfo()
            
        train_info.update(self.default_train_info)

        ##################################
        ###### Initilize attributes ######
        ##################################
        for k in self.default_train_info:
            setattr(self, k, train_info[k])

        if mode == 'train':
            if self.step == 0:
                #dataset = dataset.shuffle(buffer_size=10000000).batch(512)  #
                dataset = dataset.batch(512)  #
                iterator = dataset.make_one_shot_iterator()
                self.next_element = iterator.get_next()
                    
            # Count examples on training set
            if not hasattr(self, 'num_examples_train'):
                self.logger.info("Counting number of examples on train set.")
                print ('dataset:',dataset)

                with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                    while True:
                        try:
                            example, labels = sess.run(self.next_element)
                            example = np.squeeze(example)
                            self.X.extend(example)
                            self.Y.extend(labels)
                            self.data_step += 1
                            if self.data_step in [2, 6]:
                                break
                        except tf.errors.OutOfRangeError:
                            self.data_step = -1
                            break
                            
                self.X_train = np.array(self.X)
                self.y_train = np.array(self.Y)

            dataset = MyDataset(self.X_train, self.y_train)
            # return train_dataloader # pyroch dataset
            return dataset  #
        elif mode == 'test':
            start1 = time.time()
            # Count examples on test set
            if not hasattr(self, 'num_examples_test'):
                self.logger.info("Counting number of examples on test set.")
                dataset = dataset.batch(128)
                iterator = dataset.make_one_shot_iterator()
                example, labels = iterator.get_next()
                X = []
                with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                    while True:
                        try:
                            ex, la = sess.run((example, labels))
                            n_classes = la.shape[-1]
                            ex = np.squeeze(ex)
                            X.extend(ex)
                        except tf.errors.OutOfRangeError:
                            break
                self.X_test = np.array(X)
                self.num_examples_test = self.X_test.shape[0]
                self.logger.info("Finished counting. There are {} examples for test set." \
                            .format(self.num_examples_test))
            print ('###test to_numpy time:', time.time() - start1)
            n_examples = len(self.X_test)
            y_test = np.zeros((n_examples, n_classes))
            return MyDataset(self.X_test, y_test)
    
'''
    The __getitem__ method of custom dataset must return 
    the feature and label for a single data sample (numpy array with 
    data type np.float32).
'''
class MyDataset(Dataset):
    def __init__(self, x, y):
        self._x, self._y = x, y
    def __getitem__(self, i):
        return self._x[i], self._y[i]
    def __len__(self):
        return len(self._x)



