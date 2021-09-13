from automl_workflow.api import DataIngestor
import logging
import tensorflow as tf
from torch.utils.data import Dataset
import numpy as np

from .model import TrainInfo
from .model import get_logger

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


    def ingest(self, dataset, train_info):
        ### REWRITE
        # logger = train_info['logger']
        # self.step = train_info['step']
        # self.next_element = train_info['next_element']
        # # self.num_examples_train = train_info.get('num_examples_train', None)
        # self.X = train_info['X']
        # self.Y = train_info['Y']
        # self.data_step = train_info['data_step']
        train_info.update(self.default_train_info)
        for k in self.default_train_info:
            setattr(self, k, train_info[k])

        if self.step == 0:
            #dataset = dataset.shuffle(buffer_size=10000000).batch(512)  #
            dataset = dataset.batch(512)  #
            iterator = dataset.make_one_shot_iterator()
            self.next_element = iterator.get_next()
                
        # Count examples on training set
        if not hasattr(self, 'num_examples_train'):
            self.logger.info("Counting number of examples on train set.")
            print ('dataset:',dataset)

#             X = []
#             Y = []
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
    
'''
    The __getitem__ method of custom dataset must return 
    the feature and label for a single data sample (numpy array with 
    data type np.float32).
'''
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return len(self.X_train)



