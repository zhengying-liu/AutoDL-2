from automl_workflow.api import TrainInfo
from data_ingestor import MyDataIngestor
from hp_optimizer import MyHPOptimizer
from classic_learner import MyClassicLearner
from learner import MyLearner

import logging
import sys
import numpy as np


class Model():

    def __init__(self, metadata):
        # `train_info` starts with the metadata and can be updated to store 
        # any intermediate training information
        self.train_info = TrainInfo()
        self.train_info['metadata'] = metadata

        # Instantiate an HPOptimizer
        hp_optimizer = MyHPOptimizer()

        # Instantiate a DataIngestor
        self.data_ingestor = MyDataIngestor()

        # Get learner using the HPOptimizer. 
        # The learner can absorb `training_info` as its own attribute

        self.learner = MyLearner(
            data_ingestor=self.data_ingestor,
            hp_optimizer=hp_optimizer,
            train_info=self.train_info,
        )

        self.done_training = False

    def train(self, dataset, remaining_time_budget=None):
        train_dataset = self.learner.data_ingestor.ingest(
            dataset, self.train_info, mode='train')
        print(train_dataset)
        # sys.exit()
        # dataset_uw = self.learner.data_ingestor.ingest(dataset, mode='train')    # uw for universal workflow
        # self.predictor = self.learner.learn(train_dataset)

    def test(self, dataset, remaining_time_budget=None):
        test_dataset = self.learner.data_ingestor.ingest(
            dataset, self.train_info, mode='test')
        # dataset_uw = self.learner.data_ingestor.ingest(dataset, mode='test')
        # predictions = self.predictor.predict(test_dataset)
        self.done_training = True
        # return predictions
        print("ha"*50)
        print(test_dataset._x.shape)
        n_examples = 18
        n_classes = len(test_dataset[0][1])
        return np.zeros((n_examples, n_classes))