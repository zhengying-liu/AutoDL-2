# from learner import MyLearner
# from hp_optimizer import MyHPOptimizer
from data_ingestor import MyDataIngestor
import logging
import sys

##################################################
######## To be moved to automl-workflow ##########
##################################################
class TrainInfo(dict):
    pass


def get_logger(verbosity_level):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model_dnn.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger
##################################################
##################################################
##################################################


class Model():

    def __init__(self, metadata):
        # `train_info` starts with the metadata and can be updated to store 
        # any intermediate training information
        train_info = TrainInfo()
        train_info['metadata'] = metadata

        # Instantiate an HPOptimizer

        # hp_optimizer = MyHPOptimizer()

        # Instantiate a DataIngestor
        self.data_ingestor = MyDataIngestor(info=train_info)

        # Get learner using the HPOptimizer. 
        # The learner can absorb `training_info` as its own attribute

        # self.learner = MyLearner(
        #     hp_optimizer=hp_optimizer,
        #     train_info=train_info,
        # )

        self.done_training = False

    def train(self, dataset, remaining_time_budget=None):
        train_dataset, ingest_info = self.data_ingestor.ingest(dataset)
        print(train_dataset)
        print(ingest_info)
        sys.exit()
        self.learner.train_info.update(ingest_info)


        # dataset_uw = self.learner.data_ingestor.ingest(dataset, mode='train')    # uw for universal workflow
        # self.predictor = self.learner.learn(dataset_uw)

    def test(self, dataset, remaining_time_budget=None):
        test_dataset, ingest_info = self.data_ingestor.ingest(dataset)
        
        # dataset_uw = self.learner.data_ingestor.ingest(dataset, mode='test')
        predictions = self.predictor.predict(test_dataset)
        self.done_training = True
        return predictions