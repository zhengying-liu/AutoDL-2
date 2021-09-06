import numpy as np
import logging
import sys
import tensorflow as tf
import time

from automl_workflow.api import Predictor


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


logger = get_logger('INFO')



class MyPredictor(Predictor):

    def __init__(self, model):
        self.new_start = True
        self.model_num = -2
        self.best_res = []
        self.test_res = []
        self.model = model

    def predict(self, x):
        start1 = time.time()
        # Count examples on test set
        if not hasattr(self, 'num_examples_test'):
            logger.info("Counting number of examples on test set.")
            dataset = dataset.batch(128)
            iterator = dataset.make_one_shot_iterator()
            example, labels = iterator.get_next()
            X = []
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        ex = sess.run(example)
                        ex = np.squeeze(ex)
                        X.extend(ex)
                    except tf.errors.OutOfRangeError:
                        break
            self.X_test = np.array(X)
            # self.X_test = pd.DataFrame(self.X_test)
            # self.X_test.to_csv(r'D:\code\automl\neurips2019_autodl\AutoDL_public_data'+f'test{self.X_train.shape[0]}.csv')
            # print(1/0)
            # print(f'self.X_test.shape: {self.X_test.shape}')
            self.num_examples_test = self.X_test.shape[0]
            logger.info("Finished counting. There are {} examples for test set." \
                        .format(self.num_examples_test))
        print ('###test to_numpy time:', time.time() - start1)
        
        
        test_begin = time.time()
        logger.info("Begin testing...")
        
        if self.model_num < 1:
            self.best_res = self.model.predict(self.X_test)
            
            if self.new_start:
                self.model_num += 1
                self.test_res.append(self.best_res*2)
        else:
            self.model_num += 1
            pred = self.model.predict_proba(self.X_test)
            self.test_res.append(pred)
            
            print ('###update best result...')
            
            print ('###self.test_res', self.test_res)
            
            print ('begin ensemble...', self.best_res)
            #self.best_res = self.ensemble()
            self.best_res = np.mean(self.test_res, axis=0)
            print ('end ensemble...', self.best_res)
            
            #self.best_res = -1
        
        print('###self.best_res:', self.best_res)
        test_end = time.time()
        # Update some variables for time management
        self.test_duration = test_end - test_begin
        logger.info("[+] Successfully made one prediction. {:.2f} sec used. " \
                    .format(self.test_duration) + \
                    "Duration used for test: {:2f}".format(self.test_duration))
        
        
        if self.model_num == len(self.cand_models):
            self.done_training = True
            
        return self.best_res