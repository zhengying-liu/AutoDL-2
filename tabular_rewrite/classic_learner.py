from automl_workflow.api import ClassicLearner
from automl_workflow.api import Dataset
from automl_workflow.api import Predictor

class MyClassicLearner(ClassicLearner):

    def learn(self, train_set: Dataset) -> Predictor:
        """Return a Predictor object."""
        raise NotImplementedError