from automl_workflow.api import Learner
from predictor import MyPredictor
from hp_optimizer import MyHPOptimizer


class MyLearner(Learner):

    def __init__(self):
        hp_optimizer = MyHPOptimizer()
        train_info = {}
        super().__init__(
            hp_optimizer=hp_optimizer,
            train_info=train_info,
        )

    def learn(self, train_set):
        self.backbone_model = self.hp_optimizer.fit(self.train_info)
        predictor = MyPredictor(self.backbone_model)
        return predictor
