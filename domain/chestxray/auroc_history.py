
from keras.callbacks import Callback


class AurocHistory(Callback):
    """
    Class that assists with callbacks from Keras
    in the ChestXrayEvaluator.
    """

    def __init__(self, tester, model_evaluation, training_model,
                    domain_config, data_dict):
        super(AurocHistory, self).__init__()
        self.aurocs = None
        self.task_aurocs = None

        self.tester = tester
        self.model_evaluation = model_evaluation
        self.training_model = training_model
        self.domain_config = domain_config
        self.data_dict = data_dict

    def on_train_begin(self, logs=None):
        self.aurocs = []
        self.task_aurocs = []

    def on_epoch_end(self, epoch, logs=None):
        metrics = {}
        overall_auroc, task_auroc, metrics = self.tester.test(
                    self.model_evaluation, self.training_model,
                    metrics, self.domain_config, self.data_dict,
                    val=True)
        self.aurocs.append(overall_auroc)
        self.task_aurocs.append(task_auroc)
