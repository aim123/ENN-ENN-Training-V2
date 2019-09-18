class MultiTaskEvaluation():
    """
    Interface definition for MultiTask Evaluation.
    """

    def __init__(self):
        """
        SoftOrderEvaluator and its subclasses are specifically stateless
        so their methods can be used with any model/candidate pass in
        without fear of side effects.
        """

    def compile_model(self, model, global_hyperparameters):
        """
        :param model: a Model to compile
        :param global_hyperparameters: a dictionary of global hyperparameters
                    ("global" to the model)
        :return: a compiled version of the given model
        """
        raise NotImplementedError

    def train_on_batch(self, inputs, targets, train_model, global_hyperparameters):
        """
        :param inputs: inputs to the network
        :param targets: outputs from the network
        :param train_model: the model to be trained
        :param global_hyperparameters: the global hyperparameters ("global" to the model)
        :return: ??? Whatever Keras Model.train_on_batch() returns
        """
        raise NotImplementedError

    def predict(self, task_idx, num_tasks, inputs, train_model,
                global_hyperparameters, verbose=1):
        """
        :param task_idx: the index of the task whose sub-network we want
                         to test
        :param num_tasks: the total number of tasks in the trained network
        :param inputs: the inputs to the test sub-network for the task
        :param train_model: the training model from which a test model for
                            prediction is created
        :param global_hyperparameters: the global hyperparameters ("global" to the model)
        :param verbose: flag for extra output
        :return: ??? whatever Keras Model.predict() returns
        """
        raise NotImplementedError

    def evaluate(self, task_idx, num_tasks, inputs, targets, train_model,
                 global_hyperparameters, verbose=1):
        """
        :param task_idx: the index of the task whose sub-network we want
                         to test
        :param num_tasks: the total number of tasks in the trained network
        :param inputs: the inputs to the test sub-network for the task
        :param targets: outputs from the network
        :param train_model: the training model from which a test model for
                            prediction is created
        :param global_hyperparameters: the global hyperparameters ("global" to the model)
        :param verbose: flag for extra output
        :return: ??? whatever Keras Model.evaluate() returns
        """
        raise NotImplementedError
