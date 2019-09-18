import copy

import numpy as np

from keras.optimizers import Adam, RMSprop, SGD, Adadelta
from keras.models import Model

from framework.soft_ordering.enn.loss_functions \
    import unweighted_binary_crossentropy
from framework.soft_ordering.enn.multi_task_evaluation \
    import MultiTaskEvaluation


class EnnSoftOrderMultiTaskEvaluation(MultiTaskEvaluation):
    """
    Implementation of the MultiTaskEvaluation interface for Soft Order models.
    """

    def __init__(self, domain_config):
        self.domain_config = domain_config
        super(EnnSoftOrderMultiTaskEvaluation, self).__init__()

    def compile_model(self, model, global_hyperparameters):
        learning_rate = global_hyperparameters.get('learning_rate', None)
        if learning_rate is None:
            learning_rate = global_hyperparameters.get('lr')

        # Clipnorm is set to 5 to avoid exploding gradients.
        # XXX: It may be better to have clipnorm be tunable.
        clipnorm = global_hyperparameters.get('clipnorm', 5.)

        # Create the optimizer used for training.
        # XXX: Optimizer creation might better happen in a utility function.
        optimizer_name = global_hyperparameters.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            # XXX: For backwards compatibility, not clipping adam norms.
            optimizer = Adam(lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(lr=learning_rate, clipnorm=clipnorm)
        elif optimizer_name == 'adadelta':
            optimizer = Adadelta(lr=learning_rate, clipnorm=clipnorm)
        elif optimizer_name == 'sgd':
            # SGD requires some additional parameters to perform well.
            momentum = global_hyperparameters.get('momentum', 0.9)
            decay = global_hyperparameters.get('lr_decay', 0.0)
            nesterov = global_hyperparameters.get('nesterov', True)
            optimizer = SGD(lr=learning_rate,
                            momentum=momentum,
                            decay=decay,
                            nesterov=nesterov,
                            clipnorm=clipnorm)

        loss_function_name = self.domain_config.get('loss_function',
                                                    'categorical_crossentropy')
        # Use a custom loss function if necessary.
        if loss_function_name == 'unweighted_binary_crossentropy':
            loss_function = unweighted_binary_crossentropy
        else:
            # This assumes the loss function string is supported by Keras.
            loss_function = loss_function_name

        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])

    def train_on_batch(self, inputs, targets, train_model,
                        global_hyperparameters):

        x_in = self.determine_train_data_inputs(train_model, inputs)
        return train_model.train_on_batch(x_in, targets)

    def predict(self, task_idx, num_tasks, inputs, train_model,
                global_hyperparameters, verbose=1):

        test_model = self._create_test_model(train_model, task_idx, num_tasks)
        self.compile_model(test_model, global_hyperparameters)

        x_in = self.determine_test_data_inputs(test_model, inputs)
        return test_model.predict(x_in, verbose=verbose)

    def evaluate(self, task_idx, num_tasks, inputs, targets, train_model,
                    global_hyperparameters, verbose=1):

        test_model = self._create_test_model(train_model, task_idx, num_tasks)
        self.compile_model(test_model, global_hyperparameters)

        x_in = self.determine_test_data_inputs(test_model, inputs)
        return test_model.evaluate(x_in, targets, batch_size=1, verbose=verbose)

    def _create_test_model(self, train_model, test_idx, num_tasks):

        # Create the testing models in terms of the training model
        train_inputs = copy.copy(train_model.inputs)
        train_outputs = copy.copy(train_model.outputs)

        # Check the number of tasks in the train_model
        if len(train_inputs) > num_tasks:
            # Pop the first input off for constant input.
            # What remains is the rest of the inputs.
            # XXX: This assumes each task has a single input terminal.
            train_constant_input = train_inputs.pop(0)
            input_layer = train_inputs[test_idx]
            inputs = [train_constant_input, input_layer]
        else:
            # It seems that when the train_model gets converted to JSON
            # and back, sometimes the constant_input can get optimized out
            # of that representation if it is not connected at all. In this
            # case do not add the constant inputs as an input to begin with.
            input_layer = train_inputs[test_idx]
            inputs = [input_layer]

        output = train_outputs[test_idx]

        test_model = Model(inputs=inputs, outputs=output)

        return test_model

    def determine_train_data_inputs(self, model, inputs):

        num_model_inputs = len(model.inputs)
        num_x_inputs = len(inputs)

        # Check to see if the constant input data has been optimized out
        # in the transfer via JSON.
        if num_model_inputs == num_x_inputs + 1:
            constant_input_data = self._update_constant_input_data(inputs)
            x_in = [constant_input_data] + inputs
        else:
            x_in = inputs

        assert len(x_in) == num_model_inputs
        return x_in

    def determine_test_data_inputs(self, model, inputs):

        num_model_inputs = len(model.inputs)

        # Check to see if the constant input data has been optimized out
        # in the transfer via JSON.
        if num_model_inputs == 2:
            constant_input_data = self._update_constant_input_data(inputs)
            x_in = [constant_input_data, inputs]
        else:
            x_in = [inputs]

        return x_in

    def _update_constant_input_data(self, inputs):

        if isinstance(inputs, list):
            batch_size = inputs[0].shape[0]
        else:
            batch_size = inputs.shape[0]

        constant_input_data = np.zeros(batch_size)

        return constant_input_data
