import logging
import pickle
from itertools import zip_longest
from os import makedirs
from os.path import dirname, exists, join
from typing import Union, Tuple, Dict, List

from numpy import ndarray, empty, inf
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, ModelCheckpoint
from tensorflow.python.keras.optimizers import adam, rmsprop, sgd, adagrad, adadelta, adamax
from tensorflow.python.keras.saving import save_model
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow_datasets import load, as_numpy

from core.adaptation import Method
from core.losses import distillation_loss, pkt_loss

OptimizerType = Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]


def load_data(dataset: str) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray], int]:
    """
    Loads the dataset.

    :return: the data and the number of classes.
    """
    data, info = load(dataset, with_info=True)
    data = as_numpy(data)
    data_shape = info.features.shape['image']
    labels_shape = info.features.shape['label']
    train_examples_num = info.splits['train'].num_examples
    test_examples_num = info.splits['test'].num_examples
    classes = info.features['label'].num_classes

    train, test = data['train'], data['test']
    train_data = empty((train_examples_num,) + data_shape)
    train_labels = empty((train_examples_num,) + labels_shape)
    test_data = empty((test_examples_num,) + data_shape)
    test_labels = empty((test_examples_num,) + labels_shape)

    for i, (sample_train, sample_test) in enumerate(zip_longest(train, test)):
        train_data[i] = sample_train['image']
        train_labels[i] = sample_train['label']

        if i < test_examples_num:
            test_data[i] = sample_test['image']
            test_labels[i] = sample_test['label']

    return (train_data, train_labels), (test_data, test_labels), classes


def preprocess_data(dataset: str, train: ndarray, test: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Preprocess the given data.

    :param dataset: The name of the dataset used.
    Currently not being used, but preprocessing should be based on the dataset.
    :param train: the train data.
    :param test: the test data.
    :return: the preprocessed data.
    """
    train, test = train / 255, test / 255

    return train, test


def initialize_optimizer(optimizer_name: str, learning_rate: float = None, decay: float = None, beta1: float = None,
                         beta2: float = None, rho: float = None, momentum: float = None, clip_norm=None,
                         clip_value=None) -> OptimizerType:
    """
    Initializes an optimizer based on the user's choices.
    Refer to Keras docs for the parameters and the optimizers.
    Any parameter which is not used from the given compiler, will be ignored.

    :param optimizer_name: the name of the optimizer to be used.
    Available names: 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
    :return: the optimizer.
    """
    if optimizer_name == 'adam':
        opt = adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=decay)
    elif optimizer_name == 'rmsprop':
        opt = rmsprop(lr=learning_rate, rho=rho, decay=decay)
    elif optimizer_name == 'sgd':
        opt = sgd(lr=learning_rate, momentum=momentum, decay=decay)
    elif optimizer_name == 'adagrad':
        opt = adagrad(lr=learning_rate, decay=decay)
    elif optimizer_name == 'adadelta':
        opt = adadelta(lr=learning_rate, rho=rho, decay=decay)
    elif optimizer_name == 'adamax':
        opt = adamax(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=decay)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')

    if clip_norm is not None:
        opt.clip_norm = clip_norm
    if clip_value is not None:
        opt.clip_value = clip_value
    return opt


class _SelectiveLearningCallback(Callback):
    """ Callback for the Selective Learning framework. """

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch: int, logs=None):
        # Find the layer(sub-model) of the model with the largest benefit for the student.
        best_layer_idx = None
        best_metric_result = inf
        range_of_submodels = range(len(self.model.output))

        for submodel_idx in range_of_submodels:
            layer_idx = submodel_idx + 1
            current_metric_result = logs['val_student_{}_loss'.format(submodel_idx)]
            # We want the best metric result of the submodels for each epoch.
            if current_metric_result < best_metric_result:
                best_layer_idx = layer_idx
                best_metric_result = current_metric_result

        # Retrieve the best weights.
        best_weights = self.model.layers[best_layer_idx].get_weights()

        # Set all the sub-output-layers weights equal with the best one's. (Greedy approach)
        for submodel_idx in range_of_submodels:
            layer_idx = submodel_idx + 1
            if layer_idx != best_layer_idx:
                self.model.layers[layer_idx].set_weights(best_weights)


def init_callbacks(monitor: str, lr_patience: int, lr_decay: float, lr_min: float, early_stopping_patience: int,
                   verbosity: int, weights_path: str, selective_learning: bool) -> []:
    """
    Initializes callbacks for the training procedure.

    :param monitor: the metric to monitor.
    :param lr_patience: the number of epochs to wait before decaying the learning rate. Set it to 0 to ignore decaying.
    :param lr_decay: the decay of the learning rate.
    :param lr_min: the minimum learning rate to be reached.
    :param early_stopping_patience: the number of epochs to wait before early stopping.
    :param verbosity: the verbosity of the callbacks.
    :param weights_path: path to the weights to be saved. Pass None, in order to not save the best model.
    :param selective_learning: whether the selective_learning framework is being used.
    :return: the callbacks list.
    """
    callbacks = []

    if not selective_learning and weights_path is not None:
        callbacks.append(ModelCheckpoint(weights_path, monitor, save_weights_only=True, save_best_only=True))

    if lr_decay > 0 or lr_patience == 0:
        learning_rate_reduction = ReduceLROnPlateau(monitor=monitor, patience=lr_patience, verbose=verbosity,
                                                    factor=lr_decay, min_lr=lr_min)
        callbacks.append(learning_rate_reduction)

    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor=monitor, patience=early_stopping_patience, min_delta=.0002,
                                       mode='max', restore_best_weights=True, verbose=verbosity)
        callbacks.append(early_stopping)

    if selective_learning:
        callbacks.append(_SelectiveLearningCallback())

    return callbacks


def create_path(filepath: str, base: bool = False) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    :param base: whether to create the base dir only, or the full path.
    """
    # Get the file's directory.
    directory = dirname(filepath) if base else filepath

    # Create directory if it does not exist
    if not exists(directory):
        makedirs(directory)


def save_students(save_students_mode: str, results: list, out_folder: str, results_name_prefix: str) -> None:
    """
    Saves the student network(s).

    :param save_students_mode: the save mode.
    :param results: the KT results.
    :param out_folder: the folder in which the student networks will be saved.
    :param results_name_prefix: a prefix for the results name.
    """
    # Get project logger.
    kt_logging = logging.getLogger('KT')

    if save_students_mode == 'all':
        for result in results:
            model_name = join(out_folder, '{}{}_model.h5'.format(results_name_prefix, result['method']))
            save_model(result['network'], model_name)
            kt_logging.info('Student network has been saved as {}.'.format(model_name))

    elif save_students_mode == 'best':
        best = -1
        best_model = None
        for result in results:
            if result['method'] != 'Probabilistic Knowledge Transfer':
                accuracy_idx = result['network'].metrics_names.index('categorical_accuracy')
                accuracy = result['evaluation'][accuracy_idx]
                if accuracy > best:
                    best = accuracy
                    best_model = result['network']

        model_name = join(out_folder, '{}best_model.h5'.format(results_name_prefix))

        if best_model is not None:
            save_model(best_model, model_name)
            kt_logging.info('The best student network has been saved as {}.'.format(model_name))


def _get_model_results(scores: list, metrics_names: list) -> str:
    """
    Makes a model's results string.

    :param scores: the model's scores.
    :param metrics_names: the model's metrics names.
    :return: the results string.
    """
    results = ''
    for i in range(len(scores)):
        results += "    {}: {:.4}\n".format(metrics_names[i], scores[i])

    return results


def log_results(results: List[Dict]) -> None:
    """
    Logs the KT results formatted.

    :param results: the results list.
    """
    # Get project logger.
    kt_logging = logging.getLogger('KT')

    # Show final results.
    final_results = 'Final results: \n'

    teacher_params = count_params(results[-1]['network'].trainable_weights)
    student_params = count_params(results[0]['network'].trainable_weights)
    final_results += 'Parameters:\n    Teacher params: {}\n    Student params: {}\n    Ratio: T/S={:.4} S/T={:.4}\n' \
        .format(teacher_params, student_params, teacher_params / student_params, student_params / teacher_params)

    for result in results:
        final_results += result['method'] + ': \n'
        final_results += _get_model_results(result['evaluation'], result['network'].metrics_names)
    kt_logging.info(final_results)


def save_res(results: List, filepath: str) -> None:
    """
    Saves the results object as a pickle file, after deleting the model.

    :param results: the results object to be saved.
    :param filepath: the filepath for the pickle file to be saved.
    """
    for result in results:
        del result['network']

    with open(filepath, 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)


def generate_appropriate_methods(kt_methods: Union[str, List[str]], temperature: float, kd_lambda_supervised: float,
                                 pkt_lambda_supervised: float, selective_learning_outputs: int) -> List[dict]:
    """
    Generates and returns a list of the methods which need to be applied, depending on the user input.
    
    :param kt_methods: the methods to be used for KT.
    :param temperature: the temperature for the distillation.
    :param kd_lambda_supervised: the weight for supervised KD loss.
    :param pkt_lambda_supervised: the weight for supervised PKT loss.
    :param selective_learning_outputs: the number of outputs if selective_learning framework is being used.
    :return: a list of dicts containing all the methods, formatted appropriately.
    """
    methods: List[dict] = []

    kd = {
        'name': 'Knowledge Distillation',
        'method': Method.DISTILLATION,
        'loss': distillation_loss(temperature, kd_lambda_supervised)
    }

    pkt = {
        'name': 'Probabilistic Knowledge Transfer',
        'method': Method.PKT,
        'loss': pkt_loss(pkt_lambda_supervised)
    }

    pkt_plus_distillation = {
        'name': 'PKT plus Distillation',
        'method': Method.PKT_PLUS_DISTILLATION,
        'loss': [distillation_loss(temperature, kd_lambda_supervised), pkt_loss(0)]
    }

    if isinstance(kt_methods, str):
        if kt_methods == 'distillation':
            methods.append(kd)
        elif kt_methods == 'pkt':
            methods.append(pkt)
        elif kt_methods == 'pkt+distillation':
            methods.append(pkt_plus_distillation)
    else:
        if 'distillation' in kt_methods:
            methods.append(kd)
        if 'pkt' in kt_methods:
            methods.append(pkt)
        if 'pkt+distillation' in kt_methods:
            methods.append(pkt_plus_distillation)

    if selective_learning_outputs:
        new_loss = []
        for method in methods:
            if isinstance(method['loss'], list):
                [new_loss.extend(method['loss']) for _ in range(selective_learning_outputs)]
                method['loss'] = new_loss
            else:
                method['loss'] = [method['loss'] for _ in range(selective_learning_outputs)]

    return methods
