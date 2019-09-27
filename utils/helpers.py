import logging
from os import makedirs
from os.path import dirname, exists, isfile, join
from typing import Union, Tuple, Any, Dict, List

from numpy import ndarray
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.models import clone_model
from tensorflow.python.keras.optimizers import adam, rmsprop, sgd, adagrad, adadelta, adamax
from tensorflow.python.keras.saving import save_model

from student_networks.cifar10_tiny_1 import cifar10_tiny_1

OptimizerType = Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]


def setup_logger(debug: bool) -> None:
    """
    Sets the program's logger up.

    :param debug: Whether the logger should be set in debugging mode.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=level)


def load_data(dataset: str) -> [Tuple[ndarray, ndarray], Tuple[Any, ndarray], int]:
    """
    Loads the dataset.

    :param dataset: The name of the dataset to load.
    :return: the data and the number of classes.
    """
    if dataset == 'cifar10':
        data, classes = cifar10.load_data(), 10
    elif dataset == 'cifar100':
        data, classes = cifar100.load_data(), 100
    else:
        raise ValueError("Unrecognised dataset!")

    return data, classes


def preprocess_data(dataset: str, train: ndarray, test: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Preprocess the given data.

    :param dataset: The name of the dataset used.
    :param train: the train data.
    :param test: the test data.
    :return: the preprocessed data.
    """
    if dataset == 'cifar10' or dataset == 'cifar100':
        train, test = train / 255, test / 255
    else:
        raise ValueError("Unrecognised dataset!")

    return train, test


def create_student(student_name: str, input_shape: tuple, n_classes: int, start_weights: str = None) -> Model:
    """
    Creates the student model and loads weights as a start point if they exist.

    :param student_name: the name of the student model to be used. It can be one of: 'cifar10_tiny_1'
    :param input_shape: the student model's input shape.
    :param n_classes: the number of classes to predict.
    :param start_weights: path to weights to initialize the student model with.
    :return: Keras Sequential model.
    """
    if student_name == 'cifar10_tiny_1':
        model_generator = cifar10_tiny_1
    else:
        raise ValueError('Unrecognised student model!')

    if start_weights != '' and start_weights is not None:
        if isfile(start_weights):
            return model_generator(input_shape=input_shape, weights_path=start_weights, n_classes=n_classes)
        else:
            raise FileNotFoundError('Checkpoint file \'{}\' not found.'.format(start_weights))
    else:
        return model_generator(input_shape=input_shape, n_classes=n_classes)


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


def init_callbacks(lr_patience: int, lr_decay: float, lr_min: float, early_stopping_patience: int,
                   verbosity: int) -> []:
    """
    Initializes callbacks for the training procedure.

    :param lr_patience: the number of epochs to wait before decaying the learning rate. Set it to 0 to ignore decaying.
    :param lr_decay: the decay of the learning rate.
    :param lr_min: the minimum learning rate to be reached.
    :param early_stopping_patience: the number of epochs to wait before early stopping.
    :param verbosity: the verbosity of the callbacks.
    :return: the callbacks list.
    """
    callbacks = []

    if lr_decay > 0 or lr_patience == 0:
        learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy', patience=lr_patience, verbose=verbosity,
                                                    factor=lr_decay, min_lr=lr_min)
        callbacks.append(learning_rate_reduction)

    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor='accuracy', patience=early_stopping_patience, min_delta=.002,
                                       verbose=verbosity)
        callbacks.append(early_stopping)

    return callbacks


def create_path(filepath: str) -> None:
    """
    Creates a path to a file, if it does not exist.

    :param filepath: the filepath.
    """
    # Get the file's directory.
    directory = dirname(filepath)

    # Create directory if it does not exist
    if not exists(directory):
        makedirs(directory)


def copy_model(model: Model) -> Model:
    """
    Copies a Keras Model.

    :param model: the model to be copied.
    :return: the copied Model.
    """
    copy = clone_model(model)
    copy.build(model.input_shape)
    copy.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    copy.set_weights(model.get_weights())

    return copy


def save_students(save_students_mode: str, results: list, out_folder: str) -> None:
    """
    Saves the student network(s).

    :param save_students_mode: the save mode.
    :param results: the KT results.
    :param out_folder: the folder in which the student networks will be saved.
    """
    if save_students_mode == 'all':
        for result in results:
            name = join(out_folder, result['method'], '_model.h5')
            save_model(result['network'], name)
            logging.info('Student network has been saved as {}...'.format(name))

    elif save_students_mode == 'best':
        best = -1
        best_model = None
        for result in results:
            if result['network']['acc'] > best:
                best = result['network']
                best_model = result['model']

        name = join(out_folder, 'best_model.h5')
        save_model(best_model, )
        logging.info('The best student network has been saved as {}...'.format(name))


def _get_model_results(scores: list, metrics_names: list) -> str:
    """
    Makes a model's results string.

    :param scores: the model's scores.
    :param metrics_names: the model's metrics names.
    :return: the results string.
    """
    results = ''
    for i in range(len(scores)):
        results += "{}: {}\n".format(metrics_names[i], scores[1])

    return results


def log_results(results: List[Dict], save_results: bool, out_folder: str) -> None:
    """
    Prints the KT comparison results string.

    :param results: the comparison results list.
    :param save_results: whether the results should be saved to a file.
    :param out_folder: the out folder for the results.
    """
    # Show final results.
    final_results = 'Final results: \n'
    for result in results:
        final_results += result['method'] + ': \n'
        final_results = _get_model_results(result['evaluation'], result['network'].metrics_names)
    logging.info(final_results)

    # Save final results.
    if save_results:
        results_filepath = out_folder + 'final_results.log'
        logging.log(msg=final_results, filename=results_filepath, filemode='w', format='%(message)s',
                    level=logging.INFO)
        logging.info('Evaluation results have been saved as {}.\n'.format(results_filepath))
