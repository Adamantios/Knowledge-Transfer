from os import makedirs
from os.path import dirname, exists, isfile
from typing import Union, Tuple, Any

from numpy import ndarray
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.optimizers import adam, rmsprop, sgd, adagrad, adadelta, adamax

from student_networks.cifar10_tiny_1 import cifar10_tiny_1

OptimizerType = Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]


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

    if start_weights != '' or start_weights is not None:
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


def init_callbacks(save_checkpoint: bool, checkpoint_filepath: str, lr_patience: int, lr_decay: float, lr_min: float,
                   early_stopping_patience: int, verbosity: int) -> []:
    """
    Initializes callbacks for the training procedure.

    :param save_checkpoint: whether a checkpoint should be saved.
    :param checkpoint_filepath: the filepath to the checkpoint.
    :param lr_patience: the number of epochs to wait before decaying the learning rate. Set it to 0 to ignore decaying.
    :param lr_decay: the decay of the learning rate.
    :param lr_min: the minimum learning rate to be reached.
    :param early_stopping_patience: the number of epochs to wait before early stopping.
    :param verbosity: the verbosity of the callbacks.
    :return: the callbacks list.
    """
    callbacks = []
    if save_checkpoint:
        # Create path for the file.
        create_path(checkpoint_filepath)

        # Create checkpoint.
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=verbosity,
                                     save_best_only=True, mode='max')
        callbacks.append(checkpoint)

    if lr_decay > 0 or lr_patience == 0:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=lr_patience, verbose=verbosity,
                                                    factor=lr_decay, min_lr=lr_min)
        callbacks.append(learning_rate_reduction)

    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(monitor='val_acc', patience=early_stopping_patience, min_delta=.002,
                                       verbose=verbosity)
        callbacks.append(early_stopping)

    return callbacks


def plot_results(histories: list, save_folder: str):
    """
    Plots the KT results.

    :param histories: the history for each one of the KT method applied.
    :param save_folder:
    :return:
    """
    pass


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
