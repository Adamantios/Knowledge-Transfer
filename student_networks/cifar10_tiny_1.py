import logging
from os.path import isfile
from typing import Union

from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.engine import InputLayer
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.python.keras.regularizers import l2


def cifar10_tiny_1(n_classes: int, input_shape=None, input_tensor=None, weights_path: Union[None, str] = None) -> Model:
    """
    Defines a cifar10 tiny network.

    :param n_classes: the number of classes.
    :param input_shape: the input shape of the network. Can be omitted if input_tensor is used.
    :param input_tensor: the input tensor of the network. Can be omitted if input_shape is used.
    :param weights_path: a path to a trained cifar10 tiny network's weights.
    :return: Keras functional Model.
    """
    if input_shape is None and input_tensor is None:
        raise ValueError('You need to specify input shape or input tensor for the network.')

    # Create input.
    if input_shape is None:
        # Create an InputLayer using the input tensor.
        inputs = InputLayer(input_tensor=input_tensor, name='input')
    else:
        inputs = Input(shape=input_shape, name='input')

    # Define a weight decay for the regularisation.
    weight_decay = 1e-4

    # Block1.
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv1',
               kernel_regularizer=l2(weight_decay))(inputs)

    x = BatchNormalization(name='block1_batch-norm1')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='elu', name='block1_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block1_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)
    x = Dropout(0.2, name='block1_dropout')(x)

    # Block2.
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='block2_conv1',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm1')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='elu', name='block2_conv2',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(name='block2_batch-norm2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(x)
    x = Dropout(0.3, name='block2_dropout')(x)

    # Add top layers.
    x = Flatten()(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    # Create model.
    model = Model(inputs, outputs, name='cifar10_tiny_1')

    if weights_path is not None:
        if not isfile(weights_path):
            raise FileNotFoundError('Network weights file {} does not exist.'.format(weights_path))
        model.load_weights(weights_path, True)

    logging.debug('Network summary:\n{}'.format(model.summary()))

    return model