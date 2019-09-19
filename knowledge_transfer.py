from os.path import join

from numpy import ndarray
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from core.losses import available_methods, LossType
from utils.helpers import initialize_optimizer, load_data, preprocess_data, create_student, init_callbacks, \
    plot_results, setup_logger
from utils.parser import create_parser


def check_args() -> None:
    """ Checks the input arguments. """
    if clip_norm is not None and clip_value is not None:
        raise ValueError('You cannot set both clip norm and clip value.')


def knowledge_transfer(loss: LossType) -> History:
    """
    Perform KT using a certain loss.

    :param loss: the KT loss to be used.
    :return: Keras History object.
    """
    # Initialize optimizer and compile student.
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)
    student.compile(
        optimizer=optimizer, loss=loss,
        metrics=[accuracy, categorical_crossentropy]
    )

    if augment_data:
        # Generate batches of tensor image data with real-time data augmentation.
        datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
        datagen.fit(x_train)
        # Fit network.
        history = student.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                        steps_per_epoch=x_train.shape[0] // batch_size,
                                        validation_data=(x_test, y_test),
                                        callbacks=callbacks_list)
    else:
        history = student.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),
                              callbacks=callbacks_list)

    return history


def evaluate_results(results: list) -> None:
    """
    Evaluates the KT comparison results.

    :param results: the results list.
    """
    # Plot results.
    save_folder = out_folder if save_results else None
    plot_results(results, save_folder)


def compare_kt_methods() -> None:
    """ Compares all the available KT methods. """
    results = []
    for method in available_methods:
        results.append({'method': method['name'], 'results': knowledge_transfer(method['function']).history})

    evaluate_results(results)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    teacher: Model = load_model(args.teacher, compile=False)
    student_name: str = args.student
    dataset: str = args.dataset
    start_weights: str = args.start_weights
    save_checkpoint: bool = not args.omit_checkpoint
    temperature: float = args.temperature
    lambda_supervised: float = args.lambda_supervised
    save_results: bool = not args.omit_results
    out_folder: str = args.out_folder
    debug: bool = args.debug
    optimizer_name: str = args.optimizer
    augment_data: bool = not args.no_augmentation
    learning_rate: float = args.learning_rate
    lr_patience: int = args.learning_rate_patience
    lr_decay: float = args.learning_rate_decay
    lr_min: float = args.learning_rate_min
    early_stopping_patience: int = args.early_stopping_patience
    clip_norm: float = args.clip_norm
    clip_value: float = args.clip_value
    beta1: float = args.beta1
    beta2: float = args.beta2
    rho: float = args.rho
    momentum: float = args.momentum
    decay: float = args.decay
    batch_size: int = args.batch_size
    evaluation_batch_size: int = args.evaluation_batch_size
    epochs: int = args.epochs
    verbosity: int = args.verbosity
    check_args()

    # Set logger up.
    setup_logger(debug)

    # Load dataset.
    ((x_train, y_train), (x_test, y_test)), n_classes = load_data(dataset)

    # Preprocess data.
    x_train, x_test = preprocess_data(dataset, x_train, x_test)
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    # Create student model.
    student = create_student(student_name, x_train.shape[1:], n_classes, start_weights)

    # Initialize callbacks list.
    checkpoint_filepath = join(out_folder, 'checkpoint.h5')
    callbacks_list = init_callbacks(save_checkpoint, checkpoint_filepath, lr_patience, lr_decay, lr_min,
                                    early_stopping_patience, verbosity)

    # Run comparison.
    compare_kt_methods()
