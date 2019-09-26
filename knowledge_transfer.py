import logging
from typing import Tuple, Union, List

from numpy import concatenate
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from core.adaptation import Method, kt_metric, kd_student_adaptation
from core.losses import LossType, distillation_loss, pkt_loss
from utils.helpers import initialize_optimizer, load_data, preprocess_data, create_student, init_callbacks, \
    setup_logger, OptimizerType, save_students, log_results
from utils.parser import create_parser
from utils.plotter import plot_results


def check_args() -> None:
    """ Checks the input arguments. """
    if clip_norm is not None and clip_value is not None:
        raise ValueError('You cannot set both clip norm and clip value.')


def knowledge_transfer(optimizer: OptimizerType, method: Method, loss: LossType) -> \
        Tuple[Model, History, Union[None, float, List[float]]]:
    """
    Performs KT.

    :param optimizer: the optimizer to be used for the KT.
    :param method: the method used fot the KT.
    :param loss: the KT loss to be used.
    :return: Keras History object.
    """
    # Create student model.
    logging.info('Creating student...')
    student = create_student(student_name, x_train.shape[1:], n_classes, start_weights)

    if method == Method.DISTILLATION:
        student = kd_student_adaptation(student, temperature)

    logging.info('Configuring...')
    # Compile student.
    student.compile(
        optimizer=optimizer, loss=loss,
        metrics=[kt_metric(accuracy, method), kt_metric(categorical_crossentropy, method)]
    )

    # Fit student.
    history = student.fit(x_train, y_train_concat, batch_size=batch_size, epochs=epochs,
                          validation_data=(x_test, y_test_concat),
                          callbacks=callbacks_list)

    # Evaluating results.
    logging.info('Evaluating student\'s results.')
    evaluation = student.evaluate(x_test, y_test, evaluation_batch_size, verbosity)

    return student, history, evaluation


def evaluate_results(results: list) -> None:
    """
    Evaluates the KT comparison results.

    :param results: the results list.
    """
    # Get baseline.
    logging.info('Calculating baseline.')
    baseline = teacher.evaluate(x_test, y_test, evaluation_batch_size, verbosity)
    # Add baseline to the results list.
    results.append({
        'method': 'Teacher',
        'network': teacher,
        'history': None,
        'evaluation': baseline
    })
    # Plot training information.
    save_folder = out_folder if save_results else None
    plot_results(results, save_folder)

    # Log results.
    log_results(results, save_results, out_folder)


def compare_kt_methods() -> None:
    """ Compares all the available KT methods. """
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)
    methods = [
        {
            'name': 'Knowledge Distillation',
            'method': Method.DISTILLATION,
            'loss': distillation_loss(temperature, lambda_supervised)
        },
        {
            'name': 'Probabilistic Knowledge Transfer',
            'method': Method.PKT,
            'loss': pkt_loss(lambda_supervised)
        }
    ]
    results = []

    for method in methods:
        logging.info('Performing {}...'.format(method['name']))
        student, history, evaluation = knowledge_transfer(optimizer, method['method'], method['loss'])
        results.append({
            'method': method['name'],
            'network': student,
            'history': history.history,
            'evaluation': evaluation
        })

    logging.debug(results)

    logging.info('Saving student network(s)...')
    save_students(save_students_mode, results, out_folder)

    logging.info('Evaluating results...')
    evaluate_results(results)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    teacher: Model = load_model(args.teacher, compile=False)
    student_name: str = args.student
    dataset: str = args.dataset
    start_weights: str = args.start_weights
    temperature: float = args.temperature
    lambda_supervised: float = args.lambda_supervised
    save_students_mode: str = args.save_students
    save_results: bool = not args.omit_results
    out_folder: str = args.out_folder
    debug: bool = args.debug
    optimizer_name: str = args.optimizer
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
    logging.info('Loading dataset...')
    ((x_train, y_train), (x_test, y_test)), n_classes = load_data(dataset)

    # Preprocess data.
    logging.info('Preprocessing data...')
    x_train, x_test = preprocess_data(dataset, x_train, x_test)
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    # Get teacher's outputs.
    logging.info('Getting teacher\'s predictions...')
    y_teacher_train = teacher.predict(x_train, evaluation_batch_size, verbosity)
    y_teacher_test = teacher.predict(x_test, evaluation_batch_size, verbosity)

    # Concatenate teacher's outputs with true labels.
    y_train_concat = concatenate([y_train, y_teacher_train])
    y_test_concat = concatenate([y_test, y_teacher_test])

    # Initialize callbacks list.
    logging.info('Configuring...')
    callbacks_list = init_callbacks(lr_patience, lr_decay, lr_min, early_stopping_patience, verbosity)

    # Run comparison.
    logging.info('Starting KT methods comparison...')
    compare_kt_methods()
    logging.info('Finished!')
