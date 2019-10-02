import logging
from typing import Tuple

from numpy import concatenate
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import categorical_crossentropy, mse
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from core.adaptation import Method, kt_metric, kd_student_adaptation, kd_student_rewind
from core.losses import LossType, distillation_loss, pkt_loss
from utils.helpers import initialize_optimizer, load_data, preprocess_data, create_student, init_callbacks, \
    setup_logger, OptimizerType, save_students, log_results, copy_model, create_path
from utils.parser import create_parser
from utils.plotter import plot_results


def check_args() -> None:
    """ Checks the input arguments. """
    if clip_norm is not None and clip_value is not None:
        raise ValueError('You cannot set both clip norm and clip value.')


def knowledge_transfer(optimizer: OptimizerType, method: Method, loss: LossType) -> Tuple[Model, History]:
    """
    Performs KT.

    :param optimizer: the optimizer to be used for the KT.
    :param method: the method used fot the KT.
    :param loss: the KT loss to be used.
    :return: Tuple containing a student Keras model and its training History object.
    """
    # Create student model.
    logging.info('Creating student...')
    student = create_student(student_name, x_train.shape[1:], n_classes, start_weights)

    # Adapt student for distillation if necessary.
    if method == Method.DISTILLATION:
        student = kd_student_adaptation(student, temperature)

    logging.debug('Configuring student...')

    # Create KT metrics and give them names.
    kt_acc = kt_metric(categorical_accuracy, method)
    kt_acc.__name__ = 'accuracy'
    kt_crossentropy = kt_metric(categorical_crossentropy, method)
    kt_crossentropy.__name__ = 'crossentropy'

    # Compile student.
    student.compile(optimizer=optimizer, loss=loss, metrics=[kt_acc, kt_crossentropy])

    # Fit student.
    history = student.fit(x_train, y_train_concat, batch_size=batch_size, epochs=epochs,
                          validation_data=(x_test, y_test_concat),
                          callbacks=callbacks_list)

    # Rewind student to normal, if necessary.
    if method == Method.DISTILLATION:
        student = kd_student_rewind(student)

    return copy_model(student), history


def evaluate_results(optimizer: OptimizerType, results: list) -> None:
    """
    Evaluates the KT comparison results.

    :param optimizer: the optimizer to be used for the teacher.
    :param results: the results list.
    """
    # Add baseline to the results list.
    results.append({
        'method': 'Teacher',
        'network': teacher,
        'history': None,
        'evaluation': None
    })

    for result in results:
        logging.info('Evaluating {}...'.format(result['method']))
        result['network'].compile(optimizer, mse, [categorical_accuracy, categorical_crossentropy])
        result['evaluation'] = result['network'].evaluate(x_test, y_test, evaluation_batch_size, verbosity)
    logging.debug(results)

    # Plot training information.
    save_folder = out_folder if save_results else None
    plot_results(results, epochs, save_folder)

    # Log results.
    log_results(results)


def compare_kt_methods() -> None:
    """ Compares all the available KT methods. """
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)
    methods = [
        {
            'name': 'Knowledge Distillation',
            'method': Method.DISTILLATION,
            'loss': distillation_loss(temperature, kd_lambda_supervised)
        },
        {
            'name': 'Probabilistic Knowledge Transfer',
            'method': Method.PKT,
            'loss': pkt_loss(pkt_lambda_supervised)
        }
    ]
    results = []

    for method in methods:
        logging.info('Performing {}...'.format(method['name']))
        student, history = knowledge_transfer(optimizer, method['method'], method['loss'])
        # TODO model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        #  and save student model there, when we stop needing it,
        #  because it is inefficient to have it in memory until - if ever - we need to save it.
        #  That way, when the time comes, we will just need to move it to the out folder.
        results.append({
            'method': method['name'],
            'network': student,
            'history': history.history,
            'evaluation': None
        })

    logging.info('Evaluating results...')
    evaluate_results(optimizer, results)

    logging.info('Saving student network(s)...')
    save_students(save_students_mode, results[:-1], out_folder)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    teacher: Model = load_model(args.teacher, compile=False)
    student_name: str = args.student
    dataset: str = args.dataset
    start_weights: str = args.start_weights
    temperature: float = args.temperature
    kd_lambda_supervised: float = args.kd_lambda_supervised
    pkt_lambda_supervised: float = args.pkt_lambda_supervised
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

    # Create out folder path.
    create_path(out_folder)

    # Set logger up.
    setup_logger(debug, save_results, out_folder)
    logging.info('\n------------------------------------------------------------------------------------------------\n')

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
    y_train_concat = concatenate([y_train, y_teacher_train], axis=1)
    y_test_concat = concatenate([y_test, y_teacher_test], axis=1)

    # Initialize callbacks list.
    logging.debug('Initializing Callbacks...')
    callbacks_list = init_callbacks(lr_patience, lr_decay, lr_min, early_stopping_patience, verbosity)

    # Run comparison.
    logging.info('Starting KT methods comparison...')
    compare_kt_methods()
    logging.info('Finished!')
