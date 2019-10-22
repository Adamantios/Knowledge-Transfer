import logging
from os.path import join
from typing import Tuple, List, Union

from numpy import concatenate
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import categorical_crossentropy, mse
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical

from core.adaptation import Method, kt_metric, kd_student_adaptation, kd_student_rewind, \
    pkt_plus_kd_student_adaptation, pkt_plus_kd_rewind
from core.losses import LossType
from utils.helpers import initialize_optimizer, load_data, preprocess_data, create_student, init_callbacks, \
    save_students, log_results, copy_model, create_path, save_res, generate_appropriate_methods
from utils.logging import KTLogger
from utils.parser import create_parser
from utils.plotter import plot_results


def check_args() -> None:
    """ Checks the input arguments. """
    if clip_norm is not None and clip_value is not None:
        raise ValueError('You cannot set both clip norm and clip value.')


def knowledge_transfer(method: Method, loss: LossType) -> Tuple[Model, History]:
    """
    Performs KT.

    :param method: the method used fot the KT.
    :param loss: the KT loss to be used.
    :return: Tuple containing a student Keras model and its training History object.
    """
    # Create student model.
    kt_logging.info('Creating student...')
    student = create_student(student_name, x_train.shape[1:], n_classes, start_weights)

    # Adapt student, if necessary.
    if method == Method.DISTILLATION:
        student = kd_student_adaptation(student, temperature)
    if method == Method.PKT_PLUS_DISTILLATION:
        student = pkt_plus_kd_student_adaptation(student, temperature)

    kt_logging.debug('Configuring student...')

    # Create optimizer.
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)

    # Create KT metrics for Distillation and give them names.
    # PKT performs KT, but also rotates the space, thus evaluating results has no meaning,
    # since the neurons representing the classes are not the same anymore.
    metrics = {}
    if method == Method.DISTILLATION or method == Method.PKT_PLUS_DISTILLATION:
        kt_acc = kt_metric(categorical_accuracy, method)
        kt_acc.__name__ = 'accuracy'
        kt_crossentropy = kt_metric(categorical_crossentropy, method)
        kt_crossentropy.__name__ = 'crossentropy'
        metrics['concatenate'] = [kt_acc, kt_crossentropy]
    else:
        metrics['softmax'] = []

    # Create importance weights for the different losses if method is PKT_PLUS_DISTILLATION.
    weights = None
    if method == Method.PKT_PLUS_DISTILLATION:
        weights = [kd_importance_weight, pkt_importance_weight]

    # Compile student.
    student.compile(optimizer, loss, metrics, weights)

    # Initialize callbacks list.
    kt_logging.debug('Initializing Callbacks...')
    if method == Method.DISTILLATION:
        callbacks_list = init_callbacks('val_accuracy', lr_patience, lr_decay, lr_min, early_stopping_patience,
                                        verbosity)
    elif method == Method.PKT:
        callbacks_list = init_callbacks('val_loss', lr_patience, lr_decay, lr_min, early_stopping_patience, verbosity)
    else:
        callbacks_list = init_callbacks('val_concatenate_accuracy', lr_patience, lr_decay, lr_min,
                                        early_stopping_patience, verbosity)

    # Fit student.
    if method == Method.PKT_PLUS_DISTILLATION:
        history = student.fit(x_train, [y_train_concat, y_train_concat], batch_size=batch_size, epochs=epochs,
                              validation_data=(x_test, [y_test_concat, y_test_concat]),
                              callbacks=callbacks_list)
    else:
        history = student.fit(x_train, y_train_concat, batch_size=batch_size, epochs=epochs,
                              validation_data=(x_test, y_test_concat),
                              callbacks=callbacks_list)

    # Rewind student to normal, if necessary.
    if method == Method.DISTILLATION:
        student = kd_student_rewind(student)
    if method == Method.PKT_PLUS_DISTILLATION:
        student = pkt_plus_kd_rewind(student)

    return copy_model(student), history


def evaluate_results(results: list) -> None:
    """
    Evaluates the KT results.

    :param results: the results list.
    """
    # Add baseline to the results list.
    results.append({
        'method': 'Teacher',
        'network': teacher,
        'history': None,
        'evaluation': None
    })

    # Create optimizer.
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)

    for result in results:
        kt_logging.info('Evaluating {}...'.format(result['method']))
        result['network'].compile(optimizer, mse, [categorical_accuracy, categorical_crossentropy])
        if result['method'] != 'Probabilistic Knowledge Transfer':
            result['evaluation'] = result['network'].evaluate(x_test, y_test, evaluation_batch_size, verbosity)
        else:
            # Get pkt features and pass them through a knn classifier, in order to calculate accuracy.
            pkt_features_train = result['network'].predict(x_train, evaluation_batch_size, verbosity)
            pkt_features_test = result['network'].predict(x_test, evaluation_batch_size, verbosity)
            knn = KNeighborsClassifier(k, n_jobs=-1)
            knn.fit(pkt_features_train, y_train)
            y_pred = knn.predict(pkt_features_test)
            result['evaluation'] = [
                result['network'].evaluate(x_test, y_test, evaluation_batch_size, verbosity)[0],
                accuracy_score(y_test, y_pred),
                log_loss(y_test, y_pred)
            ]

    kt_logging.debug(results)

    # Plot training information.
    save_folder = out_folder if save_results else None
    plot_results(results, epochs, save_folder)

    # Log results.
    log_results(results)


def run_kt_methods() -> None:
    """ Runs all the available KT methods. """
    methods = generate_appropriate_methods(kt_methods, temperature, kd_lambda_supervised, pkt_lambda_supervised)
    results = []

    for method in methods:
        kt_logging.info('Performing {}...'.format(method['name']))
        student, history = knowledge_transfer(method['method'], method['loss'])
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

    kt_logging.info('Evaluating results...')
    evaluate_results(results)

    kt_logging.info('Saving student network(s)...')
    save_students(save_students_mode, results[:-1], out_folder)

    if save_results:
        kt_logging.info('Saving results...')
        save_res(results, join(out_folder, results_name_prefix + '_results.pkl'))


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    teacher: Model = load_model(args.teacher, compile=False)
    student_name: str = args.student
    dataset: str = args.dataset
    kt_methods: Union[str, List[str]] = args.method
    start_weights: str = args.start_weights
    temperature: float = args.temperature
    kd_lambda_supervised: float = args.kd_lambda_supervised
    pkt_lambda_supervised: float = args.pkt_lambda_supervised
    k: int = args.neighbors
    kd_importance_weight: float = args.kd_importance_weight
    pkt_importance_weight: float = args.pkt_importance_weight
    save_students_mode: str = args.save_students
    save_results: bool = not args.omit_results
    results_name_prefix: str = args.results_name_prefix
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
    kt_logger = KTLogger(join(out_folder, results_name_prefix + '_output.log'))
    kt_logger.setup_logger(debug, save_results)
    kt_logging = logging.getLogger('KT')
    kt_logging.info('\n---------------------------------------------------------------------------------------------\n')

    # Load dataset.
    kt_logging.info('Loading dataset...')
    ((x_train, y_train), (x_test, y_test)), n_classes = load_data(dataset)

    # Preprocess data.
    kt_logging.info('Preprocessing data...')
    x_train, x_test = preprocess_data(dataset, x_train, x_test)
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    # Get teacher's outputs.
    kt_logging.info('Getting teacher\'s predictions...')
    y_teacher_train = teacher.predict(x_train, evaluation_batch_size, verbosity)
    y_teacher_test = teacher.predict(x_test, evaluation_batch_size, verbosity)

    # Concatenate teacher's outputs with true labels.
    y_train_concat = concatenate([y_train, y_teacher_train], axis=1)
    y_test_concat = concatenate([y_test, y_teacher_test], axis=1)

    # Run kt.
    kt_logging.info('Starting KT method(s)...')
    run_kt_methods()

    # Show close message.
    kt_logging.info('Finished!')

    # Close logger.
    kt_logger.close_logger()

    # TODO evaluate on test data and split train data for the validation set.
