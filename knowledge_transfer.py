import logging
from os import remove
from os.path import join, exists
from tempfile import gettempdir, _get_candidate_names
from typing import Tuple, List, Union

from numpy import concatenate, newaxis
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.losses import categorical_crossentropy, mse
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.models import clone_model
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras.utils import to_categorical
from tqdm import trange

from core.adaptation import Method, kt_metric, kd_student_adaptation, kd_student_rewind, \
    pkt_plus_kd_student_adaptation, pkt_plus_kd_rewind
from core.attention_framework import teacher_adaptation
from core.losses import LossType
from utils.helpers import initialize_optimizer, load_data, preprocess_data, init_callbacks, \
    save_students, log_results, create_path, save_res, generate_appropriate_methods, copy_model
from utils.logging import KTLogger
from utils.parser import create_parser
from utils.plotter import plot_results
from utils.tools import Crop


def check_args() -> None:
    """ Checks the input arguments. """
    if clip_norm is not None and clip_value is not None:
        raise ValueError('You cannot set both clip norm and clip value.')


def generate_supervised_metrics(method: Method) -> List:
    """ Generates and returns a list with supervised KT metrics. """
    kt_acc = kt_metric(categorical_accuracy, method)
    kt_crossentropy = kt_metric(categorical_crossentropy, method)
    kt_acc.__name__ = 'accuracy'
    kt_crossentropy.__name__ = 'crossentropy'
    return [kt_acc, kt_crossentropy]


def knowledge_transfer(current_student: Model, method: Method, loss: LossType) -> Tuple[Model, History]:
    """
    Performs KT.

    :param current_student: the student to be used for the current KT method.
    :param method: the method to be used for the KT.
    :param loss: the KT loss to be used.
    :return: Tuple containing a student Keras model and its training History object.
    """
    kt_logging.debug('Configuring student...')
    weights = None
    y_train_adapted = y_train_concat
    y_val_adapted = y_val_concat
    metrics = {}

    if method == Method.DISTILLATION:
        # Adapt student
        current_student = kd_student_adaptation(current_student, temperature)
        # Create KT metrics.
        metrics['concatenate'] = generate_supervised_metrics(method)
        monitoring_metric = 'val_accuracy'
    elif method == Method.PKT_PLUS_DISTILLATION:
        # Adapt student
        current_student = pkt_plus_kd_student_adaptation(current_student, temperature)
        # Create importance weights for the different losses.
        weights = [kd_importance_weight, pkt_importance_weight]
        #  Adapt the labels.
        if not attention:
            y_train_adapted = [y_train_concat, y_train_concat]
        y_val_adapted = [y_val_concat, y_val_concat]
        # Create KT metrics.
        metrics['concatenate'] = generate_supervised_metrics(method)
        monitoring_metric = 'val_concatenate_accuracy'
    else:
        # PKT performs KT, but also rotates the space, thus evaluating results has no meaning,
        # since the neurons representing the classes are not the same anymore.
        metrics['softmax'] = []
        monitoring_metric = 'val_loss'

    # Create optimizer.
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)

    # Compile student.
    current_student.compile(optimizer, loss, metrics, weights)

    # Initialize callbacks list.
    kt_logging.debug('Initializing Callbacks...')
    # Create a temp file, in order to save the model, if needed.
    tmp_weights_path = None
    if use_best_model:
        tmp_weights_path = join(gettempdir(), next(_get_candidate_names()) + '.h5')

    callbacks_list = init_callbacks(monitoring_metric, lr_patience, lr_decay, lr_min, early_stopping_patience,
                                    verbosity, tmp_weights_path)

    # Train student.
    if attention:
        history = History()
        best_student = copy_model(current_student, optimizer=optimizer, loss=loss, metrics=metrics,
                                  loss_weights=weights)
        progressbar_epochs = trange(epochs, unit='epoch')

        for epoch in progressbar_epochs:
            tmp_history = []
            best_student_idx = 0
            best_result = 0
            n_models = y_train_adapted.shape[1]
            progressbar_submodels = trange(n_models, desc='Submodel training', unit='submodel', leave=False)

            for model_idx in progressbar_submodels:
                # TODO replace unnecessary loop of serial training of the submodels and convert it to parallel,
                #  by changing the student's outputs to an output for each model
                #  and creating a loss function which chooses the best output, by means of a loss metric
                #  and updates all the weights of the submodels output layers
                #  with the weights of the best submodel's layer.
                # Adapt the labels.
                if method == Method.PKT_PLUS_DISTILLATION:
                    y_train_submodel = [y_train_adapted[:, model_idx], y_train_adapted[:, model_idx]]
                else:
                    y_train_submodel = y_train_adapted[:, model_idx]

                candidate_student = copy_model(best_student, optimizer=optimizer, loss=loss, metrics=metrics,
                                               loss_weights=weights)
                tmp_history.append(candidate_student.fit(x_train, y_train_submodel,
                                                         batch_size=batch_size, epochs=1, verbose=0,
                                                         validation_data=(x_val, y_val_adapted)))
                result = tmp_history[model_idx].history['val_loss'][0]
                if result < best_result or model_idx == 0:
                    best_result = result
                    best_student_idx = model_idx
                    best_student = copy_model(candidate_student, optimizer=optimizer, loss=loss, metrics=metrics,
                                              loss_weights=weights)
                    progressbar_epochs.set_description(desc='Current loss {:.4f}'.format(best_result))

            # Update history object.
            if epoch:
                history.epoch.append(epoch)
                for key, value in tmp_history[best_student_idx].history.items():
                    history.history.setdefault(key, []).append(value[0])
                history.model = tmp_history[best_student_idx].model
                history.params = tmp_history[best_student_idx].params
            else:
                history = tmp_history[best_student_idx]

        # Set student with the best trained.
        current_student = best_student
    else:
        history = current_student.fit(x_train, y_train_adapted, batch_size=batch_size, callbacks=callbacks_list,
                                      epochs=epochs, validation_data=(x_val, y_val_adapted), verbose=verbosity)

    if exists(tmp_weights_path):
        # Load best weights and delete the temp file.
        current_student.load_weights(tmp_weights_path)
        remove(tmp_weights_path)

    # Rewind student to its normal state, if necessary.
    if method == Method.DISTILLATION:
        current_student = kd_student_rewind(current_student)
    elif method == Method.PKT_PLUS_DISTILLATION:
        current_student = pkt_plus_kd_rewind(current_student)

    return current_student, history


def evaluate_results(results: list) -> None:
    """
    Evaluates the KT results.

    :param results: the results list.
    """
    # Create optimizer.
    optimizer = initialize_optimizer(optimizer_name, learning_rate, decay, beta1, beta2, rho, momentum,
                                     clip_norm, clip_value)

    for result in results:
        kt_logging.info('Evaluating {}...'.format(result['method']))
        result['network'].compile(optimizer, mse, [categorical_accuracy, categorical_crossentropy])
        if result['method'] == 'Teacher' and attention:
            result['evaluation'] = result['network'].evaluate(x_test, y_test, evaluation_batch_size,
                                                              verbosity)
        elif result['method'] != 'Probabilistic Knowledge Transfer':
            result['evaluation'] = result['network'].evaluate(x_test, y_test, evaluation_batch_size, verbosity)
        else:
            # Get pkt features and pass them through a knn classifier, in order to calculate accuracy.
            pkt_features_train = result['network'].predict(x_train, evaluation_batch_size, verbose=0)
            pkt_features_test = result['network'].predict(x_test, evaluation_batch_size, verbose=0)
            knn = KNeighborsClassifier(k, n_jobs=-1)
            knn.fit(pkt_features_train, y_train)
            y_pred = knn.predict(pkt_features_test)
            result['evaluation'] = [
                result['network'].evaluate(x_test, y_test, evaluation_batch_size, verbose=0)[0],
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
        trained_student, history = knowledge_transfer(clone_model(student), method['method'], method['loss'])
        # TODO model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        #  and save student model there, when we stop needing it,
        #  because it is inefficient to have it in memory until - if ever - we need to save it.
        #  That way, when the time comes, we will just need to move it to the out folder.
        results.append({
            'method': method['name'],
            'network': trained_student,
            'history': history.history,
            'evaluation': None
        })

    # Add baseline to the results list.
    results.append({
        'method': 'Teacher',
        'network': teacher,
        'history': None,
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
    teacher: Model = load_model(args.teacher, custom_objects={'Crop': Crop}, compile=False)
    student = load_model(args.student, compile=False)
    dataset: str = args.dataset
    kt_methods: Union[str, List[str]] = args.method
    attention = args.attention
    temperature: float = args.temperature
    kd_lambda_supervised: float = args.kd_lambda_supervised
    pkt_lambda_supervised: float = args.pkt_lambda_supervised
    k: int = args.neighbors
    kd_importance_weight: float = args.kd_importance_weight
    pkt_importance_weight: float = args.pkt_importance_weight
    use_best_model: bool = not args.use_final_model
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
    (x_train, y_train), (x_test, y_test), n_classes = load_data(dataset)

    # Preprocess data.
    kt_logging.info('Preprocessing data...')
    x_train, x_test = preprocess_data(dataset, x_train, x_test)
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)
    # Split data to train and val sets.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    # Adapt for Attention KT framework if needed.
    if attention:
        # Adapt for attention framework.
        kt_logging.info('Preparing Attention KT framework...')
        attention_teacher = teacher_adaptation(teacher)

        # Get attention teacher's outputs.
        kt_logging.info('Getting teacher\'s predictions...')
        y_teacher_train = attention_teacher.predict(x_train, evaluation_batch_size, verbosity)

        # Concatenate train labels as many times as the number of sub-teachers in the teacher model.
        y_train_concat = concatenate([y_train[:, newaxis] for _ in range(y_teacher_train.shape[1])], axis=1)
        # Concatenate teacher's outputs with true labels.
        y_train_concat = concatenate([y_train_concat, y_teacher_train], axis=2)
    else:
        # Get teacher's outputs.
        kt_logging.info('Getting teacher\'s predictions...')
        y_teacher_train = teacher.predict(x_train, evaluation_batch_size, verbosity)
        # Concatenate teacher's outputs with true labels.
        y_train_concat = concatenate([y_train, y_teacher_train], axis=1)

    y_teacher_val = teacher.predict(x_val, evaluation_batch_size, verbosity)
    y_val_concat = concatenate([y_val, y_teacher_val], axis=1)

    # Run kt.
    kt_logging.info('Starting KT method(s)...')
    run_kt_methods()

    # Show close message.
    kt_logging.info('Finished!')

    # Close logger.
    kt_logger.close_logger()
