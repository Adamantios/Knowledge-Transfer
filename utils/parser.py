from argparse import ArgumentParser

# ----------------------------------- DEFAULT ARGUMENTS ------------------------------------------
DATASET_CHOICES = 'cifar10', 'cifar100', 'svhn_cropped', 'fashion_mnist', 'mnist'
METHOD = ['distillation', 'pkt', 'pkt+distillation']
METHOD_CHOICES = 'distillation', 'pkt', 'pkt+distillation'
SELECTIVE_LEARNING = False
TEMPERATURE = 2
KD_LAMBDA_SUPERVISED = 0.1
PKT_LAMBDA_SUPERVISED = 1E-4
K = 5
KD_IMPORTANCE_WEIGHT = 1
PKT_IMPORTANCE_WEIGHT = 1
KEEP_BEST = True
SAVE_STUDENTS = 'best'
SAVE_STUDENTS_CHOICES = 'all', 'best', 'none'
SAVE_RESULTS = True
RESULTS_NAME_PREFIX = ''
OUT_FOLDER_NAME = 'out'
OPTIMIZER = 'adam'
OPTIMIZER_CHOICES = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
LEARNING_RATE = 1E-3
LR_PATIENCE = 8
LR_DECAY = 0.1
LR_MIN = 1E-8
EARLY_STOPPING_PATIENCE = 15
BETA1 = .9
BETA2 = .999
RHO = .9
MOMENTUM = .0
DECAY = 1E-6
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
EPOCHS = 125
VERBOSITY = 1
DEBUG = False
SEED = 0


# ------------------------------------------------------------------------------------------------


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the KT script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Transfer the knowledge between two Neural Networks, '
                                        'using different methods and compare the results.')
    parser.add_argument('teacher', type=str, help='Path to a trained teacher network.')
    parser.add_argument('student', type=str, help='Path to a student network to be used.')
    parser.add_argument('dataset', type=str, choices=DATASET_CHOICES, help='The name of the dataset to be used.')
    parser.add_argument('-m', '--method', type=str.lower, nargs='+', default=METHOD, required=False,
                        choices=METHOD_CHOICES, help='The KT method(s) to be used. (default %(default)s).')
    parser.add_argument('-sl', '--selective_learning', default=SELECTIVE_LEARNING, required=False, action='store_true',
                        help='Whether the models should be designed for the KT with Selective Learning framework '
                             '(default %(default)s).')
    parser.add_argument('-w', '--start_weights', type=str, required=False,
                        help='Filepath containing existing weights to initialize the model.')
    parser.add_argument('-t', '--temperature', default=TEMPERATURE, required=False, type=float,
                        help='The temperature for the distillation (default %(default)s).')
    parser.add_argument('-kdl', '--kd_lambda_supervised', default=KD_LAMBDA_SUPERVISED, required=False, type=float,
                        help='The lambda value for the KD supervised term (default %(default)s).')
    parser.add_argument('-pktl', '--pkt_lambda_supervised', default=PKT_LAMBDA_SUPERVISED, required=False, type=float,
                        help='The lambda value for the PKT supervised term (default %(default)s).')
    parser.add_argument('-k', '--neighbors', default=K, required=False, type=int,
                        help='The number of neighbors for the PKT method evaluation (default %(default)s).')
    parser.add_argument('-kdw', '--kd_importance_weight', default=KD_IMPORTANCE_WEIGHT, required=False, type=float,
                        help='The importance weight for the KD loss, if method is PKT plus KD (default %(default)s).')
    parser.add_argument('-pktw', '--pkt_importance_weight', default=PKT_IMPORTANCE_WEIGHT, required=False, type=float,
                        help='The importance weight for the PKT loss, if method is PKT plus KD (default %(default)s).')
    parser.add_argument('-ufm', '--use_final_model', default=not KEEP_BEST, required=False, action='store_true',
                        help='Whether the final model should be used for saving and results evaluation '
                             'and not the best one achieved through the training procedure (default %(default)s).')
    parser.add_argument('-s', '--save_students', type=str.lower, default=SAVE_STUDENTS, required=False,
                        choices=SAVE_STUDENTS_CHOICES,
                        help='The save mode for the final student networks. (default %(default)s).')
    parser.add_argument('-or', '--omit_results', default=not SAVE_RESULTS, required=False, action='store_true',
                        help='Whether the KT comparison results should not be saved (default %(default)s).')
    parser.add_argument('-res', '--results_name_prefix', default=RESULTS_NAME_PREFIX, required=False, type=str,
                        help='The prefix for the results filenames (default %(default)s).')
    parser.add_argument('-out', '--out_folder', default=OUT_FOLDER_NAME, required=False, type=str,
                        help='Path to the folder where the outputs will be stored (default %(default)s).')
    parser.add_argument('-o', '--optimizer', type=str.lower, default=OPTIMIZER, required=False,
                        choices=OPTIMIZER_CHOICES,
                        help='The optimizer to be used. (default %(default)s).')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, required=False,
                        help='The learning rate for the optimizer (default %(default)s).')
    parser.add_argument('-lrp', '--learning_rate_patience', type=int, default=LR_PATIENCE, required=False,
                        help='The number of epochs to wait before decaying the learning rate (default %(default)s).')
    parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=LR_DECAY, required=False,
                        help='The learning rate decay factor. '
                             'If 0 is given, then the learning rate will remain the same during the training process. '
                             '(default %(default)s).')
    parser.add_argument('-lrm', '--learning_rate_min', type=float, default=LR_MIN, required=False,
                        help='The minimum learning rate which can be reached (default %(default)s).')
    parser.add_argument('-esp', '--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, required=False,
                        help='The number of epochs to wait before early stopping. '
                             'If 0 is given, early stopping will not be applied. (default %(default)s).')
    parser.add_argument('-cn', '--clip_norm', type=float, required=False,
                        help='The clip norm for the optimizer (default %(default)s).')
    parser.add_argument('-cv', '--clip_value', type=float, required=False,
                        help='The clip value for the optimizer (default %(default)s).')
    parser.add_argument('-b1', '--beta1', type=float, default=BETA1, required=False,
                        help='The beta 1 for the optimizer (default %(default)s).')
    parser.add_argument('-b2', '--beta2', type=float, default=BETA2, required=False,
                        help='The beta 2 for the optimizer (default %(default)s).')
    parser.add_argument('-rho', type=float, default=RHO, required=False,
                        help='The rho for the optimizer (default %(default)s).')
    parser.add_argument('-mm', '--momentum', type=float, default=MOMENTUM, required=False,
                        help='The momentum for the optimizer (default %(default)s).')
    parser.add_argument('-d', '--decay', type=float, default=DECAY, required=False,
                        help='The decay for the optimizer (default %(default)s).')
    parser.add_argument('-bs', '--batch_size', type=int, default=TRAIN_BATCH_SIZE, required=False,
                        help='The batch size for the optimization (default %(default)s).')
    parser.add_argument('-ebs', '--evaluation_batch_size', type=int, default=EVAL_BATCH_SIZE, required=False,
                        help='The batch size for the evaluation (default %(default)s).')
    parser.add_argument('-e', '--epochs', type=int, default=EPOCHS, required=False,
                        help='The number of epochs to train the network (default %(default)s).')
    parser.add_argument('-v', '--verbosity', type=int, default=VERBOSITY, required=False,
                        help='The verbosity for the optimization procedure (default %(default)s).')
    parser.add_argument('--debug', default=DEBUG, required=False, action='store_true',
                        help='Whether debug mode should be enabled (default %(default)s).')
    parser.add_argument('-seed', '--seed', type=int, default=SEED, required=False,
                        help='The seed for all the random operations. Pass a negative number, '
                             'in order to have non-deterministic behavior (default %(default)s).')
    return parser

# TODO use Hydra (https://hydra.cc/) instead.
