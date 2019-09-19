from argparse import ArgumentParser

# ----------------------------------- DEFAULT ARGUMENTS ------------------------------------------

STUDENT_CHOICES = {'cifar10_tiny_1'}
DATASET_CHOICES = {'cifar10', 'cifar100'}
TEMPERATURE = 0.1
LAMBDA_SUPERVISED = 0.1
SAVE_STUDENTS = 'best'
SAVE_STUDENTS_CHOICES = 'all', 'best', 'none'
SAVE_RESULTS = True
OUT_FOLDER_NAME = 'out'
OPTIMIZER = 'rmsprop'
OPTIMIZER_CHOICES = 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'
LEARNING_RATE = 1E-3
LR_PATIENCE = 8
LR_DECAY = 0.1
LR_MIN = 0.00000001
EARLY_STOPPING_PATIENCE = 15
CLIP_NORM = 1
CLIP_VALUE = .5
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


# ------------------------------------------------------------------------------------------------


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the KT script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Transfer the knowledge between two Neural Networks, '
                                        'using different methods and compare the results.')
    parser.add_argument('teacher', type=str, help='Path to a trained teacher network.')
    parser.add_argument('student', type=str, choices=STUDENT_CHOICES, help='Path to a student network.')
    parser.add_argument('dataset', type=str, choices=DATASET_CHOICES, help='The name of the dataset to be used.')
    parser.add_argument('-w', '--start_weights', type=str, required=False,
                        help='Filepath containing existing weights to initialize the model.')
    parser.add_argument('-t', '--temperature', default=TEMPERATURE, required=False, type=float,
                        help='The temperature for the distillation (default %(default)s).')
    parser.add_argument('-l', '--lambda_supervised', default=LAMBDA_SUPERVISED, required=False, type=float,
                        help='The lambda value for the supervised term (default %(default)s).')
    parser.add_argument('-s', '--save_students', type=str.lower, default=SAVE_STUDENTS, required=False,
                        choices=SAVE_STUDENTS_CHOICES,
                        help='The save mode for the final student networks. (default %(default)s).')
    parser.add_argument('-or', '--omit_results', default=not SAVE_RESULTS, required=False, action='store_true',
                        help='Whether the KT comparison results should not be saved (default %(default)s).')
    parser.add_argument('-o', '--out_folder', default=OUT_FOLDER_NAME, required=False, type=str,
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
                        help='The number of epochs to wait before early stopping'
                             'If 0 is given, early stopping will not be applied. (default %(default)s).')
    parser.add_argument('-cn', '--clip_norm', type=float, default=CLIP_NORM, required=False,
                        help='The clip norm for the optimizer (default %(default)s).')
    parser.add_argument('-cv', '--clip_value', type=float, default=CLIP_VALUE, required=False,
                        help='The clip value for the optimizer (default %(default)s).')
    parser.add_argument('-b1', '--beta1', type=float, default=BETA1, required=False,
                        help='The beta 1 for the optimizer (default %(default)s).')
    parser.add_argument('-b2', '--beta2', type=float, default=BETA2, required=False,
                        help='The beta 2 for the optimizer (default %(default)s).')
    parser.add_argument('-rho', type=float, default=RHO, required=False,
                        help='The rho for the optimizer (default %(default)s).')
    parser.add_argument('-m', '--momentum', type=float, default=MOMENTUM, required=False,
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
    parser.add_argument('-d', '--debug', default=DEBUG, required=False, action='store_true',
                        help='Whether debug mode should be enabled (default %(default)s).')
    return parser
