import logging

from tensorflow.python.keras.saving import load_model

from core.losses import available_methods
from utils.parser import create_parser


def knowledge_transfer(method) -> None:
    pass


def evaluate_results(results):
    pass


def compare_kt_methods():
    results = []
    for method in available_methods:
        logging.debug('Name: {}'.format(method['name']))
        logging.debug('Function: {}'.format(method['function']))
        results.append(knowledge_transfer(method))

    evaluate_results(results)


if __name__ == '__main__':
    # Get arguments.
    args = create_parser().parse_args()
    teacher_filename = args.teacher
    teacher = load_model(teacher_filename)
    debug = args.debug
    out_folder = args.out_folder

    # Set up logger.
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=level)

    # Run comparison.
    compare_kt_methods()
