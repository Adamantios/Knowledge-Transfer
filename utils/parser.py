from argparse import ArgumentParser

# ----------------------------------- DEFAULT ARGUMENTS ------------------------------------------

DEBUG = False
OUT_FOLDER_NAME = 'out'


# ------------------------------------------------------------------------------------------------


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser for the KT script.

    :return: ArgumentParser object.
    """
    parser = ArgumentParser(description='Transfer the knowledge between two Neural Networks, '
                                        'using different methods and compare the results.')
    parser.add_argument('teacher', type=str, help='Path to a teacher network.')
    parser.add_argument('-d', '--debug', default=DEBUG, required=False, action='store_true',
                        help='Whether debug mode should be enabled (default %(default)s).')
    parser.add_argument('-out', '--out_folder', default=OUT_FOLDER_NAME, required=False, type=str,
                        help='Path to the folder where the outputs will be stored (default %(default)s).')
    return parser
