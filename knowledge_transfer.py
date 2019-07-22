import logging
import sys

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.info('An info message!')
    logging.debug('A debug message!')
