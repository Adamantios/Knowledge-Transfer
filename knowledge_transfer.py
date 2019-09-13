import logging

from core.metrics import available_losses


def knowledge_transfer() -> None:
    for loss in available_losses:
        logging.debug('Name: {}'.format(loss['name']))
        logging.debug('Function: {}'.format(loss['function']))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.info('An info message!')
    knowledge_transfer()
