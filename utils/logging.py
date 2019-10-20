import logging


class KTLogger(object):
    def __init__(self, filepath: str):
        """
        :param filepath: the filepath for the logs.
        """
        self.kt_logger = logging.getLogger('KT')
        self.file_handler = logging.FileHandler(filepath)
        self.console_handler = logging.StreamHandler()

    def setup_logger(self, debug: bool, save: bool) -> None:
        """
        Sets the program's logger up.

        :param debug: Whether the logger should be set in debugging mode.
        :param save: whether the logs should be saved to a file.
        """
        level = logging.DEBUG if debug else logging.INFO

        log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

        if save:
            self.file_handler.setFormatter(log_formatter)
            self.kt_logger.addHandler(self.file_handler)

        self.console_handler.setFormatter(log_formatter)
        self.kt_logger.addHandler(self.console_handler)
        self.kt_logger.setLevel(level)

    def close_logger(self) -> None:
        """Closes the logger."""
        self.file_handler.close()
        self.kt_logger.removeHandler(self.file_handler)
        del self.file_handler
        self.console_handler.close()
        self.kt_logger.removeHandler(self.console_handler)
        del self.console_handler
