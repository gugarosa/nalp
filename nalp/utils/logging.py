"""Logging helpers."""

import logging
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = "nalp.log"
LOG_LEVEL = logging.DEBUG


class Logger(logging.Logger):
    """Logger with an explicit file-only write method."""

    def to_file(
        self,
        msg: str,
        *args,
        exc_info=None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra=None,
    ) -> None:
        """Log ``msg`` only through configured file handlers."""

        if not self.isEnabledFor(logging.INFO):
            return

        try:
            file_name, line_number, function, stack = self.findCaller(
                stack_info, stacklevel + 1
            )
        except ValueError:
            file_name, line_number, function, stack = (
                "(unknown file)",
                0,
                "(unknown function)",
                None,
            )

        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()

        record = self.makeRecord(
            self.name,
            logging.INFO,
            file_name,
            line_number,
            msg,
            args,
            exc_info,
            function,
            extra,
            stack,
        )
        if not self.filter(record):
            return

        for handler in self.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and record.levelno >= handler.level
            ):
                handler.handle(record)


def get_console_handler() -> StreamHandler:
    """Return the configured console handler."""

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    return handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Return the configured rotating file handler."""

    handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    handler.setFormatter(FORMATTER)
    return handler


def get_logger(logger_name: str) -> Logger:
    """Return an idempotently configured NALP logger."""

    logging.setLoggerClass(Logger)
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
    logger.propagate = False
    return logger
