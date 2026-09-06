# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Logging helpers."""

import logging
import sys
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        """Emit an INFO record through configured file handlers only.

        Honor logger and handler filters without changing console-handler levels.
        Handler owners remain responsible for closing their handlers.

        Args:
            msg: Message format string.
            *args: Positional values used for logging interpolation.
            exc_info: Exception instance, exception tuple, or flag requesting current exception details.
            stack_info: Whether to attach current stack information.
            stacklevel: Caller-frame offset used for record metadata.
            extra: Additional attributes merged into the logging record.

        Raises:
            KeyError: Extra attributes overwrite a reserved logging record attribute.

        """

        if not self.isEnabledFor(logging.INFO):
            return

        try:
            file_name, line_number, function, stack = self.findCaller(stack_info, stacklevel + 1)
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
        filtered_record = self.filter(record)
        if not filtered_record:
            return
        if isinstance(filtered_record, logging.LogRecord):
            record = filtered_record

        for handler in self.handlers:
            if isinstance(handler, logging.FileHandler) and record.levelno >= handler.level:
                handler.handle(record)


def get_console_handler() -> StreamHandler:
    """Create a stdout handler using the NALP formatter.

    Returns:
        A console handler owned by its caller or the logger to which it is attached.

    """

    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    return handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Create a delayed file handler that rotates NALP logs at midnight.

    The file is opened when the first record is emitted.

    Returns:
        A rotating file handler that its caller or owning logger must close.

    """

    handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    handler.setFormatter(FORMATTER)
    return handler


def get_logger(logger_name: str) -> Logger:
    """Return a named logger configured for NALP diagnostics.

    Set the default logger class for subsequently created loggers and retain existing handlers.
    Add NALP console and file handlers only when none exist, then disable propagation.

    Args:
        logger_name: Name identifying the logger.

    Returns:
        The named logger with the NALP level, handlers, and propagation setting.

    """

    logging.setLoggerClass(Logger)
    logger = logging.getLogger(logger_name)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_timed_file_handler())
    logger.propagate = False
    return logger
