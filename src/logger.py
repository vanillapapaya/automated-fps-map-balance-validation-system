"""
Logging Configuration Module

Provides centralized logging setup for the entire map generation system.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class LoggerConfig:
    """
    Centralized logging configuration for the map validation system.
    """

    _initialized = False
    _log_dir = "logs"
    _log_level = logging.INFO

    @classmethod
    def setup_logging(cls,
                     log_dir: str = "logs",
                     log_level: int = logging.INFO,
                     console_output: bool = True,
                     file_output: bool = True) -> logging.Logger:
        """
        Setup the logging system with file and console handlers.

        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file

        Returns:
            Root logger instance
        """
        if cls._initialized:
            return logging.getLogger()

        cls._log_dir = log_dir
        cls._log_level = log_level

        # Create log directory if it doesn't exist
        if file_output:
            os.makedirs(log_dir, exist_ok=True)

        # Create root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # Clear any existing handlers
        logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)

        # File handler - detailed log
        if file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"map_generation_{timestamp}.log")

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

            # Also create a latest.log symlink/copy
            latest_log = os.path.join(log_dir, "latest.log")
            latest_handler = logging.FileHandler(latest_log, mode='w', encoding='utf-8')
            latest_handler.setLevel(logging.DEBUG)
            latest_handler.setFormatter(detailed_formatter)
            logger.addHandler(latest_handler)

        cls._initialized = True

        logger.info("=" * 80)
        logger.info("Logging system initialized")
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Log level: {logging.getLevelName(log_level)}")
        logger.info("=" * 80)

        return logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.

        Args:
            name: Name of the module (typically __name__)

        Returns:
            Logger instance
        """
        if not cls._initialized:
            cls.setup_logging()

        return logging.getLogger(name)

    @classmethod
    def set_level(cls, level: int):
        """
        Change the logging level for all loggers.

        Args:
            level: New logging level
        """
        logging.getLogger().setLevel(level)
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        Logger instance
    """
    return LoggerConfig.get_logger(name)


class PerformanceLogger:
    """
    Context manager for logging performance metrics.
    """

    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance to use
            operation: Name of the operation being timed
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        """Start timing the operation."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log the duration."""
        import time
        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} (took {duration:.3f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} (took {duration:.3f}s) - {exc_val}")


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments and results.

    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args[:3] if len(args) > 3 else args}, kwargs={kwargs}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {e}", exc_info=True)
                raise

        return wrapper
    return decorator
