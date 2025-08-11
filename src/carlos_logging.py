import logging
import os

def init_logger(log_file_path: str = None):
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    carlos_log = logging.getLogger("carlos_app")
    carlos_log.setLevel(logging.DEBUG)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        carlos_log.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    carlos_log.addHandler(console_handler)
    carlos_log.info("Logger initialized.")

def log_message(message: str, level: str = "info"):
    carlos_log = logging.getLogger("carlos_app")
    if carlos_log is None:
        raise ValueError("Logger is not initialized. Call init_logger() first.")
    if level == "debug":
        carlos_log.debug(message)
    elif level == "info":
        carlos_log.info(message)
    elif level == "warning":
        carlos_log.warning(message)
    elif level == "error":
        carlos_log.error(message)
    elif level == "critical":
        carlos_log.critical(message)
    else:
        raise ValueError(
            f"Invalid logging level: {level}. Use 'debug', 'info', 'warning', 'error', or 'critical'."
        )
