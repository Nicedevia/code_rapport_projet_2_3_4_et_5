# --- logs/log_config.py ---
import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_dir: str = "logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler pour les logs INFO+ (INFO, WARNING, ERROR, CRITICAL)
    info_handler = RotatingFileHandler(
        os.path.join(log_dir, "app_info.log"),
        maxBytes=1_000_000,
        backupCount=5
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    # Handler pour les logs ERROR+ (ERROR, CRITICAL uniquement)
    error_handler = RotatingFileHandler(
        os.path.join(log_dir, "app_error.log"),
        maxBytes=1_000_000,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Pour éviter les doublons si déjà configuré
    if not logger.hasHandlers():
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)

    return logger
