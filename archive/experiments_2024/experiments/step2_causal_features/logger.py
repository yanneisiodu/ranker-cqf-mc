import logging
import sys

def setup_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create a handler that writes log messages to stdout.
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create a formatter and add it to the handler.
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    
    # Avoid adding duplicate handlers.
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger 