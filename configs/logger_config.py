import logging

def setup_logger(name="root"):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%H:%M:%S')

    handler = logging.StreamHandler()  # Print to console
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

# Create global logger
global_logger = setup_logger("global_logger")
