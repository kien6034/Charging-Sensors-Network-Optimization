import logging
targets = ["a", "b", "c"]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

for target in targets:
    log_file = "{}.log".format(target)
    log_format = "|%(levelname)s| : [%(filename)s]--[%(funcName)s] : %(message)s"
    formatter = logging.Formatter(log_format)

    # create file handler and set the formatter
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # add handler to the logger
    logger.addHandler(file_handler)

    # sample message
    logger.info("Log file: {}".format(target))
    logger.handlers.pop()