import logging

logger = logging.getLogger('main')
hdlr = logging.FileHandler('logfile.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)