import logging
import os.path
import time

def init_logging(starttime,log_file=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  

    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    
    if log_file:
        if not os.path.exists('./log'):
            os.mkdir('./log')
        fh = logging.FileHandler(f'./log/{starttime}.txt', mode='w')
        fh.setLevel(logging.INFO)  
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger