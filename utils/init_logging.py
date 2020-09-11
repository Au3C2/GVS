import logging  # 引入logging模块
import os.path
import time

def init_logging(starttime,log_file=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    
    if log_file:
        if not os.path.exists('./log'):
            os.mkdir('./log')
        fh = logging.FileHandler(f'./log/{starttime}.txt', mode='w')
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger