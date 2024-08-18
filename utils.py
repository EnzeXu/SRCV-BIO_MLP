import datetime
import pytz
import numpy as np
import torch
import random


def get_timestring(time_string="%Y%m%d_%H%M%S_%f"):
    est = pytz.timezone('America/New_York')
    utc_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    est_now = utc_now.astimezone(est)
    return est_now.strftime(time_string)


def set_random_seeds(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multiple GPUs
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


