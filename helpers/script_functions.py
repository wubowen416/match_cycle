from argparse import Namespace


def fix_seed(seed):
    """
    ported from https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031 
    """
    import random
    import torch
    import numpy as np
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_dict_to_namespace(namespace: Namespace, default_dict: dict):
    namespace.__dict__.update(**default_dict)
    return namespace