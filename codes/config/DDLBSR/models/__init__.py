import logging

logger = logging.getLogger("base")


def create_model(opt, netg2=None):
    model = opt["model"]
    if model == "srgan":
        from .SRGAN_model import SRGANModel as M
    elif model == "blind":
        from .blind_model import B_Model as M
    elif model == "stage2":
        from .Stage2_model import S2_Model as M
    elif model == "dan":
        from .danv1_model import B_Model as M
    elif model == "danv2":
        from danv2_model import B_Model as M
    elif model == "dcls":
        from .dcls_model import B_Model as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt, netg2)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
