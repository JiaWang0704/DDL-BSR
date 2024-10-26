from .dcls_arch import DCLS
from .discriminator_vgg_arch import Discriminator_VGG_128, VGGFeatureExtractor
from .RRDBNet_arch import RRDBNet
from .SRResNet_arch import MSRResNet
from .D2SR_arch import D2SR_s1
from .D2SR_arch import D2SR_s2

# Due to a slight error in the model Settings during training, we need to manually change the model loading
# The pretraining parameters of x2 load the model in D2SR arch wo feaextra, and the pretraining parameters of x4 load the model in D2SR arch

# from .D2SR_arch import D2SR_s3
from .dan_arch import DAN
from .danv1_arch import DANv1
from .danv2_arch import DANv2
# from .D2SR_arch_v1 import D2SR_s3

from .D2SR_arch_wo_feaextra import D2SR_s3