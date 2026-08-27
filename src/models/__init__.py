
from .aasist import AASISTMultitask
try:
    from .ssl_backbone import SSLMultitask
except: SSLMultitask=None
try:
    from .beats_backbone import MusicMultitask, FusionModel
except: MusicMultitask=None; FusionModel=None
try:
    from .multitask import EnsembleModel, create_model
except: EnsembleModel=None; create_model=None
__all__ = ['AASISTMultitask','SSLMultitask','MusicMultitask','FusionModel','EnsembleModel','create_model']
