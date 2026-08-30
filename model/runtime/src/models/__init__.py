
from .aasist import AASISTMultitask
try:
    from .ssl_backbone import SSLMultitask
except: SSLMultitask=None
try:
    from .beats_backbone import MusicMultitask, FusionModel, SpecCNNBackbone
except: MusicMultitask=None; FusionModel=None; SpecCNNBackbone=None
try:
    from .panns import PANNsPresenceWrapper, PANNsCNN14, Cnn14
except: PANNsPresenceWrapper=None; PANNsCNN14=None; Cnn14=None
try:
    from .demucs_wrapper import HTDemucsSeparator, get_separator, separate_vocals_music
except: HTDemucsSeparator=None; get_separator=None; separate_vocals_music=None
try:
    from .multitask import EnsembleModel, create_model
except: EnsembleModel=None; create_model=None
__all__ = ['AASISTMultitask','SSLMultitask','MusicMultitask','FusionModel','SpecCNNBackbone','PANNsPresenceWrapper','PANNsCNN14','Cnn14','HTDemucsSeparator','get_separator','separate_vocals_music','EnsembleModel','create_model']
