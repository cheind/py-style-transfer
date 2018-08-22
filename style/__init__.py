# py-style-transfer
# Copyright 2018 Christoph Heindl.
# Licensed under MIT License
# ============================================================


from style.backbone import Backbone
from style.losses import Content, GramStyle, PatchStyle, SemanticStyle
from style.iterated import IteratedStyleTransfer
from style.tile import TiledGeneration
import style.image
import style.plugins


__version__ = '1.0.0'