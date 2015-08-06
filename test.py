# -*- coding: utf-8 -*-
__author__ = 'tan'

import HMM

content = "这并不是一种最好的处理技术，因为这有可能低估或高估真实概率，更加科学的方法是使用复杂一点的Good—Turing技术，这项技术的原始版本是图灵当年和他的助手Good在破解德国密码机时发明的。"

seg = HMM.HMMSegment()
# seg.init_model(trainfile="test",load=True)
print(seg.segment_sentences(content))