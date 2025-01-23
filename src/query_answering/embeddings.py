import torch as th
from box import Box
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def embedding_1p(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias, mask, r_idxs):
    # (('e', ('r',)),): '1p'
    c = class_embed(data[:, 0])
    c_offset = th.abs(class_offset(data[:, 0]))
    r_embed = rel_embed(data[:, 1])
    r_factor = rel_factor(data[:, 1])
    r_scale = scale_embed(data[:, 1])
    r_scale_bias = scale_bias(data[:, 1])
    return Box(c, c_offset).translate(r_embed, r_factor, r_scale, r_scale_bias, mask, r_idxs=r_idxs)
                                
def embedding_2p(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias, mask, r_idxs):
    c = class_embed(data[:, 0])
    c_offset = th.abs(class_offset(data[:, 0]))
    r_1_embed = rel_embed(data[:, 1])
    r_2_embed = rel_embed(data[:, 2])
    r_1_factor = rel_factor(data[:, 1])
    r_2_factor = rel_factor(data[:, 2])
    r_1_scale = scale_embed(data[:, 1])
    r_2_scale = scale_embed(data[:, 2])
    r_1_scale_bias = scale_bias(data[:, 1])
    r_2_scale_bias = scale_bias(data[:, 2])

    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scale_bias)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scale_bias, mask, r_idxs=r_idxs)
    return box
                                
def embedding_3p(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias, mask, r_idxs):
    c = class_embed(data[:, 0])
    c_offset = th.abs(class_offset(data[:, 0]))
    r_1_embed = rel_embed(data[:, 1])
    r_2_embed = rel_embed(data[:, 2])
    r_3_embed = rel_embed(data[:, 3])
    r_1_factor = rel_factor(data[:, 1])
    r_2_factor = rel_factor(data[:, 2])
    r_3_factor = rel_factor(data[:, 3])
    r_1_scale = scale_embed(data[:, 1])
    r_2_scale = scale_embed(data[:, 2])
    r_3_scale = scale_embed(data[:, 3])
    r_1_scale_bias = scale_bias(data[:, 1])
    r_2_scale_bias = scale_bias(data[:, 2])
    r_3_scale_bias = scale_bias(data[:, 3])
    
    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scale_bias)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scale_bias)
    box = box.translate(r_3_embed, r_3_factor, r_3_scale, r_3_scale_bias, mask, r_idxs=r_idxs)
    return box


def embedding_2i(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_factor = rel_factor(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_2_scale_bias = scale_bias(data[:, 3])
    
    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    return Box.intersection(box_c_1, box_c_2)

def embedding_3i(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_factor = rel_factor(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_2_scale_bias = scale_bias(data[:, 3])
    c_3 = class_embed(data[:, 4])
    c_3_offset = th.abs(class_offset(data[:, 4]))
    r_3 = rel_embed(data[:, 5])
    r_3_factor = rel_factor(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])
    r_3_scale_bias = scale_bias(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    box_c_3 = Box(c_3, c_3_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scale_bias)

    return Box.intersection(box_c_1, box_c_2, box_c_3)



def embedding_2in(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_factor = rel_factor(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_2_scale_bias = scale_bias(data[:, 3])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    
    return Box.intersection_with_negation(2, box_c_1, box_c_2)
 

def embedding_3in(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_factor = rel_factor(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_2_scale_bias = scale_bias(data[:, 3])
    c_3 = class_embed(data[:, 4])
    c_3_offset = th.abs(class_offset(data[:, 4]))
    r_3 = rel_embed(data[:, 5])
    r_3_factor = rel_factor(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])
    r_3_scale_bias = scale_bias(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    box_c_3 = Box(c_3, c_3_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scale_bias)
    
    return Box.intersection_with_negation(3, box_c_1, box_c_2, box_c_3)



def embedding_ip(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias, mask, r_idxs):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1_embed = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2_embed = rel_embed(data[:, 3])
    r_2_factor = rel_factor(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_2_scale_bias = scale_bias(data[:, 3])
    r_3_embed = rel_embed(data[:, 4])
    r_3_factor = rel_factor(data[:, 4])
    r_3_scale = scale_embed(data[:, 4])
    r_3_scale_bias = scale_bias(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scale_bias)

    box = Box.intersection(box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scale_bias, mask, r_idxs=r_idxs)
    
    return box



def embedding_pi(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_factor = rel_factor(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    r_2_scale_bias = scale_bias(data[:, 2])
    c_2 = class_embed(data[:, 3])
    c_2_offset = th.abs(class_offset(data[:, 3]))
    r_3 = rel_embed(data[:, 4])
    r_3_factor = rel_factor(data[:, 4])
    r_3_scale = scale_embed(data[:, 4])
    r_3_scale_bias = scale_bias(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_1 = box_c_1.translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scale_bias)
    return Box.intersection(box_c_1, box_c_2)



def embedding_inp(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias, mask, r_idxs):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1_embed = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2_embed = rel_embed(data[:, 3])
    r_2_factor = rel_factor(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_2_scale_bias = scale_bias(data[:, 3])
    r_3_embed = rel_embed(data[:, 5])
    r_3_factor = rel_factor(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])
    r_3_scale_bias = scale_bias(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scale_bias)

    box = Box.intersection_with_negation(2, box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scale_bias, mask, r_idxs=r_idxs)
    return box
                                                                
def embedding_pin(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_factor = rel_factor(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    r_2_scale_bias = scale_bias(data[:, 2])
    c_2 = class_embed(data[:, 3])
    c_2_offset = th.abs(class_offset(data[:, 3]))
    r_3 = rel_embed(data[:, 4])
    r_3_factor = rel_factor(data[:, 4])
    r_3_scale = scale_embed(data[:, 4])
    r_3_scale_bias = scale_bias(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_1 = box_c_1.translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scale_bias)
    return Box.intersection_with_negation(2, box_c_1, box_c_2)


def embedding_pni(data, class_embed, class_offset, rel_embed, rel_factor, scale_embed, scale_bias):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_factor = rel_factor(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_1_scale_bias = scale_bias(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_factor = rel_factor(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    r_2_scale_bias = scale_bias(data[:, 2])
    c_2 = class_embed(data[:, 4])
    c_2_offset = th.abs(class_offset(data[:, 4]))
    r_3 = rel_embed(data[:, 5])
    r_3_factor = rel_factor(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])
    r_3_scale_bias = scale_bias(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scale_bias)
    box_c_1 = box_c_1.translate(r_2, r_2_factor, r_2_scale, r_2_scale_bias)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scale_bias)

    return Box.intersection_with_negation(1, box_c_1, box_c_2)
        
