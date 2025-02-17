import torch as th
from box import Box
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def embedding_1p(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r',)),): '1p'
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    
    r_embed = translation_add(data[:, 1])
    r_factor = translation_mul(data[:, 1])
    r_scale = scaling_mul(data[:, 1])
    r_scaling_add = scaling_add(data[:, 1])
    return Box(c, c_offset).translate(r_embed, r_factor, r_scale, r_scaling_add)

def embedding_1pt(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r',)),): '1p'
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    
    r_embed = th.abs(translation_add(data[:, 1]))
    r_factor = th.tensor(1)
    r_scale = th.tensor(1)
    r_scaling_add = th.tensor(0)
    return Box(c, c_offset).translate(r_embed, r_factor, r_scale, r_scaling_add)

def embedding_1pi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r',)),): '1p'
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    
    r_embed = -th.abs(translation_add(data[:, 1]))
    r_factor = th.tensor(1)
    r_scale = th.tensor(1)
    r_scaling_add = th.tensor(0)
    return Box(c, c_offset).translate(r_embed, r_factor, r_scale, r_scaling_add)


def embedding_2p(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_2_embed = translation_add(data[:, 2])
    r_1_factor = translation_mul(data[:, 1])
    r_2_factor = translation_mul(data[:, 2])
    r_1_scale = scaling_mul(data[:, 1])
    r_2_scale = scaling_mul(data[:, 2])
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2_scaling_add = scaling_add(data[:, 2])

    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)
    return box

def embedding_2pt(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_2_embed = th.abs(translation_add(data[:, 2]))
    r_1_factor = translation_mul(data[:, 1])
    r_2_factor = th.tensor(1)
    r_1_scale = scaling_mul(data[:, 1])
    r_2_scale = th.tensor(1)
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2_scaling_add = th.tensor(0)

    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)
    return box

def embedding_2pi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_2_embed = -th.abs(translation_add(data[:, 2]))
    r_1_factor = translation_mul(data[:, 1])
    r_2_factor = th.tensor(1)
    r_1_scale = scaling_mul(data[:, 1])
    r_2_scale = th.tensor(1)
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2_scaling_add = th.tensor(0)

    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)
    return box



def embedding_3p(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    r_ids = data[:, 1]
    r_1_embed = translation_add(data[:, 1])
    r_2_embed = translation_add(data[:, 2])
    r_3_embed = translation_add(data[:, 3])
    r_1_factor = translation_mul(data[:, 1])
    r_2_factor = translation_mul(data[:, 2])
    r_3_factor = translation_mul(data[:, 3])
    r_1_scale = scaling_mul(data[:, 1])
    r_2_scale = scaling_mul(data[:, 2])
    r_3_scale = scaling_mul(data[:, 3])
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2_scaling_add = scaling_add(data[:, 2])
    r_3_scaling_add = scaling_add(data[:, 3])
    
    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)
    box = box.translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    return box

def embedding_3pt(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_2_embed = translation_add(data[:, 2])
    r_3_embed = th.abs(translation_add(data[:, 3]))
    r_1_factor = translation_mul(data[:, 1])
    r_2_factor = translation_mul(data[:, 2])
    r_3_factor = th.tensor(1)
    r_1_scale = scaling_mul(data[:, 1])
    r_2_scale = scaling_mul(data[:, 2])
    r_3_scale = th.tensor(1)
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2_scaling_add = scaling_add(data[:, 2])
    r_3_scaling_add = th.tensor(0)
    
    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)
    box = box.translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    return box

def embedding_3pi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c = center_embed(data[:, 0])
    c_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_2_embed = translation_add(data[:, 2])
    r_3_embed = -th.abs(translation_add(data[:, 3]))
    r_1_factor = translation_mul(data[:, 1])
    r_2_factor = translation_mul(data[:, 2])
    r_3_factor = th.tensor(1)
    r_1_scale = scaling_mul(data[:, 1])
    r_2_scale = scaling_mul(data[:, 2])
    r_3_scale = th.tensor(1)
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2_scaling_add = scaling_add(data[:, 2])
    r_3_scaling_add = th.tensor(0)
    
    box = Box(c, c_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)
    box = box.translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    return box

def embedding_2i(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2 = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    
    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    return Box.intersection(box_c_1, box_c_2)

def embedding_3i(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2 = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    c_3 = center_embed(data[:, 4])
    c_3_offset = th.abs(offset_embed(data[:, 4]))
    r_3 = translation_add(data[:, 5])
    r_3_factor = translation_mul(data[:, 5])
    r_3_scale = scaling_mul(data[:, 5])
    r_3_scaling_add = scaling_add(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    box_c_3 = Box(c_3, c_3_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scaling_add)

    return Box.intersection(box_c_1, box_c_2, box_c_3)



def embedding_2in(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2 = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    
    return Box.intersection_with_negation(2, box_c_1, box_c_2)
 

def embedding_3in(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2 = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    c_3 = center_embed(data[:, 4])
    c_3_offset = th.abs(offset_embed(data[:, 4]))
    r_3 = translation_add(data[:, 5])
    r_3_factor = translation_mul(data[:, 5])
    r_3_scale = scaling_mul(data[:, 5])
    r_3_scaling_add = scaling_add(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    box_c_3 = Box(c_3, c_3_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scaling_add)
    
    return Box.intersection_with_negation(3, box_c_1, box_c_2, box_c_3)

def embedding_ip(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2_embed = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    r_3_embed = translation_add(data[:, 4])
    r_3_factor = translation_mul(data[:, 4])
    r_3_scale = scaling_mul(data[:, 4])
    r_3_scaling_add = scaling_add(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)

    box = Box.intersection(box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    
    return box

def embedding_ipt(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2_embed = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    r_3_embed = th.abs(translation_add(data[:, 4]))
    r_3_factor = th.tensor(1)
    r_3_scale = th.tensor(1)
    r_3_scaling_add = th.tensor(0)

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)

    box = Box.intersection(box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    
    return box

def embedding_ipi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2_embed = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    r_3_embed = -th.abs(translation_add(data[:, 4]))
    r_3_factor = th.tensor(1)
    r_3_scale = th.tensor(1)
    r_3_scaling_add = th.tensor(0)

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)

    box = Box.intersection(box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    
    return box



def embedding_pi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2 = translation_add(data[:, 2])
    r_2_factor = translation_mul(data[:, 2])
    r_2_scale = scaling_mul(data[:, 2])
    r_2_scaling_add = scaling_add(data[:, 2])
    c_2 = center_embed(data[:, 3])
    c_2_offset = th.abs(offset_embed(data[:, 3]))
    r_3 = translation_add(data[:, 4])
    r_3_factor = translation_mul(data[:, 4])
    r_3_scale = scaling_mul(data[:, 4])
    r_3_scaling_add = scaling_add(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_1 = box_c_1.translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scaling_add)
    return Box.intersection(box_c_1, box_c_2)



def embedding_inp(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2_embed = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    r_3_embed = translation_add(data[:, 5])
    r_3_factor = translation_mul(data[:, 5])
    r_3_scale = scaling_mul(data[:, 5])
    r_3_scaling_add = scaling_add(data[:, 5])
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)

    box = Box.intersection_with_negation(2, box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    return box

def embedding_inpt(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2_embed = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    r_3_embed = th.abs(translation_add(data[:, 5]))
    r_3_factor = th.tensor(1)
    r_3_scale = th.tensor(1)
    r_3_scaling_add = th.tensor(0)
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)

    box = Box.intersection_with_negation(2, box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    return box

def embedding_inpi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1_embed = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    c_2 = center_embed(data[:, 2])
    c_2_offset = th.abs(offset_embed(data[:, 2]))
    r_2_embed = translation_add(data[:, 3])
    r_2_factor = translation_mul(data[:, 3])
    r_2_scale = scaling_mul(data[:, 3])
    r_2_scaling_add = scaling_add(data[:, 3])
    r_3_embed = -th.abs(translation_add(data[:, 5]))
    r_3_factor = th.tensor(1)
    r_3_scale = th.tensor(1)
    r_3_scaling_add = th.tensor(0)
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_embed, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_embed, r_2_factor, r_2_scale, r_2_scaling_add)

    box = Box.intersection_with_negation(2, box_c_1, box_c_2).translate(r_3_embed, r_3_factor, r_3_scale, r_3_scaling_add)
    return box


def embedding_pin(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2 = translation_add(data[:, 2])
    r_2_factor = translation_mul(data[:, 2])
    r_2_scale = scaling_mul(data[:, 2])
    r_2_scaling_add = scaling_add(data[:, 2])
    c_2 = center_embed(data[:, 3])
    c_2_offset = th.abs(offset_embed(data[:, 3]))
    r_3 = translation_add(data[:, 4])
    r_3_factor = translation_mul(data[:, 4])
    r_3_scale = scaling_mul(data[:, 4])
    r_3_scaling_add = scaling_add(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_1 = box_c_1.translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scaling_add)
    return Box.intersection_with_negation(2, box_c_1, box_c_2)


def embedding_pni(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    c_1 = center_embed(data[:, 0])
    c_1_offset = th.abs(offset_embed(data[:, 0]))
    r_1 = translation_add(data[:, 1])
    r_1_factor = translation_mul(data[:, 1])
    r_1_scale = scaling_mul(data[:, 1])
    r_1_scaling_add = scaling_add(data[:, 1])
    r_2 = translation_add(data[:, 2])
    r_2_factor = translation_mul(data[:, 2])
    r_2_scale = scaling_mul(data[:, 2])
    r_2_scaling_add = scaling_add(data[:, 2])
    c_2 = center_embed(data[:, 4])
    c_2_offset = th.abs(offset_embed(data[:, 4]))
    r_3 = translation_add(data[:, 5])
    r_3_factor = translation_mul(data[:, 5])
    r_3_scale = scaling_mul(data[:, 5])
    r_3_scaling_add = scaling_add(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1, r_1_factor, r_1_scale, r_1_scaling_add)
    box_c_1 = box_c_1.translate(r_2, r_2_factor, r_2_scale, r_2_scaling_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3, r_3_factor, r_3_scale, r_3_scaling_add)

    return Box.intersection_with_negation(1, box_c_1, box_c_2)
        
