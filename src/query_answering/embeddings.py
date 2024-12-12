import torch as th
from box import Box
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def embedding_1p(data, class_embed, class_offset, rel_embed, scale_embed):
    c = class_embed(data[:, 0])
    c_offset = th.abs(class_offset(data[:, 0]))
    r = rel_embed(data[:, 1])
    r_scale = scale_embed(data[:, 1])
    return Box(c, c_offset, normalize=True).translate(r, r_scale)
                                     

def embedding_2p(data, class_embed, class_offset, rel_embed, scale_embed):
    c = class_embed(data[:, 0])
    c_offset = th.abs(class_offset(data[:, 0]))

    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])

    r_2 = rel_embed(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    return Box(c, c_offset, normalize=True).translate(r_1+r_2, r_1_scale*r_2_scale)
        

def embedding_3p(data, class_embed, class_offset, rel_embed, scale_embed):
    c = class_embed(data[:, 0])
    c_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    r_3 = rel_embed(data[:, 3])
    r_3_scale = scale_embed(data[:, 3])

    return Box(c, c_offset, normalize=True).translate(r_1+r_2+r_3, r_1_scale*r_2_scale*r_3_scale)



def embedding_2i(data, class_embed, class_offset, rel_embed, scale_embed):
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1, r_1_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_2, r_2_scale)
    return Box.intersection(box_c_1, box_c_2)



def embedding_3i(data, class_embed, class_offset, rel_embed, scale_embed):
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    c_3 = class_embed(data[:, 4])
    c_3_offset = th.abs(class_offset(data[:, 4]))
    r_3 = rel_embed(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1, r_1_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_2, r_2_scale)
    box_c_3 = Box(c_3, c_3_offset, normalize=True).translate(r_3, r_3_scale)

    return Box.intersection(box_c_1, box_c_2, box_c_3)



def embedding_2in(data, class_embed, class_offset, rel_embed, scale_embed):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1, r_1_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_2, r_2_scale)
    
    return Box.intersection_with_negation(2, box_c_1, box_c_2)
 

def embedding_3in(data, class_embed, class_offset, rel_embed, scale_embed):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    c_3 = class_embed(data[:, 4])
    c_3_offset = th.abs(class_offset(data[:, 4]))
    r_3 = rel_embed(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1, r_1_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_2, r_2_scale)
    box_c_3 = Box(c_3, c_3_offset, normalize=True).translate(r_3, r_3_scale)
    
    return Box.intersection_with_negation(3, box_c_1, box_c_2, box_c_3)



def embedding_ip(data, class_embed, class_offset, rel_embed, scale_embed):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_3 = rel_embed(data[:, 4])
    r_3_scale = scale_embed(data[:, 4])
     
    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1, r_1_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_2, r_2_scale)

    return Box.intersection(box_c_1, box_c_2).translate(r_3, r_3_scale)



def embedding_pi(data, class_embed, class_offset, rel_embed, scale_embed):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    c_2 = class_embed(data[:, 3])
    c_2_offset = th.abs(class_offset(data[:, 3]))
    r_3 = rel_embed(data[:, 4])
    r_3_scale = scale_embed(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1+r_2, r_1_scale*r_2_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_3, r_3_scale)
    return Box.intersection(box_c_1, box_c_2)



def embedding_inp(data, class_embed, class_offset, rel_embed, scale_embed):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    c_2 = class_embed(data[:, 2])
    c_2_offset = th.abs(class_offset(data[:, 2]))
    r_2 = rel_embed(data[:, 3])
    r_2_scale = scale_embed(data[:, 3])
    r_3 = rel_embed(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1, r_1_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_2, r_2_scale)
    return Box.intersection_with_negation(2, box_c_1, box_c_2).translate(r_3, r_3_scale)


def embedding_pin(data, class_embed, class_offset, rel_embed, scale_embed):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    c_2 = class_embed(data[:, 3])
    c_2_offset = th.abs(class_offset(data[:, 3]))
    r_3 = rel_embed(data[:, 4])
    r_3_scale = scale_embed(data[:, 4])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1+r_2, r_1_scale*r_2_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_3, r_3_scale)
    return Box.intersection_with_negation(2, box_c_1, box_c_2)


def embedding_pni(data, class_embed, class_offset, rel_embed, scale_embed):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    c_1 = class_embed(data[:, 0])
    c_1_offset = th.abs(class_offset(data[:, 0]))
    r_1 = rel_embed(data[:, 1])
    r_1_scale = scale_embed(data[:, 1])
    r_2 = rel_embed(data[:, 2])
    r_2_scale = scale_embed(data[:, 2])
    c_2 = class_embed(data[:, 4])
    c_2_offset = th.abs(class_offset(data[:, 4]))
    r_3 = rel_embed(data[:, 5])
    r_3_scale = scale_embed(data[:, 5])

    box_c_1 = Box(c_1, c_1_offset, normalize=True).translate(r_1+r_2, r_1_scale*r_2_scale)
    box_c_2 = Box(c_2, c_2_offset, normalize=True).translate(r_3, r_3_scale)

    return Box.intersection_with_negation(1, box_c_1, box_c_2)
        
