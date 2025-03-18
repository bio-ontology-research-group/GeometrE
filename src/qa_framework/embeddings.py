import torch as th
from box import Box
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


id_mul = th.tensor(1)
id_add = th.tensor(0)
empty_tensor = th.tensor([])
simple_transf_data = lambda x: (id_mul, x, id_mul, id_add)

def get_box_data(box_data, index_tensor):
    center_embed, offset_embed = box_data
    center = center_embed(index_tensor)
    offset = th.abs(offset_embed(index_tensor))
    return center, offset

def get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, index_tensor):
    transf_cen_mul, transf_cen_add, transf_off_mul, transf_off_add = role_data
    transitive_mask = th.isin(index_tensor, transitive_ids)
    projection_dims = index_tensor[transitive_mask]
    inverse_mask = th.isin(index_tensor, inverse_ids)
    trans_inv = transitive_mask & inverse_mask
    trans_not_inv = transitive_mask & ~inverse_mask
    cen_mul = transf_cen_mul(index_tensor)
    cen_add = transf_cen_add(index_tensor)

    # if transitive:
        # bs, dim = cen_add.shape
        # transitive_bs = transitive_mask.sum()
        # cen_mask = th.zeros((transitive_bs, dim), device=cen_add.device)
        # bs_ids = th.arange(transitive_bs, device=cen_mask.device)
        # cen_mask[bs_ids, projection_dims] = 1
        # cen_add[transitive_mask] = cen_add[transitive_mask] * cen_mask

        # cen_mask = th.ones((transitive_bs, dim), device=cen_add.device)
        # cen_mul[transitive_mask] = cen_mask

        # cen_add[transitive_mask] = th.abs(cen_add[transitive_mask])
        # cen_add[inverse_mask] = -cen_add[inverse_mask]

        
    # if transitive:
        # cen_mul[transitive_mask] = 1
        # cen_add[transitive_mask] = th.abs(cen_add[transitive_mask])
        # cen_add[inverse_mask] = -cen_add[inverse_mask]
        
        
    off_mul = transf_off_mul(index_tensor)
    off_add = transf_off_add(index_tensor)
    
    if inter_translation is not None:
        # inter_add = id_add
        inter_add = inter_translation(index_tensor)
    else:
        inter_add = None
    return (cen_mul, cen_add, off_mul, off_add), inter_add, (trans_inv, trans_not_inv, projection_dims)

def embedding_1p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r',)),): '1p'
    c, c_offset = get_box_data(box_data, data[:, 0])
    transf_data, _, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 1])
    return Box(c, c_offset).transform(*transf_data), *transitive_data

def embedding_2p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c, c_offset = get_box_data(box_data, data[:, 0])

    transf_data_1, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 1])
    transf_data_2, _, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 2])

    box = Box(c, c_offset).transform(*transf_data_1).transform(*transf_data_2)
    return box, *transitive_data

def embedding_3p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c, c_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 1])
    transf_data_2, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 2])
    transf_data_3, _, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 3])
    
    box = Box(c, c_offset).transform(*transf_data_1).transform(*transf_data_2).transform(*transf_data_3)
    return box, *transitive_data

def embedding_2i(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, r_1_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, r_2_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 3])
    
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2).transform(id_mul, r_2_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), *transitive_data

def embedding_3i(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, r_1_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 1])
    
    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, r_2_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 3])

    c_3, c_3_offset = get_box_data(box_data, data[:, 4])
    transf_data_3, r_3_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 5])
    
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2).transform(id_mul, r_2_inter, id_mul, id_add)
    box_c_3 = Box(c_3, c_3_offset).transform(*transf_data_3).transform(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2, box_c_3), *transitive_data

def embedding_2in(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    # approximated as 1p
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, r_1_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 1])

    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1)#.transform(id_mul, r_1_inter, id_mul, id_add)
    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return box_c_1, *transitive_data
 

def embedding_3in(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    # approximated as 2i
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, r_1_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, r_2_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 3])

    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2).transform(id_mul, r_2_inter, id_mul, id_add)
    
    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), *transitive_data

def embedding_ip(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, r_1_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, r_2_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 3])
    
    transf_data_3, _, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 4])
                
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2).transform(id_mul, r_2_inter, id_mul, id_add)

    box = Box.intersection(box_c_1, box_c_2).transform(*transf_data_3, make_abs=False)
    return box, *transitive_data

def embedding_pi(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 1])
    transf_data_2, r_2_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 2])
    c_2, c_2_offset = get_box_data(box_data, data[:, 3])
    transf_data_3, r_3_inter, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(*transf_data_2).transform(id_mul, r_2_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_3).transform(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), *transitive_data


def embedding_inp(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    # approximated as 2p
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, inter_translation, data[:, 1])
    transf_data_3, _, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 5])
                                                                        
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(*transf_data_3)
    return box_c_1, *transitive_data
    
def embedding_pin(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    # approximated as 2p
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, *_ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 1])
    transf_data_2, _, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 2])
    
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(*transf_data_2)
    return box_c_1, *transitive_data


def embedding_pni(data, box_data, role_data, transitive_ids, inverse_ids, transitive, inter_translation):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    # approximated as 1p
    c_2, c_2_offset = get_box_data(box_data, data[:, 4])
    transf_data_3, *_, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, None, data[:, 5])

    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_3)
    return box_c_2, *transitive_data
    
