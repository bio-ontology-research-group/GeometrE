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

def get_box_data(center_embed, offset_embed, index_tensor):
    center = center_embed(index_tensor)
    offset = th.abs(offset_embed(index_tensor))
    return center, offset

def get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, index_tensor):
    # print(f"transitive_ids: {transitive_ids}")
    # print(f"inverse_ids: {inverse_ids}")
    transitive_mask = th.isin(index_tensor, transitive_ids)
    # print(f"transitive_mask: {transitive_mask}")
    projection_dims = index_tensor[transitive_mask]
    # print(f"projection_dims: {projection_dims}")
    
    # trans_ids = index_tensor[transitive_mask]
    inverse_mask = th.isin(index_tensor, inverse_ids)
    # print(f"inverse_mask: {inverse_mask}")
    trans_inv = transitive_mask & inverse_mask
    trans_not_inv = transitive_mask & ~inverse_mask
    # print(f"trans inv: {trans_inv}")
    # print(f"trans not inv: {trans_not_inv}")
    factor = translation_mul(index_tensor)
    # factor[transitive_mask] = id_mul

    # trans_bs = transitive_mask.sum()
    # hid_dim = factor.shape[1]
    
    # transitive_tensor = th.zeros((trans_bs, hid_dim), device=factor.device)
    # x_dim = th.arange(trans_bs)
    # transitive_tensor[x_dim, trans_ids] = 1

    add = translation_add(index_tensor)

    # add[trans_inv] = -th.abs(add[trans_inv])
    # add[trans_not_inv] = th.abs(add[trans_not_inv])
    # add[transitive_mask] = add[transitive_mask] * transitive_tensor
    scale = scaling_mul(index_tensor)
    # scale[transitive_mask] = id_mul
    scaling_add = scaling_add(index_tensor)
    # scaling_add[transitive_mask] = id_add
    
    if inter_translation is not None:
        inter_add = inter_translation(index_tensor)
    else:
        inter_add = None
    return factor, add, scale, scaling_add, inter_add, trans_inv, trans_not_inv, projection_dims

def embedding_1p(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids):
    # (('e', ('r',)),): '1p'
    c, c_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_factor, r_add, r_scale, r_scale_add, _, trans_inv, trans_not_inv, projection_dims = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 1])
    return (Box(c, c_offset).translate(r_factor, r_add, r_scale, r_scale_add), th.zeros((c.shape[0], 1)), 0, 0), (trans_inv, trans_not_inv, projection_dims)

def embedding_2p(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids):
    c, c_offset = get_box_data(center_embed, offset_embed, data[:, 0])

    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 1])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, _, trans_inv, trans_not_inv, projection_dims = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 2])

    box = Box(c, c_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add), th.zeros((c.shape[0], 1)), 0, 0
    return box, (trans_inv, trans_not_inv, projection_dims)

def embedding_3p(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids):
    c, c_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 1])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 2])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, _, trans_inv, trans_not_inv, projection_dims = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 3])
    
    box = Box(c, c_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add)
    box = box.translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add)
    box = box.translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add)
    return (box, th.zeros((c.shape[0], 1)), 0, 0), (trans_inv, trans_not_inv, projection_dims)

def embedding_2i(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, r_1_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 2])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 3])
    
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add).translate(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), (false_tensor, false_tensor, empty_tensor.to(c_1.device))

def embedding_3i(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, r_1_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 1])
    
    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 2])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 3])

    c_3, c_3_offset = get_box_data(center_embed, offset_embed, data[:, 4])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, r_3_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 5])
    
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add).translate(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)
    box_c_3 = Box(c_3, c_3_offset).translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add).translate(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    return Box.intersection(box_c_1, box_c_2, box_c_3), (false_tensor, false_tensor, empty_tensor.to(c_1.device))

def embedding_2in(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, r_1_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 2])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 3])
    
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add).translate(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    return Box.intersection_with_negation(2, box_c_1, box_c_2), (false_tensor, false_tensor, empty_tensor.to(c_1.device))
 

def embedding_3in(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, r_1_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 2])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 3])

    c_3, c_3_offset = get_box_data(center_embed, offset_embed, data[:, 4])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, r_3_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 5])
    
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add).translate(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)
    box_c_3 = Box(c_3, c_3_offset).translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add).translate(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    return Box.intersection_with_negation(3, box_c_1, box_c_2, box_c_3), (false_tensor, false_tensor, empty_tensor.to(c_1.device))

def embedding_ip(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, r_1_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 1])

    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 2])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 3])
    
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, _, trans_inv, trans_not_inv, projection_dims = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 4])
                
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add).translate(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)

    box, corner_logit, disjoints, total_boxes = Box.intersection(box_c_1, box_c_2)
    box = box.translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add)
    return (box, corner_logit, disjoints, total_boxes), (trans_inv, trans_not_inv, projection_dims)

def embedding_pi(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 1])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 2])
    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 3])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, r_3_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add)
    box_c_1 = box_c_1.translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add).translate(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), (false_tensor, false_tensor, empty_tensor.to(c_1.device))



def embedding_inp(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, r_1_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 1])
    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 2])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 3])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, _, trans_inv, trans_not_inv, projection_dims = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 5])
                                                                        
    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add).translate(id_mul, r_1_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)

    box, corner_logit, disjoints, total = Box.intersection_with_negation(2, box_c_1, box_c_2)
    box = box.translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add)
    return (box, corner_logit, disjoints, total), (trans_inv, trans_not_inv, projection_dims)

def embedding_pin(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 1])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 2])
    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 3])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, r_3_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add)
    box_c_1 = box_c_1.translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add).translate(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    return Box.intersection_with_negation(2, box_c_1, box_c_2), (false_tensor, false_tensor, empty_tensor.to(c_1.device))


def embedding_pni(data, center_embed, offset_embed, translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    c_1, c_1_offset = get_box_data(center_embed, offset_embed, data[:, 0])
    r_1_mul, r_1_add, r_1_scale, r_1_scaling_add, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, None, data[:, 1])
    r_2_mul, r_2_add, r_2_scale, r_2_scaling_add, r_2_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 2])
    c_2, c_2_offset = get_box_data(center_embed, offset_embed, data[:, 4])
    r_3_mul, r_3_add, r_3_scale, r_3_scaling_add, r_3_inter, *_ = get_role_data(translation_mul, translation_add, scaling_mul, scaling_add, transitive_ids, inverse_ids, inter_translation, data[:, 5])

    box_c_1 = Box(c_1, c_1_offset).translate(r_1_mul, r_1_add, r_1_scale, r_1_scaling_add)
    box_c_1 = box_c_1.translate(r_2_mul, r_2_add, r_2_scale, r_2_scaling_add).translate(id_mul, r_2_inter, id_mul, id_add)
    box_c_2 = Box(c_2, c_2_offset).translate(r_3_mul, r_3_add, r_3_scale, r_3_scaling_add).translate(id_mul, r_3_inter, id_mul, id_add)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    
    return Box.intersection_with_negation(1, box_c_1, box_c_2), (false_tensor, false_tensor, empty_tensor.to(c_1.device))
        
