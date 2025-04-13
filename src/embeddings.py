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

def get_box_data(box_data, index_tensor):
    center_embed, offset_embed = box_data
    center = center_embed(index_tensor)
    offset = th.abs(offset_embed(index_tensor))
    return center, offset

def get_role_data(role_data, transitive_ids, inverse_ids, transitive, index_tensor):
    transitive_id_to_dimension = {t_id.item(): i for i, t_id in enumerate(transitive_ids)}
    transf_cen_mul, transf_cen_add, transf_off_mul, transf_off_add = role_data

    transitive_mask = th.isin(index_tensor, transitive_ids)
    projection_dims = index_tensor[transitive_mask]
    projection_dims = th.tensor([transitive_id_to_dimension[t_id.item()] for t_id in projection_dims], device=index_tensor.device)
    
    inverse_mask = th.isin(index_tensor, inverse_ids)
    trans_inv = transitive_mask & inverse_mask
    trans_not_inv = transitive_mask & ~inverse_mask

    cen_mul = transf_cen_mul(index_tensor)
    cen_add = transf_cen_add(index_tensor)
    off_mul = transf_off_mul(index_tensor)
    off_add = transf_off_add(index_tensor)

    # if transitive:
        # bs_ids = th.nonzero(transitive_mask).squeeze()
        # cen_mul[bs_ids, projection_dims] = id_mul.float()
        # cen_add[bs_ids, projection_dims] = id_add.float()
        # off_mul[bs_ids, projection_dims] = id_mul.float()
        # off_add[bs_ids, projection_dims] = id_add.float()

    return (cen_mul, cen_add, off_mul, off_add), (trans_inv, trans_not_inv, projection_dims)

def embedding_1p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r',)),): '1p'
    c, c_offset = get_box_data(box_data, data[:, 0])
    transf_data, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    return Box(c, c_offset).transform(*transf_data), *transitive_data

def embedding_2p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c, c_offset = get_box_data(box_data, data[:, 0])

    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    transf_data_2, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])

    box = Box(c, c_offset).transform(*transf_data_1).transform(*transf_data_2)
    return box, *transitive_data

def embedding_3p(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c, c_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])
    transf_data_3, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 3])
    
    box = Box(c, c_offset).transform(*transf_data_1).transform(*transf_data_2).transform(*transf_data_3)
    return box, *transitive_data

def embedding_2i(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])

    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 3])
    
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), *transitive_data

def embedding_3i(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    
    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 3])

    c_3, c_3_offset = get_box_data(box_data, data[:, 4])
    transf_data_3, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 5])
    
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)
    box_c_3 = Box(c_3, c_3_offset).transform(*transf_data_3)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2, box_c_3), *transitive_data

def embedding_2in(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    # approximated as 1p
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])

    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1)
    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return box_c_1, *transitive_data
 

def embedding_3in(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    # approximated as 2i
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])

    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 3])

    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)
    
    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), *transitive_data

def embedding_ip(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])

    c_2, c_2_offset = get_box_data(box_data, data[:, 2])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 3])
    
    transf_data_3, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 4])
                
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_2)

    box = Box.intersection(box_c_1, box_c_2).transform(*transf_data_3, make_abs=True) # True seems to work better
    return box, *transitive_data

def embedding_pi(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])
    c_2, c_2_offset = get_box_data(box_data, data[:, 3])
    transf_data_3, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 4])

    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(*transf_data_2)
    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_3)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return Box.intersection(box_c_1, box_c_2), *transitive_data


def embedding_inp(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    # approximated as 2p
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    transf_data_3, transitive_data = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 5])
                                                                        
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(*transf_data_3)
    return box_c_1, *transitive_data
    
def embedding_pin(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    # approximated as 2p. 
    c_1, c_1_offset = get_box_data(box_data, data[:, 0])
    transf_data_1, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 1])
    transf_data_2, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 2])
    
    box_c_1 = Box(c_1, c_1_offset).transform(*transf_data_1).transform(*transf_data_2)

    false_tensor = th.zeros(c_1.shape[0]).bool().to(c_1.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_1.device)
    return box_c_1, *transitive_data


def embedding_pni(data, box_data, role_data, transitive_ids, inverse_ids, transitive):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    # approximated as 1p. 
    c_2, c_2_offset = get_box_data(box_data, data[:, 4])
    transf_data_3, _ = get_role_data(role_data, transitive_ids, inverse_ids, transitive, data[:, 5])

    box_c_2 = Box(c_2, c_2_offset).transform(*transf_data_3)

    false_tensor = th.zeros(c_2.shape[0]).bool().to(c_2.device)
    transitive_data = false_tensor, false_tensor, empty_tensor.to(c_2.device)
    return box_c_2, *transitive_data
    
