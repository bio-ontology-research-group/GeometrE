import torch as th
import torch.nn.functional as F
import numpy as np

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def check_output_shape(func):
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if len(output.shape) > 1:
            raise ValueError(f"Expected output to have shape (n,), got {output.shape}")
        return output
    return wrapper
    

class Box():
    def __init__(self, lower_corner, delta, pos_delta=True):
        # if delta is not None and upper_corner is not None:
            # raise ValueError("Cannot define both upper and delta")

        self.lower_corner = lower_corner
        
        # if delta is not None:
        if pos_delta:
            delta = th.abs(delta)
        self.delta = delta
        self.upper_corner = lower_corner + delta

        # if upper_corner is not None:
            # self.upper_corner = upper_corner
            # self.delta = th.abs(upper_corner - lower_corner)
            
         
    @check_output_shape
    @staticmethod
    def inclusion(box1, box2, margin, *args):
        lower_corner_condition = th.relu(box2.lower_corner - box1.lower_corner - margin)
        upper_corner_condition = th.relu(box1.upper_corner - box2.upper_corner - margin)

        return th.linalg.norm(lower_corner_condition, axis=1) + th.linalg.norm(upper_corner_condition, axis=1)
        
    @check_output_shape
    @staticmethod
    def non_inclusion(box1, box2, margin, *args):

        return Box.inclusion(box1, box2, margin, *args)
        
        intersection_lower_corner = th.maximum(box1.lower_corner, box2.lower_corner)
        intersection_upper_corner = th.minimum(box1.upper_corner, box2.upper_corner)

        disjoint_condition = th.relu(intersection_upper_corner - intersection_lower_corner - margin)
        return th.linalg.norm(disjoint_condition, axis=1, ord=-float("inf"))


    @staticmethod
    def intersection(box1, box2):
        intersection_lower_corner = th.maximum(box1.lower_corner, box2.lower_corner)
        intersection_upper_corner = th.minimum(box1.upper_corner, box2.upper_corner)
        delta = intersection_upper_corner - intersection_lower_corner
        return Box(intersection_lower_corner, delta, pos_delta=False)

    @staticmethod
    def unbound(box, rel_mask, min_bound):
        min_bound = 0
        unbound_dimension = th.where(rel_mask == 1)
        # print(unbound_dimension)
        # print(unbound_dimension[0].shape, unbound_dimension[1].shape)
        # print(f"unbound_dimension: {unbound_dimension.shape}")
        # check = th.sum(unbound_dimension, axis=1)
        # assert th.all(check == 1), "Rel mask should have only one 1 per row"

        new_lower_corner = box.lower_corner.clone()
        new_upper_corner = box.upper_corner.clone()
        
        new_lower_corner[unbound_dimension] = min_bound
        delta = new_upper_corner - new_lower_corner
        return Box(new_lower_corner, delta)
 


            
    # @check_output_shape
    # @staticmethod
    # def inclusion(box1, box2, margin):
        # """
        # Positive margin allows the box1 to be partially outside box2
        # """
        # euc_distance = th.abs(box1.center - box2.center)
                
        # return th.linalg.norm(th.relu(euc_distance + box1.offset - box2.offset - margin), axis=1)

    # @check_output_shape
    # @staticmethod
    # def non_inclusion(box1, box2, margin):
        # euc_distance = th.abs(box1.center - box2.center)
        # return th.linalg.norm(th.relu(euc_distance + box1.offset - box2.offset - margin), axis=1)
        
    @check_output_shape
    @staticmethod
    def non_transitive_inclusion(box1, box2, margin, relation):
        # loss = th.mean(th.relu(box1.center + box1.offset - box2.center - box2.offset), axis=1)
        # return loss
        return Box.non_inclusion(box1, box2, margin, relation)

    # @staticmethod
    # def intersection(box1, box2):
        # lower_corner = th.maximum(box1.center - box1.offset, box2.center - box2.offset)
        # upper_corner = th.minimum(box1.center + box1.offset, box2.center + box2.offset)
        # center = (lower_corner + upper_corner) / 2
        # offset = th.abs(lower_corner - upper_corner) / 2
        # return Box(center, offset)

    def corners_loss(self):
        if self.upper_corner is None or self.lower_corner is None:
            raise ValueError("Upper and lower corners not defined. Box must be created with intersection method")

        loss = th.linalg.norm(th.relu(self.lower_corner - self.upper_corner), axis=1)
        return loss
        

    # @staticmethod
    # def unbound(box, rel_mask, min_bound):
        # min_bound = - abs(min_bound)
        # unbound_dimension = th.where(rel_mask == 1)
                                        
        # new_lower_corner = box.lower_corner.clone()
        # new_upper_corner = box.upper_corner.clone()
        
        # new_lower_corner[unbound_dimension] = min_bound

        # new_center = (new_lower_corner + new_upper_corner) / 2
        # new_offset = th.abs(new_lower_corner - new_upper_corner) / 2
        
        # return Box(new_center, new_offset)
 

@check_output_shape
def gci0_loss(data, class_lower, class_delta, margin, neg=False):
    c = class_lower(data[:, 0])
    d = class_lower(data[:, 1])

    off_c = class_delta(data[:, 0])
    off_d = class_delta(data[:, 1])

    box_c = Box(c, off_c)
    box_d = Box(d, off_d)
    
    if neg:
        loss = Box.non_inclusion(box_c, box_d, margin)
    else:
        loss = Box.inclusion(box_c, box_d, margin)
    return loss

@check_output_shape
def gci0_bot_loss(data, class_lower, class_delta, margin, neg=False):
    lower_c = class_lower(data[:, 0])
    delta_c = class_delta(data[:, 0])
    box_c = Box(lower_c, delta_c, pos_delta = False)

    lower_corner_condition = th.linalg.norm(th.exp(th.ones_like(box_c.lower_corner)) - box_c.lower_corner, axis=1)
    upper_corner_condition = th.linalg.norm(th.exp(box_c.upper_corner), axis=1)

    loss = (lower_corner_condition + upper_corner_condition)/2
    return loss

@check_output_shape
def gci1_loss(data, class_lower, class_delta, margin, neg=False):
    c = class_lower(data[:, 0])
    d = class_lower(data[:, 1])
    e = class_lower(data[:, 2])
    off_c = th.abs(class_delta(data[:, 0]))
    off_d = th.abs(class_delta(data[:, 1]))
    off_e = th.abs(class_delta(data[:, 2]))

    box_c = Box(c, off_c)
    box_d = Box(d, off_d)
    box_e = Box(e, off_e)

    intersection_box = Box.intersection(box_c, box_d)
    if neg:
        loss = Box.non_inclusion(intersection_box, box_e, margin)
    else:
        loss = Box.inclusion(intersection_box, box_e, margin) + intersection_box.corners_loss()
    return loss

@check_output_shape
def gci1_bot_loss(data, class_lower, class_delta, margin, neg=False):
    c = class_lower(data[:, 0])
    d = class_lower(data[:, 1])

    off_c = th.abs(class_delta(data[:, 0]))
    off_d = th.abs(class_delta(data[:, 1]))

    box_c = Box(c, off_c)
    box_d = Box(d, off_d)

    
    intersection_box = Box.intersection(box_c, box_d)

    loss = th.linalg.norm(th.relu(intersection_box.upper_corner - intersection_box.lower_corner - margin), axis=1, ord=-float("inf"))

    
    # box_c = Box(c, off_c)
    # box_d = Box(d, off_d)

    # c = box_c.center
    # d = box_d.center

    # off_c = box_c.offset
    # off_d = box_d.offset
    
    # euc = th.abs(c - d)
    # dst = th.linalg.norm(th.relu(-euc + off_c + off_d - abs(margin)), axis=1) # positive margin forces a minimum distance between c and d

    # loss = th.linalg.norm(th.relu(intersection_box.upper_corner - intersection_box.lower_corner - margin), axis=1, ord=-float("inf"))
    
    return loss

@check_output_shape
def normal_gci2_loss(data, class_lower, class_delta, rel_embed, margin, neg=False):
    c = class_lower(data[:, 0])
    r = rel_embed(data[:, 1])
    d = class_lower(data[:, 2])

    off_c = th.abs(class_delta(data[:, 0]))
    off_d = th.abs(class_delta(data[:, 2]))

    box_c = Box(c, off_c)
    box_d = Box(d - r, off_d)

    if neg:
        loss = Box.non_inclusion(box_c, box_d, margin)
    else:
        loss = Box.inclusion(box_c, box_d, margin)


    return loss


@check_output_shape
def gci2_loss(data, class_lower, class_delta, rel_embed, rel_mask, min_bound, transitive_ids, margin, transitive, neg=False, evaluate=False): #adapted

    if not transitive:
        return normal_gci2_loss(data, class_lower, class_delta, rel_embed, margin, neg = neg)

    
    r = data[:, 1]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    c_trans = class_lower(trans_data[:, 0])
    off_c_trans = th.abs(class_delta(trans_data[:, 0]))

    d_trans = class_lower(trans_data[:, 2])
    off_d_trans = th.abs(class_delta(trans_data[:, 2]))


    r_embed = rel_embed(trans_data[:, 1])
    r_mask =  rel_mask(trans_data[:, 1])
    r_trans =  th.abs(r_embed * r_mask)
            
    box_c_trans = Box(c_trans, off_c_trans)
    box_d_trans = Box(d_trans - r_trans, off_d_trans)

    box_d_unbounded = Box.unbound(box_d_trans, r_mask, min_bound)

    if neg:
        # unbound_dimension = th.where(r_mask == 1)
        # box_c_trans.lower_corner[unbound_dimension] = 0
        # box_d_unbounded.lower_corner[unbound_dimension] = 0
        transitive_loss = Box.non_inclusion(box_c_trans, box_d_unbounded, margin)
    else:
        transitive_loss = Box.inclusion(box_c_trans, box_d_unbounded, margin)

    
            
    c_non_trans = class_lower(non_trans_data[:, 0])
    off_c_non_trans = th.abs(class_delta(non_trans_data[:, 0]))

    d_non_trans = class_lower(non_trans_data[:, 2])
    off_d_non_trans = th.abs(class_delta(non_trans_data[:, 2]))

    r_non_trans = rel_embed(non_trans_data[:, 1])

    box_c_non_trans = Box(c_non_trans, off_c_non_trans)
    box_d_non_trans = Box(d_non_trans - r_non_trans, off_d_non_trans)

    if neg:
        non_trans_fn = Box.non_inclusion
    else:
        non_trans_fn = Box.inclusion
        
    non_trans_loss = non_trans_fn(box_c_non_trans, box_d_non_trans, margin)

    final_output = th.zeros(data.shape[0], device=data.device)
    final_output[mask] = transitive_loss
    final_output[~mask] = non_trans_loss

    return final_output


def normal_gci3_loss(data, class_lower, class_delta, rel_embed, margin, neg=False):
    r = rel_embed(data[:, 0])
    c = class_lower(data[:, 1])
    d = class_lower(data[:, 2])

    off_c = th.abs(class_delta(data[:, 1]))
    off_d = th.abs(class_delta(data[:, 2]))

    box_c = Box(c-r, off_c)
    box_d = Box(d, off_d)

    if neg:
        loss = Box.non_inclusion(box_c, box_d, margin)
    else:
        loss = Box.inclusion(box_c, box_d, margin)

    return loss, None, None


 
def gci3_loss(data, class_lower, class_delta, rel_embed, rel_mask, min_bound, transitive_ids, margin, transitive, neg=False):

    if not transitive:
        return normal_gci3_loss(data, class_lower, class_delta, rel_embed, margin, neg)
    
    r_raw = data[:, 0]
    mask = th.isin(r_raw, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    c_trans = class_lower(trans_data[:, 1])
    off_c_trans = th.abs(class_delta(trans_data[:, 1]))

    d_trans = class_lower(trans_data[:, 2])
    off_d_trans = th.abs(class_delta(trans_data[:, 2]))
    
    r_embed = rel_embed(trans_data[:, 0])
    r_mask = rel_mask(trans_data[:, 0])
    r_trans = th.abs(r_embed * r_mask)
    
    box_c_trans = Box(c_trans - r_trans, off_c_trans)
    box_c_unbounded = Box.unbound(box_c_trans, r_mask, min_bound)

    box_d_trans = Box(d_trans, off_d_trans)
            
    if neg:
        unbound_dimension = th.where(r_mask == 1)
        # box_c_unbounded.lower_corner[unbound_dimension] = 0
        # box_d_trans.lower_corner[unbound_dimension] = 0

        transitive_loss = Box.non_inclusion(box_c_unbounded, box_d_trans, margin)
    else:
        transitive_loss = Box.inclusion(box_c_unbounded, box_d_trans, margin)

    

    c_non_trans = class_lower(non_trans_data[:, 1])
    off_c_non_trans = th.abs(class_delta(non_trans_data[:, 1]))

    d_non_trans = class_lower(non_trans_data[:, 2])
    off_d_non_trans = th.abs(class_delta(non_trans_data[:, 2]))

    r_non_trans = rel_embed(non_trans_data[:, 0])

    box_c_non_trans = Box(c_non_trans - r_non_trans, off_c_non_trans)
    box_d_non_trans = Box(d_non_trans, off_d_non_trans)

    if neg:
        loss_fn = Box.non_inclusion
    else:
        loss_fn = Box.inclusion
    non_transitive_loss = loss_fn(box_c_non_trans, box_d_non_trans, margin)

    final_output = th.zeros(data.shape[0], device=data.device)
    final_output[mask] = transitive_loss
    final_output[~mask] = non_transitive_loss
    
    return final_output, trans_data[:, 2], th.where(r_mask==1)


@check_output_shape
def gci3_bot_loss(data, class_delta, margin, neg=False):
    off_c = th.abs(class_delta(data[:, 1]))
    if neg:
        loss = th.linalg.norm(off_c, axis=1)
    else:
        loss = th.linalg.norm(th.relu(-off_c + margin), axis=1)
    return loss


@check_output_shape
def class_assertion_loss(data, class_lower, class_delta, individual_embed, margin, neg=False):
    c = class_lower(data[:, 0])
    i = individual_embed(data[:, 1])

    off_c = th.abs(class_delta(data[:, 0]))
    off_i = th.zeros_like(off_c)

    box_c = Box(c, off_c)
    box_i = Box(i, off_i)

    if neg:
        loss = Box.non_inclusion(box_i, box_c, margin)
    else:
        loss = Box.inclusion(box_i, box_c, margin)
        
    return loss

@check_output_shape
def object_property_assertion_loss(data, individual_embed, rel_embed, rel_mask, min_bound, transitive_ids, margin, transitive, neg=False):
    # logger.debug("All data")
    # logger.debug(data)
    r = data[:, 1]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    logger.debug(trans_data)
    non_trans_data = data[~mask]

    i1_trans = individual_embed(trans_data[:, 0])
    if transitive:
        r_trans = th.abs(rel_embed(trans_data[:, 1])) #* rel_mask(trans_data[:, 1])
    i2_trans = individual_embed(trans_data[:, 2])
    off_i1_trans = th.zeros_like(i1_trans)
    off_i2_trans = th.zeros_like(off_i1_trans)

    box_c_trans = Box(i1_trans + r_trans, off_i1_trans)
    box_d_trans = Box(i2_trans, off_i2_trans)

    if neg:
        transitive_loss = Box.non_transitive_inclusion(box_c_trans, box_d_trans, r_trans, margin)
    else:
        transitive_loss = Box.transitive_inclusion(box_c_trans, box_d_trans, r_trans, margin)


    
    
    i1_non_trans = individual_embed(non_trans_data[:, 0])
    r_non_trans = rel_embed(non_trans_data[:, 1])
    # logger.debug("\n\nNon transitive data")
    # logger.debug(non_trans_data)
    i2_non_trans = individual_embed(non_trans_data[:, 2])
    off_i1_non_trans = th.zeros_like(i1_non_trans)
    off_i2_non_trans = th.zeros_like(i2_non_trans)

    box_c_non_trans = Box(i1_non_trans + r_non_trans, off_i1_non_trans)
    box_d_non_trans = Box(i2_non_trans, off_i2_non_trans)

    if neg:
        non_trans_loss = Box.non_inclusion(box_c_non_trans, box_d_non_trans, margin)
    else:
        non_trans_loss = Box.inclusion(box_c_non_trans, box_d_non_trans, margin)
        
    final_output = th.zeros(data.shape[0], device=data.device)
    # alpha = 0.9
    # final_output[mask] = alpha * transitive_loss
    # final_output[~mask] = (1-alpha) * inclusion_loss
    final_output[mask] = transitive_loss
    final_output[~mask] = non_trans_loss
            
    return final_output


def regularization_loss(class_lower, reg_factor):
    lower = class_lower.weight
    return th.relu(-lower).mean() * reg_factor


# def regularization_loss(rel_embed, rel_mask, transitive_ids, reg_factor=0.1):
    # r = th.abs(rel_embed(transitive_ids)) * rel_mask(transitive_ids)
    # norm_r = F.normalize(r, p=2, dim=1)
    # norm_loss = 0 #th.relu(0.5 - norm_r).mean()
        
    
    # cos_sim_matrix = th.mm(norm_r, norm_r.t())
    # identity = th.eye(r.shape[0], device=r.device)
    # orthogonality_loss = th.linalg.norm(cos_sim_matrix - identity, ord='fro')
    # return reg_factor * orthogonality_loss.mean() + reg_factor * norm_loss
