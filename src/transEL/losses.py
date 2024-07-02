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
    def __init__(self, center, offset, upper_corner=None, lower_corner=None):
        self.center = center
        self.offset = th.abs(offset)
        self.upper_corner = upper_corner
        self.lower_corner = lower_corner

    @check_output_shape
    @staticmethod
    def inclusion(box1, box2, margin):
        """
        Positive margin allows the box1 to be partially outside box2
        """
        euc_distance = th.abs(box1.center - box2.center)
        return th.linalg.norm(th.relu(euc_distance + box1.offset - box2.offset - margin), axis=1) # 

    @check_output_shape
    @staticmethod
    def non_inclusion(box1, box2, margin):
        euc_distance = th.abs(box1.center - box2.center)
        return th.linalg.norm(th.relu(euc_distance + box1.offset - box2.offset - margin), axis=1)
        # return th.linalg.norm(th.relu(-euc_distance + box1.offset + box2.offset - margin), axis=1)

    @check_output_shape
    @staticmethod
    def transitive_inclusion(box1, box2, relation, margin):
        # return Box.inclusion(box1, box2, margin)
        # margin = max(margin, 0.2)
        logger.debug(f"Box1 center: {box1.center.shape}. Box2 center: {box2.center.shape}. Relation: {relation.shape}")
        # order_loss = th.linalg.norm(th.relu(box1.center + box1.offset - box2.center - box2.offset + margin), axis=1) # positive margin forces box1 to be before box2
        order_loss = th.mean(th.relu(box1.center + box1.offset - box2.center - box2.offset), axis=1)
        angle_loss = 1 - F.cosine_similarity(relation, (box2.center - box1.center), dim=1)
        # angle_loss = 1- th.sigmoid(th.sum(relation * (box2.center - box1.center), axis=1)) 
        logger.debug(f"Order loss: {order_loss.mean()}. Angle loss: {angle_loss.mean()}")
        
        alpha =0.1
        return order_loss + alpha*angle_loss
        return alpha*order_loss + (1-alpha)*angle_loss

    @check_output_shape
    @staticmethod
    def non_transitive_inclusion(box1, box2, relation, margin):
        return Box.non_inclusion(box1, box2, margin)

    
    
    @staticmethod
    def intersection(box1, box2):
        lower_corner = th.maximum(box1.center - box1.offset, box2.center - box2.offset)
        upper_corner = th.minimum(box1.center + box1.offset, box2.center + box2.offset)
        center = (lower_corner + upper_corner) / 2
        offset = th.abs(lower_corner - upper_corner) / 2
        return Box(center, offset, upper_corner, lower_corner)

    #new
    def corners_loss(self):
        if self.upper_corner is None or self.lower_corner is None:
            raise ValueError("Upper and lower corners not defined. Box must be created with intersection method")

        loss = th.linalg.norm(th.relu(self.lower_corner - self.upper_corner), axis=1)
        return loss
        

@check_output_shape
def gci0_loss(data, class_embed, class_offset, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))

    box_c = Box(c, off_c)
    box_d = Box(d, off_d)

    if neg:
        loss = Box.non_inclusion(box_c, box_d, margin)
    else:
        loss = Box.inclusion(box_c, box_d, margin)
    return loss

@check_output_shape
def gci0_bot_loss(data, class_offset, margin, neg=False):
    if neg:
        off_c = th.abs(class_offset(data[:, 0]))
        loss = th.linalg.norm(th.relu(-off_c + margin), axis=1)
    else:
        off_c = th.abs(class_offset(data[:, 0]))
        loss = th.linalg.norm(off_c, axis=1)
    
    
    return loss

@check_output_shape
def gci1_loss(data, class_embed, class_offset, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))

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
def gci1_bot_loss(data, class_embed, class_offset, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))

    euc = th.abs(c - d)
    dst = th.linalg.norm(th.relu(-euc + off_c + off_d - margin), axis=1) # positive margin forces a minimum distance between c and d
    return dst

@check_output_shape
def gci2_loss(data, class_embed, class_offset, rel_embed, rel_mask, transitive_ids, margin, transitive, neg=False):
    r = data[:, 1]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    c_trans = class_embed(trans_data[:, 0])
    r_trans = rel_embed(trans_data[:, 1]) #* rel_mask(trans_data[:, 1])
    r_trans = th.abs(r_trans)
    
    d_trans = class_embed(trans_data[:, 2])
    off_c_trans = th.abs(class_offset(trans_data[:, 0]))
    off_d_trans = th.abs(class_offset(trans_data[:, 2]))

    box_c_trans = Box(c_trans + r_trans, off_c_trans)
    box_d_trans = Box(d_trans, off_d_trans)

    if transitive:
        if neg:
            transitive_fn = Box.non_transitive_inclusion
        else:
            transitive_fn = Box.transitive_inclusion
    else:
        if neg:
            transitive_fn = Box.non_inclusion
        else:
            transitive_fn = Box.inclusion

    transitive_loss = transitive_fn(box_c_trans, box_d_trans, r_trans, margin)
            
    c_non_trans = class_embed(non_trans_data[:, 0])
    r_non_trans = rel_embed(non_trans_data[:, 1])
    d_non_trans = class_embed(non_trans_data[:, 2])
    off_c_non_trans = th.abs(class_offset(non_trans_data[:, 0]))
    off_d_non_trans = th.abs(class_offset(non_trans_data[:, 2]))

    box_c_non_trans = Box(c_non_trans + r_non_trans, off_c_non_trans)
    box_d_non_trans = Box(d_non_trans, off_d_non_trans)

    if neg:
        non_trans_fn = Box.non_inclusion
    else:
        non_trans_fn = Box.inclusion
        
    non_trans_loss = non_trans_fn(box_c_non_trans, box_d_non_trans, margin)

    final_output = th.zeros(data.shape[0], device=data.device)
    final_output[mask] = transitive_loss
    final_output[~mask] = non_trans_loss

    return final_output

@check_output_shape
def gci3_loss_new(data, class_embed, class_offset, rel_embed, rel_mask, transitive_ids, margin, neg=False):
    
    r = data[:, 0]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    r_trans = rel_embed(trans_data[:, 0]) #* rel_mask(trans_data[:, 0])
    r_trans = th.abs(r_trans)
    
    c_trans = class_embed(trans_data[:, 1])
    d_trans = class_embed(trans_data[:, 2])
    off_c_trans = th.abs(class_offset(trans_data[:, 1]))
    off_d_trans = th.abs(class_offset(trans_data[:, 2]))

    box_c_trans = Box(c_trans - r_trans, off_c_trans)
    box_d_trans = Box(d_trans, off_d_trans)

    transitive_loss = Box.inclusion(box_c_trans, box_d_trans, margin)

    r_non_trans = rel_embed(non_trans_data[:, 0])
    c_non_trans = class_embed(non_trans_data[:, 1])
    d_non_trans = class_embed(non_trans_data[:, 2])
    off_c_non_trans = th.abs(class_offset(non_trans_data[:, 1]))
    off_d_non_trans = th.abs(class_offset(non_trans_data[:, 2]))

    box_c_non_trans = Box(c_non_trans-r_non_trans, off_c_non_trans)
    box_d_non_trans = Box(d_non_trans, off_d_non_trans)
    inclusion_loss = Box.inclusion(box_c_non_trans, box_d_non_trans, margin)

    final_output = th.zeros(data.shape[0], device=data.device)
    final_output[mask] = transitive_loss
    final_output[~mask] = inclusion_loss

    return final_output


def gci3_loss(data, class_embed, class_offset, rel_embed, rel_mask, transitive_ids, margin, neg=False):
    r_raw = data[:, 0]
    mask = th.isin(r_raw, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    r_trans = rel_embed(trans_data[:, 0]) #* rel_mask(trans_data[:, 0])
    r_trans = th.abs(r_trans)
    r_non_trans = rel_embed(non_trans_data[:, 0])

    r = th.zeros(data.shape[0], r_trans.shape[1], device=data.device)
    r[mask] = r_trans
    r[~mask] = r_non_trans
    
    # r = rel_embed(data[:, 0])


    
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))

    box_c = Box(c-r, off_c)
    box_d = Box(d, off_d)

    if neg:
        loss = Box.non_inclusion(box_c, box_d, margin)
    else:
        loss = Box.inclusion(box_c, box_d, margin)
    return loss


@check_output_shape
def gci3_bot_loss(data, class_offset, margin, neg=False):
    off_c = th.abs(class_offset(data[:, 1]))
    if neg:
        loss = th.linalg.norm(off_c, axis=1)
    else:
        loss = th.linalg.norm(th.relu(-off_c + margin), axis=1)
    return loss


@check_output_shape
def class_assertion_loss(data, class_embed, class_offset, individual_embed, margin, neg=False):
    c = class_embed(data[:, 0])
    i = individual_embed(data[:, 1])

    off_c = th.abs(class_offset(data[:, 0]))
    off_i = th.zeros_like(off_c)

    box_c = Box(c, off_c)
    box_i = Box(i, off_i)

    if neg:
        loss = Box.non_inclusion(box_i, box_c, margin)
    else:
        loss = Box.inclusion(box_i, box_c, margin)
        
    return loss

@check_output_shape
def object_property_assertion_loss(data, individual_embed, rel_embed, rel_mask, transitive_ids, margin, neg=False):
    # logger.debug("All data")
    # logger.debug(data)
    r = data[:, 1]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    logger.debug(trans_data)
    non_trans_data = data[~mask]

    i1_trans = individual_embed(trans_data[:, 0])
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



def regularization_loss(rel_embed, rel_mask, transitive_ids, reg_factor=0.1):
    r = th.abs(rel_embed(transitive_ids))
    norm_r = F.normalize(r, p=2, dim=1)
    norm_loss = 0 #th.relu(0.5 - norm_r).mean()
        
    
    cos_sim_matrix = th.mm(norm_r, norm_r.t())
    identity = th.eye(r.shape[0], device=r.device)
    orthogonality_loss = th.linalg.norm(cos_sim_matrix - identity, ord='fro')
    return reg_factor * orthogonality_loss.mean() + reg_factor * norm_loss
