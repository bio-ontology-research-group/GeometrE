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
        if isinstance(output, tuple):
            for o in output:
                if len(o.shape) > 1:
                    raise ValueError(f"Expected output to have shape (n,), got {output.shape}")

        else:
            if len(output.shape) > 1:
                raise ValueError(f"Expected output to have shape (n,), got {output.shape}")
        return output
    return wrapper
    

class Point():
    def __init__(self, location):
        self.location = location
        
    @check_output_shape
    def order_loss(box1, box2, margin, r_trans):
        order_loss = th.linalg.norm(th.relu(box1.location - box2.location - margin), dim=1)

        diff = (box2.location - box1.location)
        diff_norm = th.norm(diff, p=2, dim=1)
        diff = diff/diff_norm.unsqueeze(1)
        angle_loss = 1 - th.sigmoid(th.sum(diff * r_trans, dim=1))
        return order_loss + angle_loss
        
    @check_output_shape
    def point_distance(box1, box2):
        return th.linalg.norm(box1.location - box2.location, dim=1)
        
@check_output_shape
def gci2_loss(data, class_embed, rel_embed, transitive_ids, margin, transitive, neg=False, train=False):

    if not transitive:
        return normal_object_property_assertion_loss(data, class_embed, rel_embed, transitive_ids, margin, neg = neg)

    return object_property_assertion_loss(data, class_embed, rel_embed, transitive_ids, margin, transitive, neg = neg, train=train)
 
def normal_object_property_assertion_loss(data, individual_embed, rel_embed, transitive_ids, margin, neg=False):

    i1 = individual_embed(data[:, 0])
    r = rel_embed(data[:, 1])
    i2 = individual_embed(data[:, 2])

    point_i1 = Point(i1)
    point_i2 = Point(i2 - r)

    if neg:
        loss = Point.point_distance(point_i1, point_i2)
    else:
        loss = Point.point_distance(point_i1, point_i2)

    return loss
    
@check_output_shape
def object_property_assertion_loss(data, individual_embed, rel_embed, transitive_ids, margin, transitive, neg=False, train=False):

    if not transitive:
        return normal_object_property_assertion_loss(data, individual_embed, rel_embed, transitive_ids, margin, neg=neg)
    
    r = data[:, 1]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    i1_trans = individual_embed(trans_data[:, 0])
    
    r_embed = rel_embed(trans_data[:, 1])
    r_trans = th.abs(r_embed)/th.norm(r_embed, dim=1).unsqueeze(1)
    
    i2_trans = individual_embed(trans_data[:, 2])

    point_c_trans = Point(i1_trans)
    point_d_trans = Point(i2_trans - r_trans)
    
    if neg:
        transitive_loss = Point.order_loss(point_c_trans, point_d_trans, margin, r_trans)
    else:
        transitive_loss = Point.order_loss(point_c_trans, point_d_trans, margin, r_trans)
        
    i1_non_trans = individual_embed(non_trans_data[:, 0])
    r_non_trans = rel_embed(non_trans_data[:, 1])
    i2_non_trans = individual_embed(non_trans_data[:, 2])

    point_c_non_trans = Point(i1_non_trans)
    point_d_non_trans = Point(i2_non_trans - r_non_trans)

    if neg:
        non_trans_loss = Point.point_distance(point_c_non_trans, point_d_non_trans)
    else:
        non_trans_loss = Point.point_distance(point_c_non_trans, point_d_non_trans)

    if train:
        return transitive_loss, non_trans_loss

    else:
        final_output = th.zeros(data.shape[0], device=data.device)
        final_output[mask] = transitive_loss
        final_output[~mask] = non_trans_loss
            
        return final_output
