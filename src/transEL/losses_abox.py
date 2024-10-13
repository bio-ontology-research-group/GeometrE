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
        self.lower_corner = lower_corner
        
        if pos_delta:
            delta = th.abs(delta)
        self.delta = delta
        self.upper_corner = lower_corner + delta
        self.center = (self.upper_corner + self.lower_corner) / 2

    def volume(self):
        if self.upper_corner is None or self.lower_corner is None:
            raise ValueError("Upper and lower corners not defined. Box must be created with intersection method")
        return th.sum(th.log(self.delta))
        
    @check_output_shape
    @staticmethod
    def inclusion(box1, box2, margin, *args):
        # diff_1 = th.abs(box1.lower_corner - box2.lower_corner)
        # diff_2 = th.abs(box1.upper_corner - box2.upper_corner)
        # min_diff_1 = diff_1.min()
        # min_diff_2 = diff_2.min()

        # max_diff_1 = diff_1.max()
        # max_diff_2 = diff_2.max()

        # print(f"min_diff_1: {min_diff_1}, max_diff_1: {max_diff_1}")
        # print(f"min_diff_2: {min_diff_2}, max_diff_2: {max_diff_2}")

        # euc_dist = th.abs(box1.center - box2.center)
        # loss = th.nn.functional.softplus(euc_dist + box1.delta/2 - box2.delta/2 - margin)
        # loss = th.relu(euc_dist + box1.delta/2 - box2.delta/2 - margin)
        # return loss.sum(dim=1)
    
        
        lower_corner_condition = th.nn.functional.softplus(box2.lower_corner - box1.lower_corner - margin)
        upper_corner_condition = th.nn.functional.softplus(box1.upper_corner - box2.upper_corner - margin)

        # return th.norm(lower_corner_condition, p=2, dim=1) + th.norm(upper_corner_condition, p=2, dim=1)
        return lower_corner_condition.pow(2).sum(dim=1) + upper_corner_condition.pow(2).sum(dim=1)
        # return th.linalg.norm(lower_corner_condition, axis=1) + th.linalg.norm(upper_corner_condition, axis=1)


    @check_output_shape
    def point_distance(box1, box2):
        # return th.norm(box1.lower_corner - box2.lower_corner, p=2, dim=1)
        return (box1.lower_corner - box2.lower_corner).pow(2).sum(dim=1)


    @check_output_shape
    def point_to_segment_distance(box1, box2):
        """
        Computes the distance from a batch of points P to a batch of line segments AB in n-dimensional space.

        Parameters:
        P (torch.Tensor): Tensor of shape (b, n) representing the batch of points, where b is batch size and n is dimensions.
        A (torch.Tensor): Tensor of shape (b, n) representing the first endpoints of the line segments.
        B (torch.Tensor): Tensor of shape (b, n) representing the second endpoints of the line segments.

        Returns:
        torch.Tensor: Tensor of shape (b,) representing distances from each point to the corresponding line segment.
        """
        P = box1.lower_corner
        A = box2.lower_corner
        B = box2.upper_corner
        
        # Vector from A to B for each batch
        AB = B - A  # shape (b, n)
        # Vector from A to P for each batch
        AP = P - A  # shape (b, n)
        # Vector from B to P for each batch
        BP = P - B  # shape (b, n)

        # Squared length of segment AB for each batch
        AB_squared = th.sum(AB * AB, dim=1)  # shape (b,)

        # To avoid division by zero in case A and B are the same, we add a small epsilon.
        epsilon = 1e-7
        AB_squared = th.where(AB_squared == 0, th.tensor(epsilon, dtype=AB_squared.dtype), AB_squared)

        # Projection of AP onto AB, normalized by the length of AB (batch version)
        t = th.sum(AP * AB, dim=1) / AB_squared  # shape (b,)

        # Case 1: t < 0, the closest point is A
        distance_A = th.norm(AP, dim=1)  # shape (b,)

        # Case 2: t > 1, the closest point is B
        distance_B = th.norm(BP, dim=1)  # shape (b,)

        # Case 3: 0 <= t <= 1, the closest point is on the segment, compute perpendicular distance
        t_clamped = th.clamp(t, 0.0, 1.0).unsqueeze(1)  # shape (b, 1) to match (b, n)
        projection = A + t_clamped * AB  # shape (b, n)
        distance_perpendicular = th.norm(P - projection, dim=1)  # shape (b,)

        # Combine the distances based on the value of t
        distances = th.where(t < 0.0, distance_A, th.where(t > 1.0, distance_B, distance_perpendicular))  # shape (b,)

        return distances

     
    @check_output_shape
    @staticmethod
    def non_inclusion(box1, box2, margin, *args):

        return Box.inclusion(box1, box2, margin, *args) # - box1.volume()+ box2.volume()
                         
    @staticmethod
    def intersection(box1, box2):
        intersection_lower_corner = th.maximum(box1.lower_corner, box2.lower_corner)
        intersection_upper_corner = th.minimum(box1.upper_corner, box2.upper_corner)
        delta = intersection_upper_corner - intersection_lower_corner
        return Box(intersection_lower_corner, delta, pos_delta=False)

    @staticmethod
    def unbound(box, rel_mask):
        min_bound = 0
        unbound_dimension = th.where(rel_mask == 1)
                                        
        new_lower_corner = box.lower_corner.clone()
        new_upper_corner = box.upper_corner.clone()
        
        new_lower_corner[unbound_dimension] = min_bound
        delta = new_upper_corner - new_lower_corner
        delta[unbound_dimension] = new_upper_corner[unbound_dimension]
        return Box(new_lower_corner, delta)
 
    @check_output_shape
    @staticmethod
    def non_transitive_inclusion(box1, box2, margin, relation):
        return Box.non_inclusion(box1, box2, margin, relation)

    def corners_loss(self):
        if self.upper_corner is None or self.lower_corner is None:
            raise ValueError("Upper and lower corners not defined. Box must be created with intersection method")

        loss = th.linalg.norm(th.relu(self.lower_corner - self.upper_corner), axis=1)
        return loss
        
@check_output_shape
def gci0_loss(data, class_lower, class_delta, margin, neg=False):
    raise NotImplementedError


@check_output_shape
def gci0_bot_loss(data, class_lower, class_delta, margin, max_bound, neg=False):
    raise NotImplementedError
     

@check_output_shape
def gci1_loss(data, class_lower, class_delta, margin, neg=False):
    raise NotImplementedError

@check_output_shape
def gci1_bot_loss(data, class_lower, class_delta, margin, neg=False):
    raise NotImplementedError

@check_output_shape
def normal_gci2_loss(data, class_lower, class_delta, rel_embed, rel_mask, transitive_ids, margin, neg=False):
    raise NotImplementedError

@check_output_shape
def gci2_loss(data, class_lower, class_delta, rel_embed, rel_mask, transitive_ids, margin, transitive, neg=False, evaluate=False): #adapted

    if not transitive:
        return normal_object_property_assertion_loss(data, class_lower, rel_embed, rel_mask, transitive_ids, margin, neg = neg)

    return object_property_assertion_loss(data, class_lower, rel_embed, rel_mask, transitive_ids, margin, transitive, neg = neg)
 

def normal_gci3_loss(data, class_lower, class_delta, rel_embed, rel_mask, transitive_ids, margin, neg=False):
    raise NotImplementedError

def gci3_loss(data, class_lower, class_delta, rel_embed, rel_mask, transitive_ids, margin, transitive, neg=False):
    raise NotImplementedError

@check_output_shape
def gci3_bot_loss(data, class_delta, margin, neg=False):
    raise NotImplementedError

@check_output_shape
def class_assertion_loss(data, class_lower, class_delta, individual_embed, margin, neg=False):
    raise NotImplementedError
                                                    
def normal_object_property_assertion_loss(data, individual_embed, rel_embed, rel_mask, transitive_ids, margin, neg=False):

    i1 = individual_embed(data[:, 0])
    r = rel_embed(data[:, 1])
    # r_mask = rel_mask(data[:, 1])
    i2 = individual_embed(data[:, 2])

    # trans_mask = th.isin(data[:, 1], transitive_ids)
    # r[trans_mask] = th.abs(r[trans_mask] * r_mask[trans_mask])

    off_i1 = th.zeros_like(i1)
    off_i2 = th.zeros_like(i2)

    box_i1 = Box(i1, off_i1)
    box_i2 = Box(i2 - r, off_i2)

    if neg:
        loss = Box.point_distance(box_i1, box_i2)
    else:
        loss = Box.point_distance(box_i1, box_i2)

    return loss
    
@check_output_shape
def object_property_assertion_loss(data, individual_embed, rel_embed, rel_mask, transitive_ids, margin, transitive, neg=False):

    if not transitive:
        return normal_object_property_assertion_loss(data, individual_embed, rel_embed, rel_mask, transitive_ids, margin, neg=neg)
    
    r = data[:, 1]
    mask = th.isin(r, transitive_ids)

    trans_data = data[mask]
    non_trans_data = data[~mask]

    i1_trans = individual_embed(trans_data[:, 0])
    
    r_embed = rel_embed(trans_data[:, 1])
    r_mask = rel_mask(trans_data[:, 1])
    r_trans = th.abs(r_embed * r_mask)

    i2_trans = individual_embed(trans_data[:, 2])

    off_i1_trans = th.zeros_like(i1_trans)
    off_i2_trans = th.zeros_like(off_i1_trans)

    box_c_trans = Box(i1_trans, off_i1_trans)
    box_d_trans = Box(i2_trans - r_trans, off_i2_trans)

    if neg:
        # transitive_loss = Box.non_inclusion(box_c_trans, box_d_trans, r_trans, margin)
        transitive_loss = Box.point_to_segment_distance(box_c_trans, box_d_trans)
    else:
        # transitive_loss = Box.inclusion(box_c_trans, box_d_trans, r_trans, margin)
        transitive_loss = Box.point_to_segment_distance(box_c_trans, box_d_trans)

    i1_non_trans = individual_embed(non_trans_data[:, 0])
    r_non_trans = rel_embed(non_trans_data[:, 1])
    i2_non_trans = individual_embed(non_trans_data[:, 2])
    off_i1_non_trans = th.zeros_like(i1_non_trans)
    off_i2_non_trans = th.zeros_like(i2_non_trans)

    box_c_non_trans = Box(i1_non_trans, off_i1_non_trans)
    box_d_non_trans = Box(i2_non_trans - r_non_trans, off_i2_non_trans)

    if neg:
        non_trans_loss = Box.point_distance(box_c_non_trans, box_d_non_trans)
    else:
        non_trans_loss = Box.point_distance(box_c_non_trans, box_d_non_trans)
        
    final_output = th.zeros(data.shape[0], device=data.device)
    final_output[mask] = transitive_loss
    final_output[~mask] = non_trans_loss
            
    return final_output


def regularization_loss(ind_lower, max_bound, reg_factor):
    lower = ind_lower.weight
    lower_condition =  th.relu(-lower).mean() * reg_factor
    upper_condition = th.relu(lower - max_bound).mean() * reg_factor
    return lower_condition + upper_condition

# def regularization_loss(rel_embed, rel_mask, transitive_ids, reg_factor=0.1):
    # r = th.abs(rel_embed(transitive_ids)) * rel_mask(transitive_ids)
    # norm_r = F.normalize(r, p=2, dim=1)
    # norm_loss = 0 #th.relu(0.5 - norm_r).mean()
        
    
    # cos_sim_matrix = th.mm(norm_r, norm_r.t())
    # identity = th.eye(r.shape[0], device=r.device)
    # orthogonality_loss = th.linalg.norm(cos_sim_matrix - identity, ord='fro')
    # return reg_factor * orthogonality_loss.mean() + reg_factor * norm_loss
