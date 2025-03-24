import torch as th
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Box():
    def __init__(self, center, offset=None, check_shape=True, as_point=False):
        
        self.center = center
        self.offset = offset if offset is not None else th.zeros_like(center)
        
    def __len__(self):
        return len(self.center)

    
    @property
    def lower(self):
        return self.center - self.offset
                                                    
    @property
    def upper(self):
        return self.center + self.offset

    @staticmethod
    def cat(boxes, dim=0):
        centers = th.cat([box.center for box in boxes], dim=dim)
        offsets = th.cat([box.offset for box in boxes], dim=dim)
        return Box(centers, offsets)
                
    def slice(self, index_tensor):
        return Box(self.center[index_tensor], self.offset[index_tensor])

    def mask(self, mask):
        return Box(self.center[mask], self.offset[mask])

    def assign_with_mask(self, mask, box):
        self.center[mask] = box.center
        self.offset[mask] = box.offset
 

    def project(self, projection_dims):
        box_shape = self.center.shape
        bs = box_shape[0]
        dim = box_shape[-1]

        self.center = self.center.view(bs, -1, dim)
        self.offset = self.offset.view(bs, -1, dim)
        
        intermediate_dim = self.center.shape[1]
        
        projection_dims = projection_dims.unsqueeze(1).expand(bs, intermediate_dim)
        bs_ids = th.arange(bs, device=self.center.device).unsqueeze(1).expand(bs, intermediate_dim)
        ns_ids = th.arange(intermediate_dim, device=self.center.device).expand(bs, intermediate_dim)

        non_projected_center = self.center[bs_ids, ns_ids, projection_dims].reshape(*box_shape[:-1]).unsqueeze(-1)
        non_projected_offset = self.offset[bs_ids, ns_ids, projection_dims].reshape(*box_shape[:-1]).unsqueeze(-1)
        non_projected_box = Box(non_projected_center, non_projected_offset)
            
        self.center[bs_ids, ns_ids, projection_dims] = 0
        self.offset[bs_ids, ns_ids, projection_dims] = 0
        self.center = self.center.view(*box_shape)
        self.offset = self.offset.view(*box_shape)
        
        return self, non_projected_box

    
    def transform(self, cen_mul, cen_add, off_mul, off_add, make_abs=True):
        new_center = self.center * cen_mul + cen_add
        new_offset = self.offset * off_mul + off_add
        if make_abs:
            new_offset = th.abs(new_offset)
        return Box(new_center, new_offset)

    @staticmethod
    def box_composed_score(box_1, box_2, alpha, trans_inv, trans_not_inv, negative=False):
        shape_1 = box_1.center.shape[:-1]
        shape_2 = box_2.center.shape[:-1]
        shape = tuple([max(s1, s2) for s1, s2 in zip(shape_1, shape_2)])
        
        not_trans_or_inv = ~(trans_inv | trans_not_inv)
        loss = -1 * th.ones(shape, device=box_1.center.device)

        normal_loss = Box.box_inclusion_score(box_1.mask(not_trans_or_inv), box_2.mask(not_trans_or_inv), alpha, negative)
        trans_loss = Box.box_order_score(box_1.mask(trans_not_inv), box_2.mask(trans_not_inv), negative)
        inv_loss = Box.box_order_score(box_1.mask(trans_inv), box_2.mask(trans_inv), negative, inverse=True)
        loss[not_trans_or_inv] = normal_loss
        loss[trans_not_inv] = trans_loss
        loss[trans_inv] = inv_loss

        return loss
    
    @staticmethod
    def box_composed_score_with_projection(box_1, box_2, alpha, trans_inv, trans_not_inv, projection_dims, negative=False):
        bs, *_ = box_1.center.shape
        hid_dim = box_1.center.shape[-1]
        
        shape_1 = box_1.center.shape[:-1]
        shape_2 = box_2.center.shape[:-1]
        shape = tuple([max(s1, s2) for s1, s2 in zip(shape_1, shape_2)])
                
        not_trans_or_inv = ~(trans_inv | trans_not_inv)
                                                
        order_loss = th.zeros(shape, device=box_1.center.device)

        if len(projection_dims) > 0:
            projected_boxes_1, single_dim_boxes_1 = box_1.mask(~not_trans_or_inv).project(projection_dims)
            projected_boxes_2, single_dim_boxes_2 = box_2.mask(~not_trans_or_inv).project(projection_dims)
            box_1.assign_with_mask(~not_trans_or_inv, projected_boxes_1)
            box_2.assign_with_mask(~not_trans_or_inv, projected_boxes_2)

            single_centers_1, single_offsets_1 = th.zeros_like(box_1.center), th.zeros_like(box_1.offset)
            single_centers_2, single_offsets_2 = th.zeros_like(box_2.center), th.zeros_like(box_2.offset)

            single_centers_1[~not_trans_or_inv] = single_dim_boxes_1.center
            single_offsets_1[~not_trans_or_inv] = single_dim_boxes_1.offset
            single_centers_2[~not_trans_or_inv] = single_dim_boxes_2.center
            single_offsets_2[~not_trans_or_inv] = single_dim_boxes_2.offset
            
            single_dim_boxes_1 = Box(single_centers_1, single_offsets_1)
            single_dim_boxes_2 = Box(single_centers_2, single_offsets_2)
        
        inclusion_loss = Box.box_inclusion_score(box_1, box_2, alpha, negative)

        if len(projection_dims) > 0:
            trans_loss = Box.box_order_score(single_dim_boxes_1.mask(trans_not_inv), single_dim_boxes_2.mask(trans_not_inv), negative)
            inv_loss = Box.box_order_score(single_dim_boxes_1.mask(trans_inv), single_dim_boxes_2.mask(trans_inv), negative, inverse=True)

            order_loss[not_trans_or_inv] = 0
            order_loss[trans_not_inv] = trans_loss
            order_loss[trans_inv] = inv_loss

        weight = 1/hid_dim
        return weight*order_loss + inclusion_loss

    @staticmethod
    def box_inclusion_score(box_1, box_2, alpha, negative=False):
                                                        
        dist_outside = th.linalg.norm(th.relu(box_2.center - box_1.upper ) + th.relu(box_1.lower - box_2.center), dim=-1, ord=1)
        dist_inside = th.linalg.norm(box_1.center - th.min(box_1.upper, th.max(box_1.lower, box_2.center)), dim=-1, ord=1)

        loss = dist_outside + alpha*dist_inside
                 
        if not negative:
            corner_loss = Box.corner_loss(box_1)
        else:
            corner_loss = th.zeros_like(loss)
        return loss + corner_loss


    @staticmethod
    def box_order_score(box_1, box_2, negative, inverse=False):
        
        if inverse:
            order_loss = th.linalg.norm(th.relu(box_1.lower - box_2.center), dim=-1, ord=1)
        else:
            order_loss = th.linalg.norm(th.relu(box_2.center - box_1.upper), dim=-1, ord=1)
        if not negative:
            corner_loss = Box.corner_loss(box_1)
        else:
            corner_loss = th.zeros_like(order_loss)
            
        return order_loss + corner_loss

    @staticmethod
    def corner_loss(box):
        loss = th.linalg.norm(th.relu(box.lower - box.upper), dim=-1, ord=1)
        return loss

    @staticmethod
    def _get_lower_and_upper_corners(box1, box2):
        lower = th.max(box1.center - box1.offset, box2.center - box2.offset)
        upper = th.min(box1.center + box1.offset, box2.center + box2.offset)
        return lower, upper
    
    @staticmethod
    def _pair_intersection(box_1, box_2):
        lower, upper = Box._get_lower_and_upper_corners(box_1, box_2)
        intersection_box = Box((lower + upper) / 2, (upper - lower) / 2)
        return intersection_box

    
    @staticmethod
    def intersection(*boxes):
        intersection_box = boxes[0]
        for box in boxes[1:]:
            intersection_box = Box._pair_intersection(intersection_box, box)
        return intersection_box

    @staticmethod
    def intersection_with_negation(position, *boxes, include_negated_component=False):
        boxes = list(boxes)
        position -= 1
        _ = boxes.pop(position)
        intermediate_intersection = Box.intersection(*boxes)
        return intermediate_intersection
