import torch as th
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Box():
    def __init__(self, center, offset=None, check_shape=True, as_point=False):
        
        if check_shape and not as_point:
            assert center.shape == offset.shape, f"center: {center.shape}, offset: {offset.shape}"

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

    def translate(self, translation_mul, translation_add, scaling_mul, scaling_add):
        new_center = self.center * translation_mul + translation_add
        new_offset = th.abs(self.offset * scaling_mul + scaling_add)
        return Box(new_center, new_offset)
 
    @staticmethod
    def box_inclusion_score(box_1, box_2, alpha, negative=False):
        dist_outside = th.linalg.norm(th.relu(box_2.center - box_1.upper ) + th.relu(box_1.lower - box_2.center), dim=-1, ord=1)
        dist_inside = th.linalg.norm(box_1.center - th.min(box_1.upper, th.max(box_1.lower, box_2.center)), dim=-1, ord=1)

        loss = dist_outside + alpha*dist_inside
                 
        if not negative:
            box_1_corner_loss = Box.corner_loss(box_1)
            loss += box_1_corner_loss
        
        return loss

    @staticmethod
    def box_order_score(box_1, box_2, margin, inverse=False, permute_back=False):
        
        if len(box_2.center.shape) == 3 and len(box_1.center.shape) == 2:
            box_1.center = box_1.center.unsqueeze(1)
            box_1.offset = box_1.offset.unsqueeze(1)
            
        box_1_bs, *_ = box_1.center.shape
        box_2_bs, *_ = box_2.center.shape

        if box_1_bs != box_2_bs:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)
                        
        if inverse:
            order_loss = th.linalg.norm(th.relu(box_1.lower - box_2.lower + margin), dim=-1, ord=1)
        else:
            order_loss = th.linalg.norm(th.relu(box_2.upper - box_1.upper + margin), dim=-1, ord=1)
            
                
        if permute_back:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)

        box_1_corner_loss = Box.corner_loss(box_1)
        box_2_corner_loss = Box.corner_loss(box_2)

        return order_loss + (box_1_corner_loss + box_2_corner_loss) / 2

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
        corner_logit = Box.corner_loss(intersection_box)
        
        condition = th.any(upper < lower, dim=1)
        total_boxes = len(condition)
        disjoints = 0
        if condition.any():
            disjoints = condition.sum().item()
        return intersection_box, corner_logit, disjoints, total_boxes

    
    @staticmethod
    def intersection(*boxes):
        intersection_box = boxes[0]
        all_corner_logit = Box.corner_loss(intersection_box)
        all_disjoints = 0
        all_total_boxes = 0
        for box in boxes[1:]:
            intersection_box, corner_logit, disjoints, total_boxes = Box._pair_intersection(intersection_box, box)
            all_corner_logit = all_corner_logit + corner_logit
            all_disjoints += disjoints
            all_total_boxes += total_boxes
        return intersection_box, all_corner_logit, all_disjoints, all_total_boxes

    @staticmethod
    def intersection_with_negation(position, *boxes):
        boxes = list(boxes)
        num_boxes = len(boxes)
        position -= 1
        box_to_negate = boxes.pop(position)
        assert num_boxes == len(boxes) + 1
        intermediate_intersection, corner_logit, disjoints, total_boxes = Box.intersection(*boxes)

        return intermediate_intersection, corner_logit, disjoints, total_boxes
