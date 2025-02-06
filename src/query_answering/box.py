import torch as th
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

enable_check_output_shape = True



def check_output_shape(func):
    def wrapper(*args, **kwargs):
        logger.debug(f"\nCheck output shape: {func.__name__}")
        if len(args) > 3:
            test = args[-1]
        else:
            test = False
        logger.debug(f"\nWrapper: test value: {test}")
        if isinstance(args[0], tuple):
            init, tails = args[0]
            if test:
                expected_shape = (init.shape[0], tails.shape[0])
            else:
                expected_shape = tails.shape
            
        elif isinstance(args[0], Box):
            init, tails = args[0].center, args[1].center
            expected_shape = tails.shape[:-1] # tails can have shape (B, N, D) and we aggregate over the last dimension

        output = func(*args, **kwargs)
        if enable_check_output_shape:
            if isinstance(output, tuple):
                for o in output:
                    if o.shape != expected_shape:
                        raise ValueError(f"Expected output to have shape {expected_shape}, got {output.shape}")
            else:
                if output.shape != expected_shape:
                        raise ValueError(f"Expected output to have shape {expected_shape}, got {output.shape}")
                    
        return output
    return wrapper



class Box():
    def __init__(self, center, offset, check_shape=True):
        logger.debug(f"{self.__class__.__name__} center: {center.shape}, offset: {offset.shape}")
        if check_shape:
            assert center.shape == offset.shape, f"center: {center.shape}, offset: {offset.shape}"
        self.center = center
        self.offset = offset
        
    @property
    def lower(self):
        return self.center - self.offset

    @property
    def upper(self):
        return self.center + self.offset

    def slice(self, index_tensor):
        return Box(self.center[index_tensor], self.offset[index_tensor])

    
    def translate(self, translation, factor, scaling, scaling_bias, transitive_mask=None):
        if transitive_mask is not None:
            # translation[transitive_mask] = th.zeros_like(translation[transitive_mask]) + 1e-7
            translation[transitive_mask] = 0 #th.abs(translation[transitive_mask]) /th.norm(translation[transitive_mask], dim=-1).unsqueeze(-1)
            # r_range = th.arange(translation.shape[0], device=translation.device)[transitive_mask]
            # logger.debug(f"r_range: {r_range.shape}, r_idxs: {r_idxs.shape}")
            # mask = th.ones_like(translation)
            # logger.debug(f"In translation: r_idxs: {r_idxs.shape}, mask: {mask.shape}")
            # mask[transitive_mask] = 0
            # mask[r_range, r_idxs] = 1
            # assert mask[transitive_mask].sum() == len(r_idxs), f"Mask sum: {mask[transitive_mask].sum()}, r_idxs: {len(r_idxs)}"
            
            # translation[transitive_mask][r_range, r_idxs] = 1e-7
            # translation = translation * mask
            # non_zero = (translation[transitive_mask] != 0).sum()

            factor[transitive_mask] = 1
            # scaling[transitive_mask] = 1
            # scaling_bias[transitive_mask] = 0
            
            # assert non_zero == len(r_idxs), f"Non zero: {non_zero}, r_idxs: {len(r_idxs)}"
            # scaling[transitive_mask] = 1
            # scaling[r_range, r_idxs] = 1
            # scaling[transitive_mask] = 1 #th.sigmoid(scaling[transitive_mask]) + 1
        new_center = self.center * factor + translation
        new_offset = th.abs(self.offset * th.abs(scaling) + scaling_bias) # NOTE important to provide a positive offset here.
 
        return Box(new_center, new_offset)
        
    # @check_output_shape
    @staticmethod
    def box_inclusion_score(box_1, box_2, margin):
        # return Box.box_equiv_score(box_1, box_2, margin)
        
        if len(box_2.center.shape) == 3 and len(box_1.center.shape) == 2:
            box_1.center = box_1.center.unsqueeze(1)
            box_1.offset = box_1.offset.unsqueeze(1)
                        
        box_1_bs, *_ = box_1.center.shape
        box_2_bs, *_ = box_2.center.shape

        if box_1_bs != box_2_bs:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)

        box_1_corner_loss = Box.corner_loss(box_1)
        box_2_corner_loss = Box.corner_loss(box_2)

        lower_loss = th.linalg.norm(th.relu(box_2.lower - box_1.lower + margin), dim=-1)
        upper_loss = th.linalg.norm(th.relu(box_1.upper - box_2.upper + margin), dim=-1)

        loss = (lower_loss + upper_loss) / 2 + (box_1_corner_loss + box_2_corner_loss) / 2
        return loss
    
    @staticmethod
    def box_order_score(box_1, box_2, margin, r_trans, trans_mask, inverse=False, permute_back=False):
        logger.debug(f"{Box.__name__}-{Box.box_order_score.__name__} box_1 center: {box_1.center.shape}, box_2 center: {box_2.center.shape}, r_trans: {r_trans.shape}")

        r_trans = th.abs(r_trans)
                
        if len(box_2.center.shape) == 3 and len(box_1.center.shape) == 2:
            box_1.center = box_1.center.unsqueeze(1)
            box_1.offset = box_1.offset.unsqueeze(1)
            r_trans = r_trans.unsqueeze(1)
                        
        box_1_bs, *_ = box_1.center.shape
        box_2_bs, *_ = box_2.center.shape

        if box_1_bs != box_2_bs:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)
                        
        if inverse:
            order_loss = th.linalg.norm(th.relu(box_2.lower - box_1.lower + margin), dim=-1)
            r_trans = - r_trans
        else:
            order_loss = th.linalg.norm(th.relu(box_1.upper - box_2.upper + margin), dim=-1)

        diff = (box_2.center - box_1.center)
                
        angle_loss = 1 - th.sigmoid(th.sum(diff * r_trans, dim=-1))

        if permute_back:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)
        
        return order_loss + angle_loss #+ lower_loss #+ volume_loss

    
    
    @staticmethod
    def box_equiv_score(box_1, box_2, margin):
        if len(box_2.center.shape) == 3 and len(box_1.center.shape) == 2:
            box_1.center = box_1.center.unsqueeze(1)
            box_1.offset = box_1.offset.unsqueeze(1)
                        
        box_1_bs, *_ = box_1.center.shape
        box_2_bs, *_ = box_2.center.shape
                
        if box_1_bs != box_2_bs:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)

        box_1_corner_loss = Box.corner_loss(box_1)
        box_2_corner_loss = Box.corner_loss(box_2)
        
        lower_loss = th.linalg.norm(box_1.lower - box_2.lower, dim=-1)
        upper_loss = th.linalg.norm(box_1.upper - box_2.upper, dim=-1)
        loss = (lower_loss + upper_loss) / 2 + (box_1_corner_loss + box_2_corner_loss) / 2

        return loss
        # return (lower_loss + upper_loss) / 2


    @staticmethod
    def corner_loss(box):
        loss = th.sum(th.relu(box.lower - box.upper), dim=-1)
        # loss = loss.unsqueeze(1) if len(loss.shape) == 1 else loss
        logger.debug(f"{box.__class__.__name__}-{box.corner_loss.__name__} loss: {loss.shape}")
        return loss

    @staticmethod
    def inverse_corner_loss(box):
        loss = th.sum(th.relu(box.upper - box.lower), dim=-1)
        # loss = loss.unsqueeze(1) if len(loss.shape) == 1 else loss
        logger.debug(f"{box.__class__.__name__}-{box.corner_loss.__name__} loss: {loss.shape}")
        return loss

    
    @staticmethod
    def _get_lower_and_upper_corners(box1, box2):
        lower = th.max(box1.center - box1.offset, box2.center - box2.offset)
        upper = th.min(box1.center + box1.offset, box2.center + box2.offset)
        return lower, upper
    
    @staticmethod
    def _pair_intersection(box_1, box_2):
        lower, upper = Box._get_lower_and_upper_corners(box_1, box_2)
        # new_center = (lower + upper) / 2
        # new_offset = (upper - lower) / 2
        return Box((lower + upper) / 2, (upper - lower) / 2)

    
    @staticmethod
    def intersection(*boxes):
        intersection_box = boxes[0]
        for box in boxes[1:]:
            intersection_box = Box._pair_intersection(intersection_box, box)

        # corner_loss = Box.corner_loss(intersection_box)
        return intersection_box 

    @staticmethod
    def intersection_with_negation_fast(position, *boxes):
        boxes = list(boxes)
        num_boxes = len(boxes)
        position -= 1
        box_to_negate = boxes.pop(position)
        assert num_boxes == len(boxes) + 1
        intermediate_intersection = Box.intersection(*boxes)

        lower, upper = Box._get_lower_and_upper_corners(intermediate_intersection, box_to_negate)
        condition = (upper < lower).any(dim=1)

        new_lower = th.zeros_like(lower)
        new_upper = th.zeros_like(upper)

        new_lower[condition] = intermediate_intersection.lower[condition]
        new_upper[condition] = intermediate_intersection.upper[condition]

        naive = True
        dimensionwise = False
        if naive:
            new_lower[~condition] = intermediate_intersection.lower[~condition]
            new_upper[~condition] = intermediate_intersection.upper[~condition]
        elif dimensionwise:

            intersection = Box.intersection(intermediate_intersection, box_to_negate)
            intersection_lower = intersection.lower
            intersection_upper = intersection.upper

            lower_inter = intermediate_intersection.lower
            upper_inter = intermediate_intersection.upper

            # Vectorized conditions for lower and upper sub-boxes
            lower_part_condition = lower_inter < intersection_lower
            upper_part_condition = upper_inter > intersection_upper

            # Construct lower sub-boxes
            sub_lower_lower = lower_inter.clone()
            sub_upper_lower = th.where(lower_part_condition, intersection_lower, upper_inter)
            sub_center_lower = (sub_lower_lower + sub_upper_lower) / 2
            sub_offset_lower = (sub_upper_lower - sub_lower_lower) / 2

            # Construct upper sub-boxes
            sub_lower_upper = th.where(upper_part_condition, intersection_upper, lower_inter)
            sub_upper_upper = upper_inter.clone()
            sub_center_upper = (sub_lower_upper + sub_upper_upper) / 2
            sub_offset_upper = (sub_upper_upper - sub_lower_upper) / 2

            # Combine lower and upper sub-boxes
            all_centers = th.cat([sub_center_lower.unsqueeze(-1), sub_center_upper.unsqueeze(-1)], dim=-1)
            all_offsets = th.cat([sub_offset_lower.unsqueeze(-1), sub_offset_upper.unsqueeze(-1)], dim=-1)

            # Calculate volumes (product of offsets) and find the largest sub-box
            offset_prods = th.prod(all_offsets, dim=1)
            max_volumes, max_idxs = offset_prods.max(dim=-1)

            # Select centers and offsets corresponding to max volume
            bs = lower_inter.size(0)
            fst_dim_range = th.arange(bs)
            selected_centers = all_centers[fst_dim_range, :, max_idxs]
            selected_offsets = all_offsets[fst_dim_range, :, max_idxs]

            return Box(selected_centers, selected_offsets)

    
    @staticmethod
    def intersection_with_negation(position, *boxes):
        boxes = list(boxes)
        num_boxes = len(boxes)
        position -= 1
        box_to_negate = boxes.pop(position)
        assert num_boxes == len(boxes) + 1
        intermediate_intersection = Box.intersection(*boxes)

        return intermediate_intersection
        
        lower, upper = Box._get_lower_and_upper_corners(intermediate_intersection, box_to_negate)
        condition = (upper < lower).any(dim=1)

        new_lower = th.zeros_like(lower)
        new_upper = th.zeros_like(upper)

        new_lower[condition] = intermediate_intersection.lower[condition]
        new_upper[condition] = intermediate_intersection.upper[condition]

        naive = True
        dimensionwise = False
        if naive:
            new_lower[~condition] = intermediate_intersection.lower[~condition]
            new_upper[~condition] = intermediate_intersection.upper[~condition]
        elif dimensionwise:
            intersection = Box.intersection(intermediate_intersection, box_to_negate)
            logger.debug(f"intersection: {intersection.center.shape}")
            intersection_lower = intersection.lower
            intersection_upper = intersection.upper

            lower_inter = intermediate_intersection.lower
            upper_inter = intermediate_intersection.upper
            
            sub_boxes = []

            # Iterate through dimensions to construct sub-boxes
            bs, dim = lower.shape
            
            for i in range(dim):
                # Lower part (before the intersection in dimension i)
                lower_part_condition = (lower_inter[:, i] < intersection_lower[:, i])
                sub_lower = lower_inter#.clone()
                sub_upper = upper_inter.clone()
                sub_upper[lower_part_condition] = intersection_lower[lower_part_condition]
                sub_center = (sub_lower + sub_upper) / 2
                sub_offset = (sub_upper - sub_lower) / 2
                sub_boxes.append(Box(sub_center, sub_offset))

                # Upper part (after the intersection in dimension i)
                upper_part_condition = (upper_inter[:, i] > intersection_upper[:, i])
                sub_lower = lower_inter.clone()
                sub_upper = upper_inter#.clone()
                sub_lower[upper_part_condition] = intersection_upper[upper_part_condition]
                sub_center = (sub_lower + sub_upper) / 2
                sub_offset = (sub_upper - sub_lower) / 2
                sub_boxes.append(Box(sub_center, th.abs(sub_offset)))

            logger.debug(f"sub_boxes: {len(sub_boxes)}")
            logger.debug(f"sub_boxes[0]: {sub_boxes[0].center.shape}")

            all_boxes_center = th.cat([b.center.unsqueeze(-1) for b in sub_boxes], dim=-1)
            all_boxes_offset = th.cat([b.offset.unsqueeze(-1) for b in sub_boxes], dim=-1)

            logger.debug(f"all_boxes_center: {all_boxes_center.shape}")
            logger.debug(f"all_boxes_offset: {all_boxes_offset.shape}")

            offset_prods = th.prod(all_boxes_offset, dim=1)
            logger.debug(f"offset_prods: {offset_prods.shape}")
            max_volumes, max_idxs = offset_prods.max(dim=-1)
            logger.debug(f"max_volumes: {max_volumes.shape}")
            logger.debug(f"max_idxs: {max_idxs.shape}")
            fst_dim_range = th.arange(bs)
            seleced_centers = all_boxes_center[fst_dim_range, :, max_idxs]
            logger.debug(f"seleced_centers: {seleced_centers.shape}")
            selected_offsets = all_boxes_offset[fst_dim_range, :, max_idxs]
            logger.debug(f"selected_offsets: {selected_offsets.shape}")
            return Box(seleced_centers, selected_offsets)
                
                            

        else:
            #this is not closed
            region_to_ignore = Box.intersection(intermediate_intersection, box_to_negate)
            condition_2_1 = th.allclose(region_to_ignore.upper, box_to_negate.upper, atol=1e-5)
            condition_2_2 = th.allclose(region_to_ignore.lower, box_to_negate.lower, atol=1e-5)


            condition_2 = (box_to_negate.upper < intermediate_intersection.upper).any(dim=1)
            mask = ~condition & condition_2
            mask_2 = ~condition & ~condition_2
            new_lower[mask] = box_to_negate.upper[mask]
            new_upper[mask] = intermediate_intersection.upper[mask]

            new_lower[mask_2] = intermediate_intersection.lower[mask_2]
            new_upper[mask_2] = box_to_negate.lower[mask_2]

            assert condition.sum() + mask.sum() + mask_2.sum() == len(new_lower)

        # new_center = (new_lower + new_upper) / 2
        # new_offset = (new_upper - new_lower) / 2

        
        return Box((new_lower + new_upper) / 2, (new_upper - new_lower) / 2)

          
