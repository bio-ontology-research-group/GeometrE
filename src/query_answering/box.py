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
    def __init__(self, center, offset, check_shape=True, normalize=False):
        logger.debug(f"{self.__class__.__name__} center: {center.shape}, offset: {offset.shape}")
        if check_shape:
            assert center.shape == offset.shape, f"center: {center.shape}, offset: {offset.shape}"
        self.center = center
        if normalize:
            self.center = self.center / th.norm(self.center, p=2, dim=-1, keepdim=True)
            
        self.offset = offset
        logger.debug(f"{self.__class__.__name__} lower: {self.lower.shape}, upper: {self.upper.shape}")

    @property
    def lower(self):
        return self.center - self.offset

    @property
    def upper(self):
        return self.center + self.offset
        
    def translate(self, translation, scaling):
        new_center = self.center + translation
        new_offset = self.offset * th.abs(scaling)
        
        return Box(new_center, new_offset)
        
        # if len(self.center.shape) == 3:
            # r = r.unsqueeze(1)

        # if test:
            # logger.debug(f"{self.__class__.__name__}-{self.translate.__name__}-Test. center: {self.center.shape}, offset: {self.offset.shape}, r: {r.shape}")
            # box_bs = self.center.shape[0]
            # r_bs = r.shape[0]
            # dim = self.center.shape[-1]
            # if not box_bs == r_bs:
                # self.center = self.center.permute(1, 0, 2)
                # self.offset = self.offset.permute(1, 0, 2).repeat(r_bs, 1, 1)
                # new_center = (self.center + r) #.reshape(-1, 1, dim)
            # else:
                # new_center = self.center + r
                # logger.debug(f"{self.__class__.__name__}-{self.translate.__name__} r: {r.shape}, center: {self.center.shape}, new_center: {new_center.shape}")
                # assert new_center.shape == self.center.shape, f"new_center: {new_center.shape}, center: {self.center.shape}"
        # return Box(new_center, self.offset)
        
        
    # @check_output_shape
    @staticmethod
    def box_inclusion_score(box_1, box_2, margin):
        return Box.box_equiv_score(box_1, box_2, margin)
        
        if len(box_2.center.shape) == 3 and len(box_1.center.shape) == 2:
            box_1.center = box_1.center.unsqueeze(1)
            box_1.offset = box_1.offset.unsqueeze(1)
            # box_1.lower = box_1.lower.unsqueeze(1)
            # box_1.upper = box_1.upper.unsqueeze(1)
            
        box_1_bs, *_ = box_1.center.shape
        box_2_bs, *_ = box_2.center.shape

        if box_1_bs != box_2_bs:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)
            # box_2.lower = box_2.lower.permute(1, 0, 2)
            # box_2.upper = box_2.upper.permute(1, 0, 2)
            
        logger.debug(f"box_1: {box_1.center.shape}, box_2: {box_2.center.shape}")
        logger.debug(f"box_1: {box_1.offset.shape}, box_2: {box_2.offset.shape}")

        euc = th.abs(box_1.center - box_2.center)
        logger.debug(f"euc: {euc.shape}")
        
        dst = th.linalg.norm(th.relu(euc + box_1.offset - box_2.offset + margin), axis=-1)
        logger.debug(f"dst: {dst.shape}")
        return dst

    @staticmethod
    def box_equiv_score(box_1, box_2, margin):
        logger.debug(f"{Box.__name__}-{Box.box_inclusion_score.__name__} box_1 center: {box_1.center.shape}, box_2 center: {box_2.center.shape}")
        logger.debug(f"{Box.__name__}-{Box.box_inclusion_score.__name__} box_1 offset: {box_1.offset.shape}, box_2 offset: {box_2.offset.shape}")
        
        if len(box_2.center.shape) == 3 and len(box_1.center.shape) == 2:
            box_1.center = box_1.center.unsqueeze(1)
            box_1.offset = box_1.offset.unsqueeze(1)
            # box_1.lower = box_1.lower.unsqueeze(1)
            # box_1.upper = box_1.upper.unsqueeze(1)
            
        box_1_bs, *_ = box_1.center.shape
        box_2_bs, *_ = box_2.center.shape

        if box_1_bs != box_2_bs:
            box_2.center = box_2.center.permute(1, 0, 2)
            box_2.offset = box_2.offset.permute(1, 0, 2)
            # box_2.lower = box_2.lower.permute(1, 0, 2)
            # box_2.upper = box_2.upper.permute(1, 0, 2)
            
        logger.debug(f"box_1: {box_1.center.shape}, box_2: {box_2.center.shape}")
        logger.debug(f"box_1: {box_1.offset.shape}, box_2: {box_2.offset.shape}")

        lower_loss = th.linalg.norm(box_1.lower - box_2.lower, dim=-1)
        upper_loss = th.linalg.norm(box_1.upper - box_2.upper, dim=-1)
        return (lower_loss + upper_loss) / 2


    @check_output_shape
    @staticmethod
    def order_loss(box_1, box_2, r_trans, margin):
        order_loss = th.linalg.norm(th.relu(box_1.center + box_1.offset - (box_2.center + box_2.offset) + margin), dim=1)

        diff = (box_2.center - box_1.center)
        diff_norm = th.norm(diff, p=2, dim=1)
        diff = diff/diff_norm.unsqueeze(1)
        angle_loss = 1 - th.sigmoid(th.sum(diff * r_trans, dim=1))
        return order_loss + angle_loss

    @staticmethod
    def corner_loss(box):
        loss = th.sum(th.relu(box.lower - box.upper), dim=-1)
        loss = loss.unsqueeze(1) if len(loss.shape) == 1 else loss
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

        corner_loss = Box.corner_loss(intersection_box)
        return intersection_box #, corner_loss
        
    @staticmethod
    def intersection_with_negation(position, *boxes):
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
        if naive:
            new_lower[~condition] = intermediate_intersection.lower[~condition]
            new_upper[~condition] = intermediate_intersection.upper[~condition]
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
        new_center = (new_lower + new_upper) / 2
        new_offset = (new_upper - new_lower) / 2

        new_box = Box(new_center, new_offset)
        corner_loss_2 = Box.corner_loss(new_box)
        
        return Box(new_center, new_offset) #, corner_loss_1 + corner_loss_2
