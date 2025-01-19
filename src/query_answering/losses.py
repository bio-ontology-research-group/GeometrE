import torch as th
from box import Box
import sys
import embeddings as E
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

enable_check_output_shape = True
def check_output_shape_2(func):
    def wrapper(*args, **kwargs):
        if len(args) > 3:
            test = args[-1]
        else:
           test = False
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

def check_output_shape(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
    


@check_output_shape
def query_1p_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, transitive_ids, test):
    init, tails = data
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    
    if transitive_ids is None:
        box_c = E.embedding_1p(init, class_embed, class_offset, rel_embed, scale_embed, None, None)
        box_d = Box(d, d_offset)
        loss = Box.box_inclusion_score(box_c, box_d, margin)
    else:
        r = init[:, -1]
        mask = th.isin(r, transitive_ids)
        logger.debug(f"mask in query_1p_loss: {mask.shape}")
        box_c = E.embedding_1p(init, class_embed, class_offset, rel_embed, scale_embed, mask, r_idxs = r[mask])
        box_d = Box(d, d_offset)
    
        r_embed = rel_embed(r[mask])
        

        trans_r_range = th.arange(r_embed.shape[0], device=init.device)
        zeros_mask = th.zeros_like(r_embed, device=init.device)
        zeros_mask[trans_r_range, r[mask]] = 1
        logger.debug(f"Sum of zeros mask: {zeros_mask.shape} - {zeros_mask.sum()}")
        r_trans = r_embed * zeros_mask
        r_trans = th.abs(r_trans)/th.norm(r_trans, dim=1).unsqueeze(1)
        logger.debug(f"Sum of r: {r_trans.shape} - {r_trans.sum()}")
        # r_trans[trans_r_range, r[mask]] = 0
        
        if test:
            loss = -1 * th.ones((init.shape[0], tails.shape[0]), device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d, margin, r_trans, mask, r[mask])
            box_d.center = box_d.center.permute(1, 0, 2)
            box_d.offset = box_d.offset.permute(1, 0, 2)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d, margin)

        else:
            loss = -1 * th.ones(tails.shape[:2], device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d.slice(mask), margin, r_trans, mask, r[mask])
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d.slice(~mask), margin)

            trans_loss = loss[mask].mean().item()
            non_trans_loss = loss[~mask].mean().item()
            logger.debug(f"Transitive losses 1p: trans: {trans_loss:4f} - non trans: {non_trans_loss}")
            
        assert (loss == -1).sum() == 0, f"Loss tensor contains -1 values."
        
    return loss

@check_output_shape
def query_2p_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, transitive_ids, test):
    init, tails = data
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    
    if transitive_ids is None:
        box_c = E.embedding_2p(init, class_embed, class_offset, rel_embed, scale_embed, None)
        box_d = Box(d, d_offset)
        loss = Box.box_inclusion_score(box_c, box_d, margin)
    else:
        r = init[:, -1]
        mask = th.isin(r, transitive_ids)
        box_c = E.embedding_2p(init, class_embed, class_offset, rel_embed, scale_embed, mask)
        box_d = Box(d, d_offset)

        r_embed = rel_embed(r[mask])
        r_trans = th.abs(r_embed)/th.norm(r_embed, dim=1).unsqueeze(1)

        if test:
            loss = -1 * th.ones((init.shape[0], tails.shape[0]), device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d, margin, r_trans)
            box_d.center = box_d.center.permute(1, 0, 2)
            box_d.offset = box_d.offset.permute(1, 0, 2)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d, margin)
        
        else:
            loss = -1 * th.ones(tails.shape[:2], device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d.slice(mask), margin, r_trans)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d.slice(~mask), margin)

        assert (loss == -1).sum() == 0, f"Loss tensor contains -1 values."
    
    return loss

@check_output_shape
def query_3p_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, transitive_ids, test):
    init, tails = data
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    
    if transitive_ids is None:
        box_c = E.embedding_3p(init, class_embed, class_offset, rel_embed, scale_embed, None)
        box_d = Box(d, d_offset)
        return Box.box_inclusion_score(box_c, box_d, margin)
    else:
        r = init[:, -1]
        mask = th.isin(r, transitive_ids)
        box_c = E.embedding_3p(init, class_embed, class_offset, rel_embed, scale_embed, mask)
        box_d = Box(d, d_offset)

        r_embed = rel_embed(r[mask])
        r_trans = th.abs(r_embed)/th.norm(r_embed, dim=1).unsqueeze(1)
        
        if test:
            loss = -1 * th.ones((init.shape[0], tails.shape[0]), device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d, margin, r_trans)
            box_d.center = box_d.center.permute(1, 0, 2)
            box_d.offset = box_d.offset.permute(1, 0, 2)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d, margin)
        else:
            loss = -1 * th.ones(tails.shape[:2], device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d.slice(mask), margin, r_trans)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d.slice(~mask), margin)

        assert (loss == -1).sum() == 0, f"Loss tensor contains -1 values."
    return loss
        
@check_output_shape
def query_2i_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_2i(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    inclusion_score = Box.box_inclusion_score(box_c, box_d, margin)
    return inclusion_score

@check_output_shape
def query_3i_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_3i(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    return Box.box_inclusion_score(box_c, box_d, margin)
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss

@check_output_shape
def query_2in_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    init, tails = data
    box_c = E.embedding_2in(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    return Box.box_inclusion_score(box_c, box_d, margin)
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss
    
@check_output_shape
def query_3in_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    init, tails = data
    box_c = E.embedding_3in(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    return Box.box_inclusion_score(box_c, box_d, margin)
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss
    

@check_output_shape
def query_ip_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, transitive_ids, test):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    init, tails = data
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    
    if transitive_ids is None:
        box_c = E.embedding_ip(init, class_embed, class_offset, rel_embed, scale_embed, None)
        box_d = Box(d, d_offset)
        return Box.box_inclusion_score(box_c, box_d, margin)
    else:
        r = init[:, -1]
        mask = th.isin(r, transitive_ids)
        box_c = E.embedding_ip(init, class_embed, class_offset, rel_embed, scale_embed, mask)
        box_d = Box(d, d_offset)

        r_embed = rel_embed(r[mask])
        r_trans = th.abs(r_embed)/th.norm(r_embed, dim=1).unsqueeze(1)
        
        if test:
            loss = -1 * th.ones((init.shape[0], tails.shape[0]), device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d, margin, r_trans)
            box_d.center = box_d.center.permute(1, 0, 2)
            box_d.offset = box_d.offset.permute(1, 0, 2)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d, margin)
        else:
            loss = -1 * th.ones(tails.shape[:2], device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d.slice(mask), margin, r_trans)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d.slice(~mask), margin)

        assert (loss == -1).sum() == 0, f"Loss tensor contains -1 values."
    
    return loss

    
@check_output_shape
def query_pi_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    init, tails = data
    box_c = E.embedding_pi(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    return Box.box_inclusion_score(box_c, box_d, margin)
    
@check_output_shape
def query_inp_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, transitive_ids, test):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    init, tails = data
    
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    

    if transitive_ids is None:
        box_c = E.embedding_inp(init, class_embed, class_offset, rel_embed, scale_embed, None)
        box_d = Box(d, d_offset)
        return Box.box_inclusion_score(box_c, box_d, margin)
    else:
        r = init[:, -1]
        mask = th.isin(r, transitive_ids)

        box_c = E.embedding_inp(init, class_embed, class_offset, rel_embed, scale_embed, mask)
        box_d = Box(d, d_offset)

        r_embed = rel_embed(r[mask])
        r_trans = th.abs(r_embed)/th.norm(r_embed, dim=1).unsqueeze(1)
        
        if test:
            loss = -1 * th.ones((init.shape[0], tails.shape[0]), device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d, margin, r_trans)
            box_d.center = box_d.center.permute(1, 0, 2)
            box_d.offset = box_d.offset.permute(1, 0, 2)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d, margin)
 
        else:
            loss = -1 * th.ones(tails.shape[:2], device=init.device)
            loss[mask] = Box.box_order_score(box_c.slice(mask), box_d.slice(mask), margin, r_trans)
            loss[~mask] = Box.box_inclusion_score(box_c.slice(~mask), box_d.slice(~mask), margin)

        assert (loss == -1).sum() == 0, f"Loss tensor contains -1 values."
    return loss
            
@check_output_shape
def query_pin_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    init, tails = data
    box_c = E.embedding_pin(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    return Box.box_inclusion_score(box_c, box_d, margin)


@check_output_shape
def query_pni_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    init, tails = data
    box_c = E.embedding_pni(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset)
    return Box.box_inclusion_score(box_c, box_d, margin)
