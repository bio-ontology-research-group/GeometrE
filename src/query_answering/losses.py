import torch as th
from box import Box
import embeddings as E
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


@check_output_shape
def query_1p_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_1p(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
                                   
    return Box.box_inclusion_score(box_c, box_d, margin)


@check_output_shape
def query_2p_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_2p(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    
    return Box.box_inclusion_score(box_c, box_d, margin)
    

@check_output_shape
def query_3p_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_3p(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)

    return Box.box_inclusion_score(box_c, box_d, margin)

@check_output_shape
def query_2i_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_2i(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    inclusion_score = Box.box_inclusion_score(box_c, box_d, margin)
    return inclusion_score # + corner_loss

@check_output_shape
def query_3i_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    init, tails = data
    box_c = E.embedding_3i(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss

@check_output_shape
def query_2in_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in',
    init, tails = data
    box_c = E.embedding_2in(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss
    
@check_output_shape
def query_3in_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))
    init, tails = data
    box_c = E.embedding_3in(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss
    

@check_output_shape
def query_ip_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
    init, tails = data
    box_c = E.embedding_ip(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss

@check_output_shape
def query_pi_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    init, tails = data
    box_c = E.embedding_pi(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss
    
@check_output_shape
def query_inp_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',))
    init, tails = data
    box_c = E.embedding_inp(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss
    # return Box.box_equiv_score(intersection, box_d, margin) + corner_loss

@check_output_shape
def query_pin_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r', 'r')), ('e', ('r', 'n')))
    init, tails = data
    box_c = E.embedding_pin(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss


@check_output_shape
def query_pni_loss(data, class_embed, class_offset, rel_embed, scale_embed, margin, test):
    # (('e', ('r', 'r', 'n')), ('e', ('r',)))
    init, tails = data
    box_c = E.embedding_pni(init, class_embed, class_offset, rel_embed, scale_embed)
    d = class_embed(tails)
    d_offset = th.abs(class_offset(tails))
    box_d = Box(d, d_offset, normalize=True)
    return Box.box_inclusion_score(box_c, box_d, margin) #+ corner_loss
