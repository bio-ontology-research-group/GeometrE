import torch as th
import torch.nn as nn

from mowl.nn import ELEmModule, BoxSquaredELModule, ELBoxModule

class FullELEmModule(ELEmModule):

    def __init__(self, *args, transitive_ids = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.transitive_ids = transitive_ids

        self.rel_projection = th.nn.Embedding(self.nb_rels, self.embed_dim*self.embed_dim )
        nn.init.xavier_uniform_(self.rel_projection.weight)

        self.rel_projection_2 = th.nn.Embedding(self.nb_rels, self.embed_dim*self.embed_dim )
        nn.init.xavier_uniform_(self.rel_projection_2.weight)
        
        self.proj_rad = nn.Embedding(self.nb_rels, self.nb_ont_classes)
        self.proj_rad_2 = nn.Embedding(self.nb_rels, self.nb_ont_classes)
        nn.init.uniform_(self.proj_rad.weight, a=-1, b=1)

        self.existential = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.embed_dim, self.embed_dim),
                                         )
        
    def gci1_bot_loss(self, data, neg=False):
        return gci1_bot_loss(data, self.class_embed, self.class_rad, self.margin,
                               neg=neg)
    

    
    def trans_gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return transitive_gci2_loss(data, self.transitive_ids, self.class_embed, self.class_rad, self.rel_embed, self.rel_projection, self.proj_rad, self.existential, self.margin, neg=neg)


    def trans_gci3_loss(self, data, neg=False, idxs_for_negs=None):
        return transitive_gci3_loss(data, self.transitive_ids, self.class_embed, self.class_rad, self.rel_embed, self.rel_projection_2, self.proj_rad_2, self.existential, self.margin, neg=neg)


    def regularization_loss(self):
        return regularization_loss(self.class_embed, self.reg_norm)

def regularization_loss(class_embed, reg_factor):
    res = th.relu(th.linalg.norm(class_embed.weight, axis=1) - reg_factor).mean()
    # res = th.reshape(res, [-1, 1])
    return res

    
def gci1_bot_loss(data, class_embed, class_rad, margin, neg=False):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd
    dst = th.reshape(th.linalg.norm(d - c, axis=1), [-1, 1])
    return th.relu(sr - dst + margin) 
    
def transitive_gci2_loss(data, transitive_ids, class_embed, class_rad, rel_embed, rel_proj, proj_rad, existential, margin, neg=False):

    transitive_ids = transitive_ids.to(data.device)
    
    rel_ids = data[:, 1]
    # filter rels that are in transitive ids
    mask = th.isin(rel_ids, transitive_ids)

    transitive_data = data[mask]
    non_transitive_data = data[~mask]

    c_idxs = transitive_data[:, 0]
    r_idxs = transitive_data[:, 1]
    d_idxs = transitive_data[:, 2]

    c = class_embed(c_idxs)
    r = rel_proj(r_idxs).view(-1, c.shape[1], c.shape[1])
    d = class_embed(d_idxs)

    # ex_d = existential(d)
    
    c_proj = th.bmm(c.unsqueeze(1), r).squeeze(1)
    d_proj = th.bmm(d.unsqueeze(1), r).squeeze(1)

    c_proj_rad = th.abs(proj_rad.weight[r_idxs, c_idxs])
    d_proj_rad = th.abs(proj_rad.weight[r_idxs, d_idxs])

    dist = th.linalg.norm(c_proj - d_proj, dim=1, keepdim=True) + c_proj_rad - d_proj_rad
    transitive_logits = th.relu(dist- margin)

    non_transitive_logits = gci2_score(non_transitive_data, class_embed, class_rad, rel_embed, margin)

    trans_reg_loss = th.abs(th.linalg.norm(c_proj, axis=1) - 1).mean()
    trans_reg_loss += th.abs(th.linalg.norm(d_proj, axis=1) - 1).mean()
    
    return transitive_logits, non_transitive_logits, trans_reg_loss

def transitive_gci3_loss(data, transitive_ids, class_embed, class_rad, rel_embed, rel_proj, proj_rad, existential, margin, neg=False):

    transitive_ids = transitive_ids.to(data.device)
    
    rel_ids = data[:, 0]
    # filter rels that are in transitive ids
    mask = th.isin(rel_ids, transitive_ids)
    # mask = th.isin(rel_ids, rel_ids)

    transitive_data = data[mask]
    non_transitive_data = data[~mask]

    r_idxs = transitive_data[:, 0]
    c_idxs = transitive_data[:, 1]
    d_idxs = transitive_data[:, 2]

    c = class_embed(c_idxs)
    r = rel_proj(r_idxs).view(-1, c.shape[1], c.shape[1])
    d = class_embed(d_idxs)

    # ex_c = existential(c)
    
    c_proj = th.bmm(c.unsqueeze(1), r).squeeze(1)
    d_proj = th.bmm(d.unsqueeze(1), r).squeeze(1)

    c_proj_rad = th.abs(proj_rad.weight[r_idxs, c_idxs])
    d_proj_rad = th.abs(proj_rad.weight[r_idxs, d_idxs])

    dist = th.linalg.norm(c_proj - d_proj, dim=1, keepdim=True) + c_proj_rad - d_proj_rad
    transitive_logits = th.relu(dist- margin)

    non_transitive_logits = gci3_score(non_transitive_data, class_embed, class_rad, rel_embed, margin)

    trans_reg_loss = th.abs(th.linalg.norm(c_proj, axis=1) - 1).mean()
    trans_reg_loss += th.abs(th.linalg.norm(d_proj, axis=1) - 1).mean()
    
    return transitive_logits, non_transitive_logits, trans_reg_loss


def gci2_score(data, class_embed, class_rad, rel_embed, margin):
    # C subClassOf R some D
    c = class_embed(data[:, 0])
    rE = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 2]))
    # c should intersect with d + r

    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    score = th.relu(dst + rc - rd - margin) + 10e-6
    return score



def gci3_score(data, class_embed, class_rad, rel_embed, margin):
    # R some C subClassOf D
    rE = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])

    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))
    # c should intersect with d + r

    dst = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
    score = th.relu(dst + rc - rd - margin) + 10e-6
    return score












class FullELBEModule(ELBoxModule):

    def __init__(self, *args, transitive_ids = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.transitive_ids = transitive_ids

        self.rel_projection = th.nn.Embedding(self.nb_rels, self.embed_dim*self.embed_dim )
        nn.init.xavier_uniform_(self.rel_projection.weight)

        self.rel_projection_2 = th.nn.Embedding(self.nb_rels, self.embed_dim*self.embed_dim )
        nn.init.xavier_uniform_(self.rel_projection_2.weight)
        
        self.proj_rad = nn.Embedding(self.nb_rels, self.nb_ont_classes)
        self.proj_rad_2 = nn.Embedding(self.nb_rels, self.nb_ont_classes)
        nn.init.uniform_(self.proj_rad.weight, a=-1, b=1)

        self.existential = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.embed_dim, self.embed_dim),
                                         )
        
    # def gci1_bot_loss(self, data, neg=False):
        # return gci1_bot_loss(data, self.class_embed, self.class_rad, self.margin,
                               # neg=neg)
    

    
    def trans_gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return transitive_gci2_loss_elbe(data, self.transitive_ids, self.class_embed, self.class_offset, self.rel_embed, self.rel_projection, self.proj_rad, self.existential, self.margin, neg=neg)


    def trans_gci3_loss(self, data, neg=False, idxs_for_negs=None):
        return transitive_gci3_loss_elbe(data, self.transitive_ids, self.class_embed, self.class_offset, self.rel_embed, self.rel_projection_2, self.proj_rad_2, self.existential, self.margin, neg=neg)


def transitive_gci2_loss_elbe(data, transitive_ids, class_embed, class_offset, rel_embed, rel_proj, proj_rad, existential, margin, neg=False):

    transitive_ids = transitive_ids.to(data.device)
    
    rel_ids = data[:, 1]
    # filter rels that are in transitive ids
    mask = th.isin(rel_ids, transitive_ids)

    transitive_data = data[mask]
    non_transitive_data = data[~mask]

    c_idxs = transitive_data[:, 0]
    r_idxs = transitive_data[:, 1]
    d_idxs = transitive_data[:, 2]

    c = class_embed(c_idxs)
    r = rel_proj(r_idxs).view(-1, c.shape[1], c.shape[1])
    d = class_embed(d_idxs)

    # ex_d = existential(d)
    
    c_proj = th.bmm(c.unsqueeze(1), r).squeeze(1)
    d_proj = th.bmm(d.unsqueeze(1), r).squeeze(1)

    c_proj_rad = th.abs(proj_rad.weight[r_idxs, c_idxs])
    d_proj_rad = th.abs(proj_rad.weight[r_idxs, d_idxs])

    dist = th.linalg.norm(c_proj - d_proj, dim=1, keepdim=True) + c_proj_rad - d_proj_rad
    transitive_logits = th.relu(dist- margin)

    non_transitive_logits = gci2_score_elbe(non_transitive_data, class_embed, class_offset, rel_embed, margin)

    trans_reg_loss = th.abs(th.linalg.norm(c_proj, axis=1) - 1).mean()
    trans_reg_loss += th.abs(th.linalg.norm(d_proj, axis=1) - 1).mean()
    
    return transitive_logits, non_transitive_logits, trans_reg_loss

def transitive_gci3_loss_elbe(data, transitive_ids, class_embed, class_offset, rel_embed, rel_proj, proj_rad, existential, margin, neg=False):

    transitive_ids = transitive_ids.to(data.device)
    
    rel_ids = data[:, 0]
    # filter rels that are in transitive ids
    mask = th.isin(rel_ids, transitive_ids)
    # mask = th.isin(rel_ids, rel_ids)

    transitive_data = data[mask]
    non_transitive_data = data[~mask]

    r_idxs = transitive_data[:, 0]
    c_idxs = transitive_data[:, 1]
    d_idxs = transitive_data[:, 2]

    c = class_embed(c_idxs)
    r = rel_proj(r_idxs).view(-1, c.shape[1], c.shape[1])
    d = class_embed(d_idxs)

    # ex_c = existential(c)
    
    c_proj = th.bmm(c.unsqueeze(1), r).squeeze(1)
    d_proj = th.bmm(d.unsqueeze(1), r).squeeze(1)

    c_proj_rad = th.abs(proj_rad.weight[r_idxs, c_idxs])
    d_proj_rad = th.abs(proj_rad.weight[r_idxs, d_idxs])

    dist = th.linalg.norm(c_proj - d_proj, dim=1, keepdim=True) + c_proj_rad - d_proj_rad
    transitive_logits = th.relu(dist- margin)

    non_transitive_logits = gci3_score_elbe(non_transitive_data, class_embed, class_offset, rel_embed, margin)

    trans_reg_loss = th.abs(th.linalg.norm(c_proj, axis=1) - 1).mean()
    trans_reg_loss += th.abs(th.linalg.norm(d_proj, axis=1) - 1).mean()
    
    return transitive_logits, non_transitive_logits, trans_reg_loss


def gci2_score_elbe(data, class_embed, class_offset, rel_embed, margin, neg=False):
    c = class_embed(data[:, 0])
    r = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 2]))
    
    euc = th.abs(c + r - d)
    dst = th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + margin), axis=1), [-1, 1])
    return dst


def gci3_score_elbe(data, class_embed, class_offset, rel_embed, margin, neg=False):
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))

    euc = th.abs(c - r - d)
    dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d + margin), axis=1), [-1, 1])
    return dst
