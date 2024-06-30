import torch as th
import torch.nn as nn

from mowl.nn import ELEmModule, BoxSquaredELModule

class FullELEmModule(ELEmModule):

    def __init__(self, *args, transitive_ids = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.transitive_ids = transitive_ids
        
        self.rel_projection = th.nn.Embedding(self.nb_rels, self.embed_dim*self.embed_dim )
        nn.init.xavier_uniform_(self.rel_projection.weight)

        self.proj_rad = nn.Embedding(self.nb_rels, self.nb_ont_classes)
        nn.init.uniform_(self.proj_rad.weight, a=-1, b=1)


    def trans_gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return transitive_gci2_loss(data, self.transitive_ids, self.class_embed, self.class_rad, self.rel_embed, self.rel_projection, self.proj_rad, self.margin, neg=neg)


                    

    
def transitive_gci2_loss(data, transitive_ids, class_embed, class_rad, rel_embed, rel_proj, proj_rad, margin, neg=False):

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


