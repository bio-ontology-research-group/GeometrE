import losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn
from deprecated.sphinx import deprecated


class TransitiveELModule(ELModule):
    """Implementation of Transitive Box Embeddings from []_.
    """
    def __init__(self, nb_ont_classes, nb_rels, nb_individuals, transitive, transitive_ids=None, embed_dim=50, margin=0.1, min_bound=5):
        super().__init__()


        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.transitive_ids = transitive_ids
        self.embed_dim = embed_dim
        self.transitive = transitive

        self.class_center = self.init_embeddings(nb_ont_classes, embed_dim, a = -min_bound, b=1)
        self.class_offset = self.init_embeddings(nb_ont_classes, embed_dim)
        self.individual_embed = self.init_embeddings(nb_individuals, embed_dim)

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)

        self.rel_mask = nn.Embedding(nb_rels, embed_dim)
        self.rel_mask.weight.data.fill_(0)
        self.rel_mask.weight.data[:nb_rels, :nb_rels] = th.eye(nb_rels)
        self.rel_mask.weight.requires_grad = False

        self.min_bound = min_bound
        
        self.margin = margin


    def init_embeddings(self, n_embeddings, embed_dim, a = -1, b = 1):
        embeddings = nn.Embedding(n_embeddings, embed_dim)
        nn.init.uniform_(embeddings.weight, a=a, b=b)
        weight_data_normalized = th.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        embeddings.weight.data /= weight_data_normalized
        return embeddings


    # def fix_classes(self, ids, dims):
        
        # if ids is not None:
            # centers = self.class_center(ids)
            # offsets = th.abs(self.class_offset(ids))

            # lower = centers - offsets
            # upper = centers + offsets

            # lower[dims] = -self.min_bound

        
            # new_centers = (upper + lower) / 2
            # new_offsets = (upper - lower) / 2
        
            # self.class_center.weight.data[ids] = new_centers
            # self.class_offset.weight.data[ids] = new_offsets


        # all_centers = self.class_center.weight.data
        # all_offsets = th.abs(self.class_offset.weight.data)

        # all_lower = all_centers - all_offsets
        # all_upper = all_centers + all_offsets
        # all_lower = th.max(all_lower, th.full_like(all_lower, -self.min_bound))

        # all_offsets = (all_upper - all_lower) / 2
        # all_centers = (all_upper + all_lower) / 2

        # self.class_center.weight.data = all_centers
        # self.class_offset.weight.data = all_offsets
        
        
    
    def class_assertion_loss(self, data, neg=False):
        return L.class_assertion_loss(data, self.class_center,
                                      self.class_offset, self.individual_embed, self.margin, neg=neg)
    
    def object_property_assertion_loss(self, data, neg=False):
        return L.object_property_assertion_loss(data,
                                                self.individual_embed,
                                                self.rel_embed,
                                                self.rel_mask,
                                                self.min_bound,
                                                self.transitive_ids,
                                                self.margin,
                                                self.transitive,
                                                neg=neg)

    
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_center,
                           self.class_offset, self.margin, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_center,
                               self.class_offset, self.margin, neg=neg)
    
    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_center,
                           self.class_offset, self.margin, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_center,
                               self.class_offset,  self.margin, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_center,
                           self.class_offset,
                           self.rel_embed, self.rel_mask,
                           self.min_bound, self.transitive_ids,
                           self.margin, self.transitive, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_center,
                           self.class_offset,
                           self.rel_embed, self.rel_mask,
                           self.min_bound, self.transitive_ids,
                           self.margin, self.transitive, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.class_offset, self.margin, neg=neg)


    # def regularization_loss(self, reg_factor=0.1):
        # return L.regularization_loss(self.rel_embed, self.rel_mask, self.transitive_ids)
        # return L.regularization_loss(self.class_center, self.class_offset, self.individual_embed, self.min_bound, reg_factor = reg_factor)
