import losses_abox_weak as L
from elmodule import ELModule
import torch as th
import torch.nn as nn
from deprecated.sphinx import deprecated


class TransitiveELModule(ELModule):
    """Implementation of Transitive Box Embeddings from []_.
    """
    def __init__(self, nb_ont_classes, nb_rels, nb_individuals, transitive, transitive_ids=None, embed_dim=50, margin=0.1, max_bound=10):
        super().__init__()


        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.transitive_ids = transitive_ids
        self.embed_dim = embed_dim
        self.transitive = transitive
        self.max_bound = max_bound
        
        # self.class_lower = self.init_embeddings(nb_ont_classes, embed_dim, a = 0, b = max_bound)
        
        # self.class_lower = self.init_embeddings(nb_ont_classes, embed_dim)
        # self.class_lower = nn.Embedding(nb_ont_classes, embed_dim)
        # nn.init.xavier_uniform_(self.class_lower.weight)

        self.class_lower = nn.Embedding(nb_ont_classes, embed_dim)
        nn.init.xavier_uniform_(self.class_lower.weight)
        # nn.init.uniform_(self.class_lower.weight, a=0, b=max_bound)
        
        
        self.class_delta = self.init_embeddings(nb_ont_classes, embed_dim)
        self.individual_embed = nn.Embedding(nb_individuals, embed_dim)
        nn.init.uniform_(self.individual_embed.weight, a=0, b=max_bound)
        # self.individual_embed = self.init_embeddings(nb_individuals, embed_dim)

        # self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        # nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        self.rel_embed = self.init_embeddings(nb_rels, embed_dim)
        
        self.rel_mask = nn.Embedding(nb_rels, embed_dim)
        self.rel_mask.weight.data.fill_(0)
        # self.rel_mask.weight.data[self.transitive_ids] = 0

        column_index = th.arange(len(transitive_ids))
        self.rel_mask.weight.data[self.transitive_ids, column_index] = 1
        assert th.sum(self.rel_mask.weight.data) == len(self.transitive_ids)
        self.rel_mask.weight.requires_grad = False
        self.margin = margin
        

    def init_embeddings(self, n_embeddings, embed_dim, a = -1, b = 1):
        embeddings = nn.Embedding(n_embeddings, embed_dim)
        nn.init.xavier_uniform_(embeddings.weight)
        weight_data_normalized = th.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        embeddings.weight.data /= weight_data_normalized
        return embeddings

    def class_assertion_loss(self, data, neg=False):
        return L.class_assertion_loss(data, self.class_lower,
                                      self.class_delta, self.individual_embed, self.margin, neg=neg)
    
    def object_property_assertion_loss(self, data, neg=False):
        return L.object_property_assertion_loss(data,
                                                self.individual_embed,
                                                self.rel_embed,
                                                self.rel_mask,
                                                self.transitive_ids,
                                                self.margin,
                                                self.transitive,
                                                neg=neg)

    
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_lower,
                           self.class_delta, self.margin, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_lower,
                               self.class_delta, self.margin, self.max_bound, neg=neg)
    
    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_lower,
                           self.class_delta, self.margin, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_lower,
                               self.class_delta,  self.margin, neg=neg)

    def gci2_loss(self, data, neg=False, train=False):
        return L.gci2_loss(data, self.class_lower, self.class_delta,
                           self.rel_embed, self.rel_mask,
                           self.transitive_ids, self.margin,
                           self.transitive, neg=neg, train=train)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_lower, self.class_delta,
                           self.rel_embed, self.rel_mask,
                           self.transitive_ids, self.margin,
                           self.transitive, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.class_delta, self.margin, neg=neg)


    def regularization_loss(self, reg_factor=0.1):
        inds_reg =  L.regularization_loss(self.class_lower, self.max_bound, reg_factor = reg_factor)
        rel_reg = L.rel_regularization_loss(self.rel_embed, self.rel_mask, self.transitive_ids, self.max_bound, reg_factor = reg_factor)
        return inds_reg, rel_reg
