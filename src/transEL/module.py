import losses as L
from mowl.nn import ELModule
import torch as th
import torch.nn as nn
from deprecated.sphinx import deprecated


class TransitiveELModule(ELModule):
    """Implementation of Transitive Box Embeddings from []_.
    """
    def __init__(self, nb_ont_classes, nb_rels, nb_individuals, transitive_ids=None, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.transitive_ids = transitive_ids
        self.embed_dim = embed_dim

        self.class_embed = self.init_embeddings(nb_ont_classes, embed_dim)
        self.class_offset = self.init_embeddings(nb_ont_classes, embed_dim)
        self.individual_embed = self.init_embeddings(nb_individuals, embed_dim)
        self.rel_embed = self.init_embeddings(nb_rels, embed_dim)
                                        
        self.rel_mask = nn.Embedding(nb_rels, embed_dim)
        # self.rel_mask.weight.data = th.ones_like(self.rel_mask.weight.data)
        # self.rel_mask.weight.data[:, -nb_rels:] = th.eye(nb_rels)
        # self.rel_mask.weight.requires_grad = False

        
        self.margin = margin


    def init_embeddings(self, n_embeddings, embed_dim):
        embeddings = nn.Embedding(n_embeddings, embed_dim)
        nn.init.uniform_(embeddings.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        embeddings.weight.data /= weight_data_normalized
        return embeddings


    def class_assertion_loss(self, data, neg=False):
        return L.class_assertion_loss(data, self.class_embed,
                                      self.class_offset, self.individual_embed, self.margin, neg=neg)
    
    def object_property_assertion_loss(self, data, neg=False):
        return L.object_property_assertion_loss(data, self.individual_embed,
                                                self.rel_embed, self.rel_mask,
                                                self.transitive_ids, self.margin, neg=neg)

    
    def gci0_loss(self, data, neg=False):
        return L.gci0_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci0_bot_loss(self, data, neg=False):
        return L.gci0_bot_loss(data, self.class_offset, self.margin, neg=neg)
    
    def gci1_loss(self, data, neg=False):
        return L.gci1_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci1_bot_loss(self, data, neg=False):
        return L.gci1_bot_loss(data, self.class_embed, self.class_offset, self.margin, neg=neg)

    def gci2_loss(self, data, neg=False):
        return L.gci2_loss(data, self.class_embed, self.class_offset,
                           self.rel_embed, self.rel_mask, self.transitive_ids,
                           self.margin, neg=neg)

    def gci3_loss(self, data, neg=False):
        return L.gci3_loss(data, self.class_embed, self.class_offset,
                           self.rel_embed, self.rel_mask, self.transitive_ids,
                           self.margin, neg=neg)

    def gci3_bot_loss(self, data, neg=False):
        return L.gci3_bot_loss(data, self.class_offset, self.margin, neg=neg)


    def regularization_loss(self):
        return L.regularization_loss(self.rel_embed, self.rel_mask, self.transitive_ids)
