import losses_abox_weak as L
from elmodule import ELModule
import torch as th
import torch.nn as nn
from deprecated.sphinx import deprecated


class TransitiveELModule(ELModule):
    """Implementation of Transitive Box Embeddings from []_.
    """
    def __init__(self, nb_ont_classes, nb_rels, nb_individuals, transitive, transitive_ids=None, embed_dim=50, margin=0.1):
        super().__init__()


        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.transitive_ids = transitive_ids
        self.embed_dim = embed_dim
        self.transitive = transitive
        
        self.class_embed = nn.Embedding(nb_ont_classes, embed_dim)
        nn.init.xavier_uniform_(self.class_embed.weight)
        
        self.individual_embed = nn.Embedding(nb_individuals, embed_dim)
        nn.init.xavier_uniform_(self.individual_embed.weight)
                        
        self.rel_embed = self.init_embeddings(nb_rels, embed_dim)
        self.margin = margin
        

    def init_embeddings(self, n_embeddings, embed_dim, a = -1, b = 1):
        embeddings = nn.Embedding(n_embeddings, embed_dim)
        nn.init.xavier_uniform_(embeddings.weight)
        weight_data_normalized = th.linalg.norm(embeddings.weight.data, axis=1).reshape(-1, 1)
        embeddings.weight.data /= weight_data_normalized
        return embeddings

    def class_assertion_loss(self, data, neg=False):
        return L.class_assertion_loss(data, self.class_embed,
                                      self.individual_embed,
                                      self.margin, neg=neg)
    
    def object_property_assertion_loss(self, data, neg=False):
        return L.object_property_assertion_loss(data,
                                                self.individual_embed,
                                                self.rel_embed,
                                                self.transitive_ids,
                                                self.margin,
                                                self.transitive,
                                                neg=neg)

    
                                                
    def gci2_loss(self, data, neg=False, train=False):
        return L.gci2_loss(data, self.class_embed, self.rel_embed,
                           self.transitive_ids, self.margin,
                           self.transitive, neg=neg, train=train)
         
                            
