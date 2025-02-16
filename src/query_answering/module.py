import torch as th
import torch.nn as nn
import losses as L
import embeddings as E
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ELBEQAModule(nn.Module):
    """Implementation of ELBE from [peng2020]_.
    """
    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, gamma=10.0, transitive_ids=None, inverse_ids=None, ):
        super().__init__()
        self.task_names = ["1p", "2p", "3p", "2i", "3i", "2in", "3in", "pi", "ip", "inp", "pin", "pni"]

        if transitive_ids is not None:
            logger.info("Transitive relations are enabled.")
        self.transitive_ids = transitive_ids
        self.inverse_ids = inverse_ids
            
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        self.gamma = nn.Parameter(
            th.Tensor([gamma]), 
            requires_grad=False
        )

        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            th.Tensor([(self.gamma.item() + self.epsilon) / embed_dim]), 
            requires_grad=False
        )

        
        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        xavier = False
        if xavier:
            nn.init.xavier_uniform_(self.class_embed.weight)
        else:
            nn.init.uniform_(tensor=self.class_embed.weight,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())

            
        self.class_offset = nn.Embedding(self.nb_ont_classes, embed_dim)
        if xavier:
            nn.init.xavier_uniform_(self.class_offset.weight)
        else:
            nn.init.uniform_(tensor=self.class_offset.weight,
                             a=0.,
                             b=self.embedding_range.item())

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        if xavier:
            nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.rel_embed.weight.data, axis=1).reshape(-1, 1)
            self.rel_embed.weight.data /= weight_data_normalized
        else:
            nn.init.uniform_(tensor=self.rel_embed.weight,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())

            
        self.rel_factor = nn.Embedding(nb_rels, embed_dim)
        if xavier:
            nn.init.uniform_(self.rel_factor.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.rel_factor.weight.data, axis=1).reshape(-1, 1)
            self.rel_factor.weight.data /= weight_data_normalized
        else:
            nn.init.uniform_(tensor=self.rel_factor.weight,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            
        self.scale_embed = nn.Embedding(nb_rels, embed_dim)
        if xavier:
            nn.init.uniform_(self.scale_embed.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.scale_embed.weight.data, axis=1).reshape(-1, 1)
            self.scale_embed.weight.data /= weight_data_normalized
        else:
            nn.init.uniform_(tensor=self.scale_embed.weight,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            
        self.scale_bias = nn.Embedding(nb_rels, embed_dim)
        if xavier:
            nn.init.uniform_(self.scale_bias.weight, a=-1, b=1)
            weight_data_normalized = th.linalg.norm(self.scale_bias.weight.data, axis=1).reshape(-1, 1)
            self.scale_bias.weight.data /= weight_data_normalized
        else:
            nn.init.uniform_(tensor=self.scale_bias.weight,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
        
        self.margin = 0

    def query_1p(self, data, test=False):
        return L.query_1p_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_2p(self, data, test=False):
        return L.query_2p_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_3p(self, data, test=False):
        return L.query_3p_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_2i(self, data, test=False):
        return L.query_2i_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_3i(self, data, test=False):
        return L.query_3i_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_2in(self, data, test=False):
        return L.query_2in_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)
    
    def query_3in(self, data, test=False):
        return L.query_3in_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_pi(self, data, test=False):
        return L.query_pi_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_ip(self, data, test=False):
        return L.query_ip_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_inp(self, data, test=False):
        return L.query_inp_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_pin(self, data, test=False):
        return L.query_pin_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)

    def query_pni(self, data, test=False):
        return L.query_pni_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, self.margin, self.transitive_ids, self.inverse_ids, test)
    

    def embedding_1p(self, data, test=False):
        return E.embedding_1p(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_2p(self, data, test=False):
        return E.embedding_2p(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_3p(self, data, test=False):
        return E.embedding_3p(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_2i(self, data, test=False):
        return E.embedding_2i(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_3i(self, data, test=False):
        return E.embedding_3i(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_2in(self, data, test=False):
        return E.embedding_2in(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)
    
    def embedding_3in(self, data, test=False):
        return E.embedding_3in(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_pi(self, data, test=False):
        return E.embedding_pi(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_ip(self, data, test=False):
        return E.embedding_ip(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_inp(self, data, test=False):
        return E.embedding_inp(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_pin(self, data, test=False):
        return E.embedding_pin(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)

    def embedding_pni(self, data, test=False):
        return E.embedding_pni(data, self.class_embed, self.class_offset, self.rel_embed, self.rel_factor, self.scale_embed, self.scale_bias, None, None)
    



    
    def get_loss_function(self, task_name):
        """
        This chooses the corresponding loss fuction given the name of the task.
        """

        if task_name not in self.task_names:
            raise ValueError(
                f"Parameter task_name must be one of the following: {', '.join(self.task_names)}.")

        return {
            "1p": self.query_1p,
            "2p": self.query_2p,
            "3p": self.query_3p,
            "2i": self.query_2i,
            "3i": self.query_3i,
            "2in": self.query_2in,
            "3in": self.query_3in,
            "pi": self.query_pi,
            "ip": self.query_ip,
            "inp": self.query_inp,
            "pin": self.query_pin,
            "pni": self.query_pni
        }[task_name]


    def get_embedding(self, task_name):
        """
        This chooses the corresponding loss fuction given the name of the task.
        """

        if task_name not in self.task_names:
            raise ValueError(
                f"Parameter task_name must be one of the following: {', '.join(self.task_names)}.")

        return {
            "1p": self.embedding_1p,
            "2p": self.embedding_2p,
            "3p": self.embedding_3p,
            "2i": self.embedding_2i,
            "3i": self.embedding_3i,
            "2in": self.embedding_2in,
            "3in": self.embedding_3in,
            "pi": self.embedding_pi,
            "ip": self.embedding_ip,
            "inp": self.embedding_inp,
            "pin": self.embedding_pin,
            "pni": self.embedding_pni
        }[task_name]

    def forward(self, init, tails, task_name, transitive_ids= None, test=False):
        loss_fn = self.get_loss_function(task_name)

        loss = loss_fn((init, tails), test=test)
        if test:
            return loss
        else:
            return self.gamma - loss


    def get_query_embedding(self, init, task_name):
        embedding_fn = self.get_embedding(task_name)
        return embedding_fn(init)
