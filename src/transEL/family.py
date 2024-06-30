import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.utils.random import seed_everything
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from mowl.datasets import PathDataset
from tqdm import tqdm
from module import TransitiveELModule
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from itertools import cycle
import logging
import math
import click as ck
import os
import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

th.autograd.set_detect_anomaly(True)

@ck.command()
@ck.option("--embed_dim", "-dim", default=2, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=10, help="Batch size")
@ck.option("--module_margin", "-mm", default=0.0, help="Margin for the module")
@ck.option("--loss_margin", "-lm", default=0.1, help="Margin for the loss function")
@ck.option("--learning_rate", "-lr", default=0.01, help="Learning rate")
@ck.option("--epochs", "-ep", default=10000, help="Number of epochs")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="family")
def main(embed_dim, batch_size, module_margin, loss_margin,
         learning_rate, epochs, device, wandb_description):

    seed_everything(0)
    
    wandb_logger = wandb.init(entity="ferzcam", project="onto-r", group="toy_family", name=wandb_description)

    if loss_margin == int(loss_margin):
        loss_margin = int(loss_margin)
    if module_margin == int(module_margin):
        module_margin = int(module_margin)
    
    wandb_logger.log({"embed_dim": embed_dim,
                      "batch_size": batch_size,
                      "module_margin": module_margin,
                      "loss_margin": loss_margin,
                      "learning_rate": learning_rate
                      })

    root_dir = "../../use_cases/family"
    dataset = PathDataset(os.path.join(root_dir, "family2.owl"))
    
    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    model_filepath = f"{model_dir}/family.pt"
    model = GeometricELModel(dataset, batch_size, embed_dim,
                             module_margin, loss_margin,
                             learning_rate, model_filepath, epochs,
                             device, wandb_logger)

    model.train()
        

class GeometricELModel(EmbeddingELModel):
    def __init__(self, dataset, batch_size, embed_dim, module_margin,
                 loss_margin, learning_rate, model_filepath, epochs,
                 device, wandb_logger):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath)

        self.rbox_data = self.process_rbox_axioms()
        transitive_ids = self.rbox_data["transitiveproperty"].to(device)

        print(self.dataset.classes.as_str)
        print(self.dataset.object_properties.as_str)
        print(self.dataset.individuals.as_str)
        self.module = TransitiveELModule(len(self.dataset.classes),
                                         len(self.dataset.object_properties),
                                         len(self.dataset.individuals),
                                         embed_dim = self.embed_dim,
                                         margin= module_margin,
                                         transitive_ids=transitive_ids)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_margin = loss_margin
        self.device = device
        self.wandb_logger = wandb_logger

    def process_rbox_axioms(self):
        rbox_axioms = self.dataset.ontology.getRBoxAxioms(Imports.fromBoolean(True))
        rbox_data = {"subobjectproperty": [], "inverseproperty": [], "transitiveproperty": []}

        owl2idx = self.dataset.object_properties.to_index_dict()
        
        for axiom in rbox_axioms:
            print(str(axiom.toString()))
            axiom_type = axiom.getAxiomType()
            if axiom_type == Ax.SUB_OBJECT_PROPERTY:
                
                sub = axiom.getSubProperty()
                sup = axiom.getSuperProperty()
                sub_id = owl2idx[sub]
                sup_id = owl2idx[sup]
                rbox_data["subobjectproperty"].append((sub_id, sup_id))
            elif axiom_type == Ax.INVERSE_OBJECT_PROPERTIES:
                first = axiom.getFirstProperty()
                second = axiom.getSecondProperty()

                first_id = owl2idx[first]
                second_id = owl2idx[second]
                rbox_data["inverseproperty"].append((first_id, second_id))
            elif axiom_type == Ax.TRANSITIVE_OBJECT_PROPERTY:
                property_ = axiom.getProperty()
                property_id = owl2idx[property_]
                rbox_data["transitiveproperty"].append(property_id)
                
        for k, v in rbox_data.items():
            print(f"{k}: {v}")
                
        rbox_data = {k: th.tensor(v, dtype=th.long) for k, v in rbox_data.items()}


        
        return rbox_data
                


    def tbox_forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

 
    def train(self):

        dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True)
               for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        total_dls_size = sum(dls_sizes.values())
        dls_weights = {gci_name: ds_size/total_dls_size for gci_name, ds_size in dls_sizes.items()}
        # dls_weights = {gci_name: 1 for gci_name in dls_sizes.keys()}
        main_dl = dls["gci0"]
        logger.info(f"Training with {len(main_dl)} batches of size {self.batch_size}")
        dls = {gci_name: cycle(dl) for gci_name, dl in dls.items() if gci_name != "gci0"}
        logger.info(f"Dataloaders: {dls_sizes}")

        tolerance = 15
        curr_tolerance = tolerance

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        # optimizer = th.optim.SGD(self.module.parameters(), lr=self.learning_rate)

        best_loss = float("inf")
        
        num_classes = len(self.dataset.classes)

        self.module = self.module.to(self.device)
        for epoch in tqdm(range(self.epochs)):
            self.module.train()
            total_train_loss = 0

            gci_losses = {gci_name: 0 for gci_name in dls.keys()}
            gci_losses["reg"] = 0
            gci_losses["gci0"] = 0

            for batch_data in main_dl:

                batch_data = batch_data.to(self.device)
                pos_logits = self.tbox_forward(batch_data, "gci0")
                # neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                # neg_batch = th.cat([batch_data[:, :1], neg_idxs.unsqueeze(1)], dim=1)
                # neg_logits = self.tbox_forward(neg_batch, "gci0")
                # loss = - F.logsigmoid(-pos_logits + neg_logits - self.loss_margin).mean() * dls_weights["gci0"]
                loss = pos_logits.mean() * dls_weights["gci0"]
                
                gci_losses["gci0"] += round(loss.item(), 3)
                
                for gci_name, gci_dl in dls.items():
                    if gci_name == "gci0":
                        continue

                    # if gci_name != "object_property_assertion":
                        # continue
                    
                    batch_data = next(gci_dl).to(self.device)
                    pos_logits = self.tbox_forward(batch_data, gci_name)
                    # neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                    # neg_batch = th.cat([batch_data[:, :2], neg_idxs.unsqueeze(1)], dim=1)
                        
                    # neg_logits = self.tbox_forward(neg_batch, gci_name)
                    # loss += - F.logsigmoid(-pos_logits + neg_logits - self.loss_margin).mean() * dls_weights[gci_name]
                    gci_loss = pos_logits.mean() * dls_weights[gci_name]
                    loss += gci_loss
                    gci_losses[gci_name] += round(gci_loss.item(), 3)

                reg_loss = self.module.regularization_loss()
                loss += reg_loss
                gci_losses["reg"] += round(reg_loss.item(), 3)
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()

                
            if best_loss > total_train_loss:
                best_loss = total_train_loss
                curr_tolerance = tolerance
                
            else:
                curr_tolerance -= 1

            th.save(self.module.state_dict(), self.model_filepath.replace(".pt", f"_ep_{epoch}.pt"))
                
            if curr_tolerance == 0:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            to_print = ""
            for k, v in gci_losses.items():
                to_print += f"{k}: {v} "

            
            logger.info(f"Epoch {epoch} - Total loss: {total_train_loss:3f} - {to_print}")

                                    
class DummyLogger():
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
    
if __name__ == "__main__":
    main()
