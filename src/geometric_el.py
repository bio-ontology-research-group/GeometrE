import mowl
mowl.init_jvm("10g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.datasets import PathDataset
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import RelationEvaluator
from tqdm import tqdm
from mowl.nn import ELEmModule, ELBoxModule, BoxSquaredELModule
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import torch.nn.functional as F
from itertools import cycle
import logging
import math
import click as ck
import os
import wandb
logger = logging.getLogger(__name__)
# handler = logging.StreamHandler()
# logger.addHandler(handler)
logger.setLevel(logging.INFO)

th.autograd.set_detect_anomaly(True)

@ck.command()
@ck.option("--dataset_name", "-ds", type=ck.Choice(["goslim", "go", "goplus"]), default="goslim")
@ck.option("--module_name", "-m", default="elem", help="Module to use")
@ck.option("--evaluator_name", "-e", default="relation", help="Evaluator to use")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-b", default=512, help="Batch size")
@ck.option("--module_margin", "-mm", default=0.01, help="Margin for the module")
@ck.option("--loss_margin", "-lm", default=0.01, help="Margin for the loss function")
@ck.option("--learning_rate", "-lr", default=0.0001, help="Learning rate")
@ck.option("--epochs", "-e", default=4000, help="Number of epochs")
@ck.option("--rbox_loss", "-rbox", is_flag=True)
@ck.option("--evaluate_every", "-every", default=10, help="Evaluate every n epochs")
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_wandb", "-nw", is_flag=True)
def main(dataset_name, module_name, evaluator_name, embed_dim, batch_size,
         module_margin, loss_margin, learning_rate, epochs, rbox_loss, evaluate_every,
         device, wandb_description, no_wandb):

    if no_wandb:
        wandb_logger = DummyLogger()
    else:
        wandb_logger = wandb.init(project="onto-r", group="f{dataset_name}_{module_name}_{evaluator_name}", name=wandb_description)

    
    wandb_logger.log({"dataset_name": dataset_name,
                      "module_name": module_name,
                      "evaluator_name": evaluator_name,
                      "embed_dim": embed_dim,
                      "batch_size": batch_size,
                      "module_margin": module_margin,
                      "loss_margin": loss_margin,
                      "learning_rate": learning_rate,
                      "rbox_loss": rbox_loss,
                      })

    
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/models/"
    os.makedirs(model_dir, exist_ok=True)

    model_filepath = f"{model_dir}/{module_name}_{embed_dim}_{batch_size}_{module_margin}_{loss_margin}_{learning_rate}.pt"
    model = GeometricELModel(module_name, evaluator_name, dataset, batch_size,
                             embed_dim, module_margin, loss_margin,
                             learning_rate, model_filepath,
                             epochs, rbox_loss, evaluate_every, device, wandb_logger)
    model.train()
    metrics = model.test()
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    wandb_logger.log(metrics)
        

def dataset_resolver(dataset_name):
    if dataset_name.lower() == "goslim":
        root_dir = "../data/goslim_generic_existential/"
    elif dataset_name.lower() == "go":
        root_dir = "../data/go_existential/"
    elif dataset_name.lower() == "goplus":
        root_dir = "../data/go-plus_existential/"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return root_dir, PathDataset(root_dir + "train.owl", root_dir + "valid.owl", root_dir + "test.owl")


def module_resolver(module_name, *args, **kwargs):
    if module_name.lower() == "elem":
        return ELEmModule(*args, **kwargs)
    elif module_name.lower() == "elbox":
        return ELBoxModule(*args, **kwargs)
    elif module_name.lower() == "box2el":
        return BoxSquaredELModule(*args, **kwargs)
    else:
        raise ValueError(f"Module {module_name} not found")


def evaluator_resolver(evaluator_name, *args, **kwargs):
    if evaluator_name.lower() == "relation":
        return RelationEvaluator(*args, **kwargs)
    else:
        raise ValueError(f"Evaluator {evaluator_name} not found")


class GeometricELModel(EmbeddingELModel):
    def __init__(self, module_name, evaluator_name, dataset, batch_size,
                 embed_dim, module_margin, loss_margin, learning_rate,
                 model_filepath, epochs, rbox_loss,
                 evaluate_every, device, wandb_logger):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath)

        self.module = module_resolver(module_name,
                                      len(self.dataset.classes),
                                      len(self.dataset.object_properties),
                                      self.embed_dim,
                                      module_margin)

        self.module.trans_slack = nn.Parameter(th.randn(1))
        self.module.inv_slack = nn.Parameter(th.randn(1))
        self.module.sub_slack = nn.Parameter(th.randn(1))
        
        self.evaluator = evaluator_resolver(evaluator_name, dataset, device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.evaluate_every = evaluate_every
        self.loss_margin = loss_margin
        self.device = device
        self.wandb_logger = wandb_logger

        self.rbox_loss = rbox_loss
        if rbox_loss:
            self.rbox_data = self.process_rbox_axioms()
        
            
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
                

        rbox_data = {k: th.tensor(v, dtype=th.long) for k, v in rbox_data.items()}
        return rbox_data
                
        
        
    def tbox_forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def rbox_forward(self, *args, **kwargs):
        num_samples = 100
        # samples = th.rand(num_samples, self.embed_dim, device=self.device).unsqueeze(0)

        ####
        subobjectproperty = self.rbox_data["subobjectproperty"].to(self.device)
        subs = self.module.rel_embed(subobjectproperty[:, 0])
        sups = self.module.rel_embed(subobjectproperty[:, 1])
        
        target_sub = subs.unsqueeze(1)
        target_sup = sups.unsqueeze(1)

        sub_loss = th.linalg.norm(target_sub + self.module.sub_slack - target_sup, dim=-1).mean()
        
        loss = sub_loss

        ###
        inverseproperty = self.rbox_data["inverseproperty"].to(self.device)
        firsts = self.module.rel_embed(inverseproperty[:, 0])
        seconds = self.module.rel_embed(inverseproperty[:, 1])
        diff = th.linalg.norm(firsts + self.module.inv_slack  + seconds, dim=-1).mean() 
        loss += diff

        ###
        transitiveproperty = self.rbox_data["transitiveproperty"].to(self.device)
        prop = self.module.rel_embed(transitiveproperty)
        prop = prop.unsqueeze(1)
        target = prop
        target2 = 2*prop
        trans = th.linalg.norm(target + self.module.trans_slack - target2, dim=-1).mean()
        loss += trans
        
        return loss
        

    
    def train(self):

        dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True)
               for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        total_dls_size = sum(dls_sizes.values())
        dls_weights = {gci_name: ds_size/total_dls_size for gci_name, ds_size in dls_sizes.items()}

        main_dl = dls["gci0"]
        logger.info(f"Training with {len(main_dl)} batches of size {self.batch_size}")
        dls = {gci_name: cycle(dl) for gci_name, dl in dls.items() if gci_name != "gci0"}
        logger.info(f"Dataloaders: {dls_sizes}")

        tolerance = 5
        curr_tolerance = tolerance

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

        min_lr = self.learning_rate / 10
        max_lr = self.learning_rate
        train_steps = int(math.ceil(len(main_dl) / self.batch_size))
        step_size_up = 2 * train_steps
        scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size_up, cycle_momentum=False)


        scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=1000, step_size_down=1000, cycle_momentum=False)
        
        best_mrr = 0

        num_classes = len(self.dataset.classes)

        self.module = self.module.to(self.device)
        for epoch in tqdm(range(self.epochs)):
            self.module.train()
            

            total_train_loss = 0
            total_rbox_loss = 0
            
            for batch_data in main_dl:

                batch_data = batch_data.to(self.device)
                pos_logits = self.tbox_forward(batch_data, "gci0")
                neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                neg_batch = th.cat([batch_data[:, :1], neg_idxs.unsqueeze(1)], dim=1)
                neg_logits = self.tbox_forward(neg_batch, "gci0")
                loss = - F.logsigmoid(-pos_logits + neg_logits - self.loss_margin).mean() * dls_weights["gci0"]

                for gci_name, gci_dl in dls.items():
                    if gci_name == "gci0":
                        continue

                    

                    
                    batch_data = next(gci_dl).to(self.device)
                    pos_logits = self.tbox_forward(batch_data, gci_name)
                    neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                    neg_batch = th.cat([batch_data[:, :2], neg_idxs.unsqueeze(1)], dim=1)
                    neg_logits = self.tbox_forward(neg_batch, gci_name)
                    loss += - F.logsigmoid(-pos_logits + neg_logits - self.loss_margin).mean() * dls_weights[gci_name]


                if self.rbox_loss:
                    rbox_loss = self.rbox_forward()
                    loss += rbox_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                scheduler.step()

                total_train_loss += loss.item()
                if self.rbox_loss:
                    total_rbox_loss += rbox_loss.item()
                    
            if epoch % self.evaluate_every == 0:
                valid_metrics = self.evaluator.evaluate(self.module, mode="valid")
                valid_mrr = valid_metrics["valid_mrr"]
                valid_mr = valid_metrics["valid_mr"]
                valid_metrics["train_loss"] = total_train_loss
                if self.rbox_loss:
                    valid_metrics["rbox_loss"] = rbox_loss
                self.wandb_logger.log(valid_metrics)

                if valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    curr_tolerance = tolerance
                    th.save(self.module.state_dict(), self.model_filepath)
                else:
                    curr_tolerance -= 1

                if curr_tolerance == 0:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                logger.info(f"Epoch {epoch} - Train Loss: {total_train_loss:4f} - RBox Loss: {total_rbox_loss} - Valid MRR: {valid_mrr:4f} - Valid MR: {valid_mr:4f}")

    def test(self):
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()
        return self.evaluator.evaluate(self.module)



class DummyLogger():
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
    
if __name__ == "__main__":
    main()
