import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.utils.random import seed_everything
from mowl.utils.data import FastTensorDataLoader
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import RelationEvaluator
from datasets import SubsumptionDataset
from tqdm import tqdm
from module import TransitiveELModule
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
logger.setLevel(logging.INFO)

th.autograd.set_detect_anomaly(True)

@ck.command()
@ck.option("--dataset_name", "-ds", type=ck.Choice(["go_existential", "foodon_existential"]), default="go_existential")
@ck.option("--evaluator_name", "-e", default="subsumption", help="Evaluator to use")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=78000, help="Batch size")
@ck.option("--module_margin", "-mm", default=0.1, help="Margin for the module")
@ck.option("--loss_margin", "-lm", default=0.1, help="Margin for the loss function")
@ck.option("--learning_rate", "-lr", default=0.001, help="Learning rate")
@ck.option("--epochs", "-ep", default=10000, help="Number of epochs")
@ck.option("--evaluate_every", "-every", default=50, help="Evaluate every n epochs")
@ck.option("--evaluate_deductive", "-evalded", is_flag=True, help="Use deductive closure as positive examples for evaluation")
@ck.option("--filter_deductive", "-filterded", is_flag=True, help="Filter out examples from deductive closure")
@ck.option("--transitive", "-trans", type=ck.Choice(["yes", "no"]))
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(dataset_name, evaluator_name, embed_dim, batch_size,
         module_margin, loss_margin, learning_rate, epochs,
         evaluate_every, evaluate_deductive, filter_deductive,
         transitive, device, wandb_description, no_sweep, only_test):

    seed_everything(42)

    if transitive == "yes":
        transitive = True
    elif transitive == "no":
        transitive = False
    else:
        raise ValueError(f"Transitive must be either 'yes' or 'no'")
    
    wandb_logger = wandb.init(entity="ferzcam", project="transEL", group="f{dataset_name}_{transitive}", name=wandb_description)


    if loss_margin == int(loss_margin):
        loss_margin = int(loss_margin)
    if module_margin == int(module_margin):
        module_margin = int(module_margin)
    
    if no_sweep:
        wandb_logger.log({"dataset_name": dataset_name,
                          "embed_dim": embed_dim,
                          "module_margin": module_margin,
                          "learning_rate": learning_rate,
                          "transitive": transitive
                          })
    else:
        dataset_name = wandb.config.dataset_name
        embed_dim = wandb.config.embed_dim
        module_margin = wandb.config.module_margin
        learning_rate = wandb.config.learning_rate
        transitive = wandb.config.transitive
    
    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    model_filepath = f"{model_dir}/{embed_dim}_{batch_size}_{module_margin}_{loss_margin}_{learning_rate}_{transitive}.pt"
    model = GeometricELModel(evaluator_name, dataset, batch_size,
                             embed_dim, module_margin, loss_margin,
                             learning_rate, model_filepath, epochs,
                             evaluate_every, evaluate_deductive,
                             filter_deductive, transitive, device, wandb_logger)

    
    if not only_test:
        model.train()
     

    # wandb_logger.log(metrics)
    
    metrics = model.test_by_property()["transitive_properties"]

    
    points_per_rel = [v["num_points"] for v in metrics.values()]
    total_points = sum(points_per_rel)

    avg_metrics = dict()
    for rel, v in metrics.items():
        for metric, value in v.items():
            if metric not in avg_metrics:
                avg_metrics[metric] = 0
            avg_metrics[metric] += value * v["num_points"]/total_points

    wandb_logger.log(avg_metrics)
        
                     
    for rel, rel_metrics in metrics.items():
        # print(f"Relation: {rel}")
        # print(metrics)
        print_as_md(rel_metrics, rel)
            
             
    wandb_logger.finish()
        

def print_as_md(overall_metrics, key=None):
    metrics = ["test_mr", "test_mrr", "test_auc", "test_hits@1", "test_hits@3", "test_hits@10", "test_hits@50", "test_hits@100"]
    filt_metrics = [k.replace("_", "_f_") for k in metrics]

    string_metrics = "| Property | MR | MRR | AUC | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | \n"
    string_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    string_filtered_metrics = "| Property | MR | MRR | AUC | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | \n"
    string_filtered_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"

    if key is not None:
        string_metrics += f"| {key} | "
        string_filtered_metrics += f"| {key} | "

    else:
        string_metrics += "| Overall | "
        string_filtered_metrics += "| Overall | "

    
    for metric in metrics:
        if metric == "test_mr":
            string_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_metrics += f"{overall_metrics[metric]:.4f} | "
    for metric in filt_metrics:
        if metric == "test_f_mr":
            string_filtered_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_filtered_metrics += f"{overall_metrics[metric]:.4f} | "


    print(string_metrics)
    print("\n\n")
    print(string_filtered_metrics)
        
    

    
def dataset_resolver(dataset_name):
    if dataset_name.lower() == "go_existential":
        root_dir = "../../use_cases/go_existential/data/"
    elif dataset_name.lower() == "foodon_existential":
        root_dir = "../../use_cases/foodon_existential/data/"
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return root_dir, SubsumptionDataset(root_dir)
    
def evaluator_resolver(evaluator_name, *args, **kwargs):
    if evaluator_name.lower() == "subsumption":
        return RelationEvaluator(*args, **kwargs)
    else:
        raise ValueError(f"Evaluator {evaluator_name} not found")


class GeometricELModel(EmbeddingELModel):
    def __init__(self, evaluator_name, dataset, batch_size, embed_dim,
                 module_margin, loss_margin, learning_rate,
                 model_filepath, epochs, evaluate_every,
                 evaluate_deductive, filter_deductive, transitive,
                 device, wandb_logger):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath)

        self.transitive = transitive
        self.rbox_data = self.process_rbox_axioms()
        self.transitive_ids = self.rbox_data["transitiveproperty"].to(device)
        self.module = TransitiveELModule(len(self.dataset.classes),
                                         len(self.dataset.object_properties),
                                         len(self.dataset.individuals),
                                         embed_dim= self.embed_dim,
                                         margin=module_margin,
                                         transitive=transitive,
                                         transitive_ids=self.transitive_ids)

        self.evaluator = evaluator_resolver(evaluator_name, dataset, device, evaluate_with_deductive_closure=evaluate_deductive, filter_deductive_closure=filter_deductive)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.evaluate_every = evaluate_every
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
                second = axiom.getSecondProperty().getNamedProperty()

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

        dls = {gci_name: FastTensorDataLoader(ds.data, batch_size=self.batch_size, shuffle=True)
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
        criterion = nn.MSELoss()
        
        best_mrr = 0
        best_loss = float("inf")
        
        num_classes = len(self.dataset.classes)

        self.module = self.module.to(self.device)
        for epoch in tqdm(range(self.epochs)):
            self.module.train()
            

            total_train_loss = 0
            
            for batch_data, in main_dl:
                loss = 0
                batch_data = batch_data.to(self.device)
                pos_logits = self.tbox_forward(batch_data, "gci0")
                neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                neg_batch = th.cat([batch_data[:, :1], neg_idxs.unsqueeze(1)], dim=1)
                neg_logits = self.tbox_forward(neg_batch, "gci0", neg=True)

                pos_logits = 1 - th.exp(-pos_logits)
                neg_logits = 1 - th.exp(-neg_logits)
                loss += criterion(pos_logits, th.zeros_like(pos_logits)) + criterion(neg_logits, th.ones_like(neg_logits))
                
                for gci_name, gci_dl in dls.items():
                    if gci_name == "gci0":
                        continue
                    batch_data, = next(gci_dl)
                    batch_data = batch_data.to(self.device)

                    if gci_name != "gci3":
                        pos_logits = self.tbox_forward(batch_data, gci_name)
                    else:
                        pos_logits, classes_to_fix, dims = self.tbox_forward(batch_data, gci_name)
                    pos_logits = 1 - th.exp(-pos_logits)
                    loss += criterion(pos_logits, th.zeros_like(pos_logits))
                    
                    if gci_name != "gci1_bot":
                        neg_idxs = th.randint(0, num_classes, (len(batch_data),), device=self.device)
                        neg_batch = th.cat([batch_data[:, :2], neg_idxs.unsqueeze(1)], dim=1)
                        if gci_name != "gci3":
                            neg_logits = self.tbox_forward(neg_batch, gci_name, neg=True)
                        else:
                            neg_logits, _, _ = self.tbox_forward(neg_batch, gci_name, neg=True)
                        neg_logits = 1 - th.exp(-neg_logits)
                        loss += criterion(neg_logits, th.ones_like(neg_logits))

                 # if self.transitive:
                    # loss += self.module.regularization_loss()
                
                                                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.module.fix_classes(classes_to_fix, dims)
                
                total_train_loss += loss.item()
                                    
            if epoch % self.evaluate_every == 0:
                valid_metrics = self.evaluator.evaluate(self.module, mode="valid", relations_to_evaluate=self.transitive_ids.tolist())

                valid_mrr = 0
                valid_mr = 0

                points_per_rel = [v["num_points"] for v in valid_metrics.values()]
                total_points = sum(points_per_rel)
                for rel, v in valid_metrics.items():
                    valid_mrr += v["valid_mrr"] * v["num_points"]/total_points
                    valid_mr += v["valid_mr"] * v["num_points"]/total_points
                
                                
                valid_metrics["train_loss"] = total_train_loss
                                    
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

                logger.info(f"Epoch {epoch} - Train Loss: {total_train_loss:4f}  - Valid MRR: {valid_mrr:4f} - Valid MR: {valid_mr:4f}")

    def test(self):
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.to(self.device)
        self.module.eval()
        
        return self.evaluator.evaluate(self.module)


    def test_by_property(self):
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.to(self.device)
        self.module.eval()
        return self.evaluator.evaluate_by_property(self.module)


class DummyLogger():
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass
    
if __name__ == "__main__":
    main()
