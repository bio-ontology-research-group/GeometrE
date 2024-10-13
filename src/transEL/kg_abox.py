import sys
sys.path.append("../")
import mowl
mowl.init_jvm("10g")
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.utils.random import seed_everything
from mowl.utils.data import FastTensorDataLoader
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
from evaluators import RelationKGEvaluator
from datasets import KGDataset
from utils import print_as_md
from tqdm import tqdm
from module_abox import TransitiveELModule
import torch as th
import torch.nn as nn
from itertools import cycle
import logging
import click as ck
import random
import os
import wandb
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@ck.command()
@ck.option("--dataset_name", "-ds", type=ck.Choice(["wn18rr", "wn18rr"]), default="wn18rr")
@ck.option("--evaluator_name", "-e", default="kg", help="Evaluator to use")
@ck.option("--embed_dim", "-dim", default=50, help="Embedding dimension")
@ck.option("--batch_size", "-bs", default=400000, help="Batch size")
@ck.option("--module_margin", "-mm", default=0.1, help="Margin for the module")
@ck.option("--loss_margin", "-lm", default=0.1, help="Margin for the loss function")
@ck.option("--max_bound", "-maxb", default=1.0, help="Maximum bound for the initial embedding space")
@ck.option("--learning_rate", "-lr", default=0.001, help="Learning rate")
@ck.option("--num_negs", "-negs", default=1, help="Number of negative samples")
@ck.option("--epochs", "-ep", default=10000, help="Number of epochs")
@ck.option("--evaluate_every", "-every", default=50, help="Evaluate every n epochs")
@ck.option("--evaluate_deductive", "-evalded", is_flag=True, help="Use deductive closure as positive examples for evaluation")
@ck.option("--transitive", "-trans", type=ck.Choice(["yes", "no"]))
@ck.option("--device", "-d", default="cuda", help="Device to use")
@ck.option("--wandb_description", "-desc", default="default")
@ck.option("--no_sweep", "-ns", is_flag=True)
@ck.option("--only_test", "-ot", is_flag=True)
def main(dataset_name, evaluator_name, embed_dim, batch_size,
         module_margin, loss_margin, max_bound, learning_rate,
         num_negs, epochs, evaluate_every, evaluate_deductive,
         transitive, device, wandb_description, no_sweep, only_test):

    seed_everything(42)

    if transitive == "yes":
        transitive = True
    elif transitive == "no":
        transitive = False
    else:
        raise ValueError(f"Transitive must be either 'yes' or 'no'")
    
    wandb_logger = wandb.init(entity="ferzcam", project="transEL2", group="f{dataset_name}_{transitive}", name=wandb_description)

    if loss_margin == int(loss_margin):
        loss_margin = int(loss_margin)
    if module_margin == int(module_margin):
        module_margin = int(module_margin)
    
    if no_sweep:
        wandb_logger.log({"dataset_name": dataset_name,
                          "embed_dim": embed_dim,
                          "module_margin": module_margin,
                          "loss_margin": loss_margin,
                          "max_bound": max_bound,
                          "learning_rate": learning_rate,
                          "num_negs": num_negs,
                          "transitive": transitive
                          })
    else:
        dataset_name = wandb.config.dataset_name
        embed_dim = wandb.config.embed_dim
        module_margin = wandb.config.module_margin
        loss_margin = wandb.config.loss_margin
        max_bound = wandb.config.max_bound
        learning_rate = wandb.config.learning_rate
        num_negs = wandb.config.num_negs
        transitive = wandb.config.transitive

        if transitive == "yes":
            transitive = True
        elif transitive == "no":
            transitive = False
        else:
            raise ValueError(f"Transitive must be either 'yes' or 'no'")

    root_dir, dataset = dataset_resolver(dataset_name)

    model_dir = f"{root_dir}/../models/"
    os.makedirs(model_dir, exist_ok=True)

    model_filepath = f"{model_dir}/{embed_dim}_{batch_size}_{module_margin}_{loss_margin}_{max_bound}_{learning_rate}_{num_negs}_{transitive}.pt"
    model = GeometricELModel(evaluator_name, dataset, batch_size,
                             embed_dim, module_margin, loss_margin,
                             max_bound, learning_rate, num_negs,
                             model_filepath, epochs, evaluate_every,
                             evaluate_deductive, transitive, device,
                             wandb_logger)

    
    if not only_test:
        model.train()
     
    all_metrics = model.test()
    print_as_md(all_metrics)
    wandb_logger.log(all_metrics)
    wandb_logger.finish()


    
def dataset_resolver(dataset_name):
    root_dir = f"../../use_cases/{dataset_name.lower()}/data/"
    if not os.path.exists(root_dir):
        raise ValueError(f"Path {root_dir} for dataset {dataset_name} not found")
                                 
    return root_dir, KGDataset(root_dir)
    
def evaluator_resolver(evaluator_name, *args, **kwargs):
    if evaluator_name.lower() == "kg":
        return RelationKGEvaluator(*args, **kwargs)
    else:
        raise ValueError(f"Evaluator {evaluator_name} not found")


class GeometricELModel(EmbeddingELModel):
    def __init__(self, evaluator_name, dataset, batch_size, embed_dim,
                 module_margin, loss_margin, max_bound, learning_rate,
                 num_negs, model_filepath, epochs, evaluate_every,
                 evaluate_deductive, transitive,
                 device, wandb_logger):
        super().__init__(dataset, embed_dim, batch_size, model_filepath=model_filepath, load_normalized=True)

        self.transitive = transitive
        self.num_negs = num_negs

        self.relation_to_id = {r: i for i, r in enumerate(self.dataset.object_properties.as_str)}

        self.rbox_data = self.process_rbox_axioms()
        self.transitive_ids = self.rbox_data["transitiveproperty"].to(device)
        self.module = TransitiveELModule(len(self.dataset.classes),
                                         len(self.dataset.object_properties),
                                         len(self.dataset.individuals),
                                         embed_dim= self.embed_dim,
                                         margin=module_margin,
                                         transitive=transitive,
                                         transitive_ids=self.transitive_ids,
                                         max_bound = max_bound
                                         )

        self.evaluator = evaluator_resolver(evaluator_name, dataset,
                                            device, batch_size = 32,
                                            evaluate_with_deductive_closure=evaluate_deductive)
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
                                                                                            
            if axiom_type == Ax.INVERSE_OBJECT_PROPERTIES:
                first = axiom.getFirstProperty()
                second = axiom.getSecondProperty().getNamedProperty()

                first_id = owl2idx[first]
                second_id = owl2idx[second]
                rbox_data["inverseproperty"].append((first_id, second_id))
            elif axiom_type == Ax.TRANSITIVE_OBJECT_PROPERTY:
                property_ = axiom.getProperty()
                property_id = owl2idx[property_]
                rbox_data["transitiveproperty"].append(property_id)

        for prop1, prop2 in rbox_data["inverseproperty"]:
            if prop1 in rbox_data["transitiveproperty"] and prop2 in rbox_data["transitiveproperty"]:
                rbox_data["transitiveproperty"].remove(prop2)

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

        main_dl = dls["gci2"]
        logger.info(f"Training with {len(main_dl)} batches of size {self.batch_size}")
        dls = {gci_name: cycle(dl) for gci_name, dl in dls.items() if gci_name != "gci2"}
        logger.info(f"Dataloaders: {dls_sizes}")

        tolerance = 5
        curr_tolerance = tolerance

        optimizer = th.optim.AdamW(self.module.parameters(), lr=self.learning_rate)
        # optimizer = th.optim.SGD(self.module.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        criterion_bpr = nn.LogSigmoid()
        best_mrr = 0
        best_mr = float("inf")
        best_loss = float("inf")
        
        num_classes = len(self.dataset.classes)

        self.module = self.module.to(self.device)
        for epoch in tqdm(range(self.epochs)):
            self.module.train()
            total_train_loss = 0
            total_reg_loss = 0
            for batch_data, in main_dl:
                loss = 0
                batch_data = batch_data.to(self.device)
                pos_logits = self.tbox_forward(batch_data, "gci2").unsqueeze(1)
                # print(f"Max pos logits: {pos_logits.max()}, Min pos logits: {pos_logits.min()}")
                loss += - criterion_bpr(self.loss_margin - pos_logits).mean()

                neg_idxs = th.randint(0, num_classes, (len(batch_data) * self.num_negs,), device=self.device)
                # if random.random() < 0.5:
                neg_batch = th.cat([batch_data[:, :2].repeat(self.num_negs, 1), neg_idxs.unsqueeze(1)], dim=1)
                                    
                neg_logits = self.tbox_forward(neg_batch, "gci2", neg=True).reshape(-1, self.num_negs)
                # print(f"Max neg logits: {neg_logits.max()}, Min neg logits: {neg_logits.min()}")
                loss += - criterion_bpr(neg_logits - self.loss_margin).mean()
                # print(pos_logits.shape, neg_logits.shape)
                # loss += th.relu(self.loss_margin + pos_logits - neg_logits)
                
                
                neg_idxs = th.randint(0, num_classes, (len(batch_data) * self.num_negs,), device=self.device)
                neg_batch = th.cat([neg_idxs.unsqueeze(1), batch_data[:, 1:].repeat(self.num_negs, 1)], dim=1)
                neg_logits = self.tbox_forward(neg_batch, "gci2", neg=True).reshape(-1, self.num_negs)
                # print(f"Max neg logits: {neg_logits.max()}, Min neg logits: {neg_logits.min()}")
                loss += - criterion_bpr(neg_logits - self.loss_margin).mean()
                # loss += th.relu(self.loss_margin + pos_logits - neg_logits)

                loss = loss.mean()
                
                # for i in range(self.num_negs):
                    # neg_idxs = th.randint(0, num_classes, (len(batch_data),2), device=self.device)

                    
                    # if random.random() < 0.5:
                        # neg_batch = th.cat([batch_data[:, :2], neg_idxs[:, 0].unsqueeze(1)], dim=1)
                        # neg_logits_tail = self.tbox_forward(neg_batch, "gci2", neg=True)
                        # loss += - criterion_bpr(neg_logits_tail - self.loss_margin).mean()
                        # loss += th.relu(self.loss_margin + pos_logits - neg_logits_tail).mean()
                    # else:
                        # neg_batch = th.cat([neg_idxs[:, 1].unsqueeze(1), batch_data[:, 1:]], dim=1)
                        # neg_logits_head = self.tbox_forward(neg_batch, "gci2", neg=True)
                        # loss += th.relu(self.loss_margin + pos_logits - neg_logits_head).mean()
                        # loss += - criterion_bpr(neg_logits_head - self.loss_margin).mean() 
                                                                                
                loss /= self.num_negs
                    
                # if self.transitive:
                reg_loss = self.module.regularization_loss()
                loss += reg_loss

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                total_reg_loss += reg_loss.item()

            # scheduler.step()
            total_train_loss /= len(main_dl)
            total_reg_loss /= len(main_dl)
            if epoch % self.evaluate_every == 0:
                valid_metrics = self.evaluator.evaluate_overall(self.module, mode="valid")
                print(valid_metrics)
                valid_mrr = 0
                valid_mr = 0
                valid_mr = valid_metrics["valid_mr"]
                valid_mrr = valid_metrics["valid_mrr"]
                                
                valid_metrics["train_loss"] = total_train_loss
                                    
                self.wandb_logger.log(valid_metrics)

                if valid_mrr > best_mrr:
                # if valid_mr < best_mr:
                    best_mrr = valid_mrr
                    # best_mr = valid_mr
                    curr_tolerance = tolerance
                    th.save(self.module.state_dict(), self.model_filepath)
                else:
                    curr_tolerance -= 1

                if curr_tolerance == 0:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                logger.info(f"Epoch {epoch} - Train Loss: {total_train_loss:4f} - Reg Loss: {total_reg_loss:4f} - Valid MRR: {valid_mrr:4f} - Valid MR: {valid_mr:4f}")

    def test(self):
        self.module.load_state_dict(th.load(self.model_filepath, map_location=self.device))
        self.module.to(self.device)
        self.module.eval()
        
        return self.evaluator.evaluate_overall(self.module)

if __name__ == "__main__":
    main()
