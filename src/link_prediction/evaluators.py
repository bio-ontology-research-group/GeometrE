from mowl.projection import TaxonomyProjector, TaxonomyWithRelationsProjector, Edge
from mowl.utils.data import FastTensorDataLoader
import torch as th
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Evaluator:
    def __init__(self, dataset, device, batch_size=16, evaluate_with_deductive_closure=False, filter_deductive_closure=False):

        if evaluate_with_deductive_closure and filter_deductive_closure:
            raise ValueError("Cannot evaluate with deductive closure and filter it at the same time. Set either evaluate_with_deductive_closure or filter_deductive_closure to False.")

        logger.info(f"Evaluating with deductive closure: {evaluate_with_deductive_closure}")
        logger.info(f"Filtering deductive closure: {filter_deductive_closure}")
        
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.train_tuples = self.create_tuples(dataset.ontology)
        self.valid_tuples = self.create_tuples(dataset.validation)
        self.test_tuples = self.create_tuples(dataset.testing)
        self._deductive_closure_tuples = None

        self.evaluate_with_deductive_closure = evaluate_with_deductive_closure
        self.filter_deductive_closure = filter_deductive_closure
        
        self.class_to_id = {c: i for i, c in enumerate(self.dataset.classes.as_str)}
        print(f"Number of classes: {len(self.class_to_id)}")
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        self.relation_to_id = {r: i for i, r in enumerate(self.dataset.object_properties.as_str)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        eval_heads, eval_tails = self.dataset.evaluation_classes
        
        print(f"Number of evaluation classes: {len(eval_heads)}")
        self.evaluation_heads = th.tensor([self.class_to_id[c] for c in eval_heads.as_str], dtype=th.long)
        self.evaluation_tails = th.tensor([self.class_to_id[c] for c in eval_tails.as_str], dtype=th.long)


    @property
    def deductive_closure_tuples(self):
        if self._deductive_closure_tuples is None:
            self._deductive_closure_tuples = self.create_tuples(self.dataset.deductive_closure_ontology)
        return self._deductive_closure_tuples
        
    def create_tuples(self, ontology):
        raise NotImplementedError

    def get_logits(self, batch):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        model = args[0]
        mode = kwargs.get("mode")
        
        if mode == "valid":
            eval_tuples = self.valid_tuples
        else:
            eval_tuples = self.test_tuples

        return self.evaluate_base(model, eval_tuples, **kwargs)

    
    def evaluate_base(self, model, eval_tuples, mode="test", **kwargs):
        num_heads, num_tails = len(self.evaluation_heads), len(self.evaluation_tails)
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")


        if self.evaluate_with_deductive_closure:
            mask1 = (self.deductive_closure_tuples.unsqueeze(1) == self.train_tuples).all(dim=-1).any(dim=-1)
            mask2 = (self.deductive_closure_tuples.unsqueeze(1) == self.valid_tuples).all(dim=-1).any(dim=-1)
            mask = mask1 | mask2
            deductive_closure_tuples = self.deductive_closure_tuples[~mask]
            
            # eval_tuples = th.cat([eval_tuples, deductive_closure_tuples], dim=0)
            eval_tuples = deductive_closure_tuples
        dataloader = FastTensorDataLoader(eval_tuples, batch_size=self.batch_size, shuffle=False)

        metrics = dict()
        mrr, fmrr = 0, 0
        mr, fmr = 0, 0
        ranks, franks = dict(), dict()

        if mode == "test":
            hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
            f_hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        
            filtering_labels = self.get_filtering_labels(num_heads, num_tails, **kwargs)
            if self.evaluate_with_deductive_closure:
                deductive_labels = self.get_deductive_labels(num_heads, num_tails, **kwargs)
            
        with th.no_grad():
            for batch, in dataloader:
                if batch.shape[1] == 2:
                    heads, tails = batch[:, 0], batch[:, 1]
                elif batch.shape[1] == 3:
                    heads, tails = batch[:, 0], batch[:, 2]
                else:
                    raise ValueError("Batch shape must be either (n, 2) or (n, 3)")
                aux_heads = heads.clone()
                aux_tails = tails.clone()
        
                batch = batch.to(self.device)
                logits_heads, logits_tails = self.get_logits(model, batch, *kwargs)
    
                for i, head in enumerate(aux_heads):
                    tail = tails[i]
                    tail = th.where(self.evaluation_tails == tail)[0].item()
                    preds = logits_heads[i]

                    if self.evaluate_with_deductive_closure:
                        ded_labels = deductive_labels[head].to(preds.device)
                        ded_labels[tail] = 1
                        preds = preds * ded_labels

                    
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == tail)[0].item() + 1
                    mr += rank
                    mrr += 1 / rank

                    if mode == "test":
                        f_preds = preds * filtering_labels[head].to(preds.device)

                        if self.evaluate_with_deductive_closure:
                            ded_labels = deductive_labels[head].to(preds.device)
                            ded_labels[tail] = 1
                            f_preds = f_preds * ded_labels

                        f_order = th.argsort(f_preds, descending=False)
                        f_rank = th.where(f_order == tail)[0].item() + 1
                        fmr += f_rank
                        fmrr += 1 / f_rank
                    
                                                                
                    if mode == "test":
                        for k in hits_k:
                            if rank <= int(k):
                                hits_k[k] += 1

                        for k in f_hits_k:
                            if f_rank <= int(k):
                                f_hits_k[k] += 1

                        if rank not in ranks:
                            ranks[rank] = 0
                        ranks[rank] += 1
                                
                        if f_rank not in franks:
                            franks[f_rank] = 0
                        franks[f_rank] += 1

                for i, tail in enumerate(aux_tails):
                    head = aux_heads[i]
                    head = th.where(self.evaluation_heads == head)[0].item()
                    preds = logits_tails[i]

                    if self.evaluate_with_deductive_closure:
                        ded_labels = deductive_labels[:, tail].to(preds.device)
                        ded_labels[head] = 1
                        preds = preds * ded_labels
                    
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == head)[0].item() + 1
                    mr += rank
                    mrr += 1 / rank

                    if mode == "test":
                        f_preds = preds * filtering_labels[:, tail].to(preds.device)

                        if self.evaluate_with_deductive_closure:
                            ded_labels = deductive_labels[:, tail].to(preds.device)
                            ded_labels[head] = 1
                            f_preds = f_preds * ded_labels

                        
                        f_order = th.argsort(f_preds, descending=False)
                        f_rank = th.where(f_order == head)[0].item() + 1
                        fmr += f_rank
                        fmrr += 1 / f_rank
                    

                    if mode == "test":
                        for k in hits_k:
                            if rank <= int(k):
                                hits_k[k] += 1

                        for k in f_hits_k:
                            if f_rank <= int(k):
                                f_hits_k[k] += 1

                        if rank not in ranks:
                            ranks[rank] = 0
                        ranks[rank] += 1
                                
                        if f_rank not in franks:
                            franks[f_rank] = 0
                        franks[f_rank] += 1
                                
            mr = mr / (2 * len(eval_tuples))
            mrr = mrr / (2 * len(eval_tuples))

            metrics["mr"] = mr
            metrics["mrr"] = mrr

            if mode == "test":
                fmr = fmr / (2 * len(eval_tuples))
                fmrr = fmrr / (2 * len(eval_tuples))
                auc = compute_rank_roc(ranks, num_tails)
                f_auc = compute_rank_roc(franks, num_tails)

                metrics["f_mr"] = fmr
                metrics["f_mrr"] = fmrr
                metrics["auc"] = auc
                metrics["f_auc"] = f_auc
                
                for k in hits_k:
                    hits_k[k] = hits_k[k] / (2 * len(eval_tuples))
                    metrics[f"hits@{k}"] = hits_k[k]
                    
                for k in f_hits_k:
                    f_hits_k[k] = f_hits_k[k] / (2 * len(eval_tuples))
                    metrics[f"f_hits@{k}"] = f_hits_k[k]

            metrics = {f"{mode}_{k}": v for k, v in metrics.items()}
            return metrics

        
class RelationKGEvaluator(Evaluator):
    def __init__(self, data_root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_trans_only_tuples = None
        self.data_root = data_root
        
    @property
    def test_only_transitive_tuples(self):
        if self._test_trans_only_tuples is None:
            self._test_trans_only_tuples = self.create_tuples(self.dataset.transitive_test_ontology)
        return self._test_trans_only_tuples

    def get_relation_properties(self):
        roles = self.dataset.ontology.getObjectPropertiesInSignature()
        rbox_axioms = self.dataset.ontology.getRBoxAxioms(Imports.fromBoolean(True))
        transitive_roles = []
        
        for axiom in rbox_axioms:
            axiom_type = axiom.getAxiomType()
            if axiom_type == Ax.TRANSITIVE_OBJECT_PROPERTY:
                property_ = axiom.getProperty()
                transitive_roles.append(str(property_.toStringID()))
                         
        return roles, transitive_roles

    def create_tuples(self, ontology):
        projector = TaxonomyWithRelationsProjector(relations=self.dataset.object_properties.as_str)
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()

        rel_str2owl = self.dataset.object_properties.to_dict()
        rel_owl2idx = self.dataset.object_properties.to_index_dict()
        
        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            rel = rel_owl2idx[rel_str2owl[e.rel]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, rel, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)
        
    def get_filtering_labels(self, num_heads, num_tails, relation_id=-1):

        filtering_tuples = th.cat([self.train_tuples, self.valid_tuples], dim=0)
        filtering_tuples = filtering_tuples[filtering_tuples[:, 1] == relation_id]
                
        filtering_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, rel, tail in filtering_tuples:
            filtering_labels[head, tail] = 10000
            filtering_labels[head, head] = 10000
            filtering_labels[tail, tail] = 10000
        
            
        number_of_filtered_values = th.sum(filtering_labels == 10000)
        return filtering_labels

    def get_deductive_labels(self, num_heads, num_tails, relation_id=-1):
        deductive_labels = th.ones((num_heads, num_tails), dtype=th.float)

        deductive_tuples = self.deductive_closure_tuples[self.deductive_closure_tuples[:, 1] == relation_id]

        for head, rel, tail in self.deductive_closure_tuples:
             deductive_labels[head, tail] = 10000
             deductive_labels[head, head] = 10000
             deductive_labels[tail, tail] = 10000
             
        return deductive_labels

    def get_logits(self, model, batch, relation_id=-1):
        heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        num_heads, num_tails = len(heads), len(tails)
        aux_rels = rels.clone()
        
        heads = heads.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        rels = aux_rels.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        eval_tails = self.evaluation_tails.repeat(num_heads).to(heads.device).unsqueeze(1)
        logits_heads = model(th.cat([heads, rels, eval_tails], dim=-1), "gci2")
        logits_heads = logits_heads.view(-1, len(self.evaluation_tails))
        
        tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        rels = aux_rels.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        eval_heads = self.evaluation_heads.repeat(num_tails).to(tails.device).unsqueeze(1)
        logits_tails = model(th.cat([eval_heads, rels, tails], dim=-1), "gci2")
        logits_tails = logits_tails.view(-1, len(self.evaluation_heads))

        return logits_heads, logits_tails

    def evaluate_overall(self, model, mode="test", relations_to_evaluate=None):
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")

        if relations_to_evaluate is None:
            relations_to_evaluate = range(len(self.dataset.object_properties))
                    
        results = dict()


        if mode == "valid":
            eval_tuples = self.valid_tuples
        elif mode == "test":
            eval_tuples = self.test_tuples

            if self.evaluate_with_deductive_closure:
                train_tuples = self.train_tuples
                valid_tuples = self.valid_tuples
                deductive_tuples = self.deductive_closure_tuples        
                mask1 = (deductive_tuples.unsqueeze(1) == train_tuples).all(dim=-1).any(dim=-1)
                mask2 = (deductive_tuples.unsqueeze(1) == valid_tuples).all(dim=-1).any(dim=-1)
                mask = mask1 | mask2
                deductive_closure_tuples = deductive_tuples[~mask]
                eval_tuples = deductive_closure_tuples

        logger.info(f"Shape of eval_tuples: {eval_tuples.shape}")

        all_ranks, all_franks = dict(), dict()
        num_eval_tuples = 0
        for rel in relations_to_evaluate:
            rel_str = self.id_to_relation[rel]
            logger.debug(f"Evaluating relation {rel_str}. Id: {rel}")
            logger.debug(f"Shape of eval_tuples: {eval_tuples.shape}")
            mask = eval_tuples[:, 1] == rel
            rel_eval_tuples = eval_tuples[mask]
            num_eval_tuples += 2*len(rel_eval_tuples)
            if len(rel_eval_tuples) == 0:
                logger.debug(f"No evaluation tuples for relation {rel_str}. Skipping...")
                continue
            ranks, franks = self.evaluate_base_return_ranks(model, rel_eval_tuples, mode=mode, relation_id=rel)

            for rank, count in ranks.items():
                if not rank in all_ranks:
                    all_ranks[rank] = 0
                all_ranks[rank] += count

            for frank, count in franks.items():
                if not frank in all_franks:
                    all_franks[frank] = 0
                all_franks[frank] += count

        num_test_points = sum(all_ranks.values())
        assert num_eval_tuples == num_test_points, f"num_eval_tuples: {num_eval_tuples} does not match num_test_points: {num_test_points}"
        mr, fmr = 0, 0
        mrr, fmrr = 0, 0
        hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        f_hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})

        for rank, count in all_ranks.items():
            mr += rank * count
            mrr += (1/rank) * count
            for k, v in hits_k.items():
                if rank <= int(k):
                    hits_k[k] += count

        for frank, count in all_franks.items():
            fmr += frank * count
            fmrr += (1/frank) * count
            for k, v in f_hits_k.items():
                if frank <= int(k):
                    f_hits_k[k] += count


        auc = compute_rank_roc(all_ranks, len(self.evaluation_tails))
        fauc = compute_rank_roc(all_franks, len(self.evaluation_tails))

        metrics = dict()
        metrics["mr"] = mr/num_test_points
        metrics["mrr"] = mrr/num_test_points
        for k, v in hits_k.items():
            metrics[f"hits@{k}"] = v/num_test_points
        metrics["auc"] = auc

        metrics["f_mr"] = fmr/num_test_points
        metrics["f_mrr"] = fmrr/num_test_points
        for k, v in f_hits_k.items():
            metrics[f"f_hits@{k}"] = v/num_test_points
        metrics["f_auc"] = fauc

        metrics = {f"{mode}_{k}": v for k, v in metrics.items()}
        return metrics
                                                                                                            
    def evaluate_base_return_ranks(self, model, eval_tuples, mode="test", relation_id = -1, **kwargs):
        num_heads, num_tails = len(self.evaluation_heads), len(self.evaluation_tails)
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")


        dataloader = FastTensorDataLoader(eval_tuples, batch_size=self.batch_size, shuffle=False)
        
        metrics = dict()
        ranks, franks = dict(), dict()

        deductive_labels = self.get_deductive_labels(num_heads, num_tails, relation_id = relation_id, **kwargs)
        if mode == "test":
            filtering_labels = self.get_filtering_labels(num_heads, num_tails, relation_id = relation_id, **kwargs)
            
        with th.no_grad():
            for batch, in dataloader: #tqdm(dataloader):
                if batch.shape[1] == 2:
                    heads, tails = batch[:, 0], batch[:, 1]
                elif batch.shape[1] == 3:
                    heads, tails = batch[:, 0], batch[:, 2]
                else:
                    raise ValueError("Batch shape must be either (n, 2) or (n, 3)")
                aux_heads = heads.clone()
                aux_tails = tails.clone()
        
                batch = batch.to(self.device)
                logits_heads, logits_tails = self.get_logits(model, batch, *kwargs)
    
                for i, head in enumerate(aux_heads):
                    tail = tails[i]

                    perm = th.randperm(num_tails)
                    tail = th.where(self.evaluation_tails[perm] == tail)[0].item()
                    head_perm = th.where(self.evaluation_heads[perm] == head)[0].item()
                    # print(len(self.evaluation_tails), tail)
                    # tail = th.where(self.evaluation_tails == tail)[0].item()
                    preds = logits_heads[i][perm]
                    preds[head_perm] = 10000
                    
                    if self.evaluate_with_deductive_closure or True:
                        ded_labels = deductive_labels[head][perm].to(preds.device)
                        ded_labels[tail] = 1
                        ded_labels[head_perm] = 1
                        preds = preds * ded_labels

                    
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == tail)[0].item() + 1
                                        
                    if mode == "test":
                        filtering = filtering_labels[head][perm].to(preds.device)
                        
                        if self.evaluate_with_deductive_closure:
                            ded_labels = deductive_labels[head][perm].to(preds.device)
                            all_filtering = th.max(filtering, ded_labels)
                            
                        else:
                            ded_labels = deductive_labels[head][perm].to(preds.device)
                            all_filtering = th.max(filtering, ded_labels)
                            
                            # all_filtering = filtering
                        all_filtering[tail] = 1
                        all_filtering[head_perm] = 1
                        f_preds = preds * all_filtering

                        f_order = th.argsort(f_preds, descending=False)
                        f_rank = th.where(f_order == tail)[0].item() + 1
                        assert f_rank <= rank, f"Rank: {rank}, F-Rank: {f_rank}"
                    
                                                                
                    
                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    if mode == "test":
                        if f_rank not in franks:
                            franks[f_rank] = 0
                        franks[f_rank] += 1

                for i, tail in enumerate(aux_tails):
                    head = aux_heads[i]
                    perm = th.randperm(num_heads)
                    head = th.where(self.evaluation_heads[perm] == head)[0].item()
                    tail_perm = th.where(self.evaluation_tails[perm] == tail)[0].item()
                    preds = logits_tails[i][perm]
                    preds[tail_perm] = 10000
                    
                    if self.evaluate_with_deductive_closure or True:
                        ded_labels = deductive_labels[:, tail][perm].to(preds.device)
                        ded_labels[head] = 1
                        ded_labels[tail_perm] = 1
                        preds = preds * ded_labels
                    
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == head)[0].item() + 1

                    if mode == "test":
                        filtering = filtering_labels[:, tail][perm].to(preds.device)
                        # f_preds = preds * filtering_labels[head].to(preds.device)

                        if self.evaluate_with_deductive_closure:
                            ded_labels = deductive_labels[:, tail][perm].to(preds.device)
                            all_filtering = th.max(filtering, ded_labels)
                            
                        else:
                            ded_labels = deductive_labels[:, tail][perm].to(preds.device)
                            all_filtering = th.max(filtering, ded_labels)
                        
                            # all_filtering = filtering
                        all_filtering[head] = 1
                        all_filtering[tail_perm] = 1
                        f_preds = preds * all_filtering

                        f_order = th.argsort(f_preds, descending=False)
                        f_rank = th.where(f_order == head)[0].item() + 1
                        assert f_rank <= rank, f"Rank: {rank}, F-Rank: {f_rank}"
                    

                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1
                        
                    if mode == "test":
                                
                        if f_rank not in franks:
                            franks[f_rank] = 0
                        franks[f_rank] += 1
                                
                                                
        return ranks, franks

    def evaluate_chains(self, model, entity_prefix, rel_prefix):

        chain_file = os.path.join(self.data_root, "chains.txt")

        test_data = pd.read_csv(chain_file, sep="\t", header=None, dtype=str)
        test_data.columns = ["relation", "head", "tail", "next_tail"]

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()

        rel_str2owl = self.dataset.object_properties.to_dict()
        rel_owl2idx = self.dataset.object_properties.to_index_dict()

        data_indexed = []
        for i, row in test_data.iterrows():

            relation = rel_prefix + row["relation"]
            relation = rel_owl2idx[rel_str2owl[relation]]

            head = entity_prefix + str(row["head"])
            tail = entity_prefix + str(row["tail"])
            next_tail = entity_prefix + str(row["next_tail"])

            head = class_owl2idx[class_str2owl[head]]
            tail = class_owl2idx[class_str2owl[tail]]
            next_tail = class_owl2idx[class_str2owl[next_tail]]
            
            data_indexed.append((relation, head, tail, next_tail))

        data_indexed = th.tensor(data_indexed, dtype=th.long)
                                               
        dataloader = FastTensorDataLoader(data_indexed, batch_size=self.batch_size, shuffle=False)

        all_logits = []
        for batch, in dataloader:
            batch = batch.to(self.device)
            
            rel = batch[:, 0].unsqueeze(1)
            head = batch[:, 1].unsqueeze(1)
            tail = batch[:, 2].unsqueeze(1)
            next_tail = batch[:, 3].unsqueeze(1)

            train_triple = th.cat([head, rel, tail], dim=1)

            assert len(train_triple) == len(batch), f"Length of train triple: {len(train_triple)} does not match batch length: {len(batch)}"
            
            test_triple = th.cat([tail, rel, next_tail], dim=1)
            extra_triple = th.cat([head, rel, next_tail], dim=1)

            all_triples = th.cat([train_triple, test_triple, extra_triple], dim=0)

            if return_scores:
                logits = model(all_triples, "gci2").view(-1, 3).detach().cpu().numpy().tolist()
                assert len(logits) == len(batch), f"Length of logits: {len(logits)} does not match batch length: {len(batch)}"
                for i in range(len(logits)):
                    rel_and_logits = [rel[i].item()] + logits[i]
                    all_logits.append(rel_and_logits)
                
            
            logits_heads, logits_tails = self.get_logits(model, all_triples, *kwargs)

            
        return all_logits


    def evaluate_chains_return_ranks(self, model, entity_prefix, rel_prefix, relation_id = -1, **kwargs):
        if relation_id == -1:
            raise ValueError("Relation id must be provided for chain evaluation")
        
        num_heads, num_tails = len(self.evaluation_heads), len(self.evaluation_tails)
                            
        chain_file = os.path.join(self.data_root, "chains.txt")

        test_data = pd.read_csv(chain_file, sep="\t", header=None, dtype=str)
        test_data.columns = ["relation", "head", "tail", "next_tail"]

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()

        rel_str2owl = self.dataset.object_properties.to_dict()
        rel_owl2idx = self.dataset.object_properties.to_index_dict()

        data_indexed = []
        for i, row in test_data.iterrows():

            relation = rel_prefix + row["relation"]
            relation = rel_owl2idx[rel_str2owl[relation]]

            if relation != relation_id:
                continue
            
            head = entity_prefix + str(row["head"])
            tail = entity_prefix + str(row["tail"])
            next_tail = entity_prefix + str(row["next_tail"])

            head = class_owl2idx[class_str2owl[head]]
            tail = class_owl2idx[class_str2owl[tail]]
            next_tail = class_owl2idx[class_str2owl[next_tail]]
            
            data_indexed.append((relation, head, tail, next_tail))

        data_indexed = th.tensor(data_indexed, dtype=th.long)

        dataloader = FastTensorDataLoader(data_indexed, batch_size=self.batch_size, shuffle=False)
        
        metrics = dict()

        final_ranks = list()
        # ranks, franks = dict(), dict()

        deductive_labels = self.get_deductive_labels(num_heads, num_tails, relation_id = relation_id, **kwargs).to(self.device)
        filtering_labels = self.get_filtering_labels(num_heads, num_tails, relation_id = relation_id, **kwargs)

        self.evaluation_heads = self.evaluation_heads.to(self.device)
        self.evaluation_tails = self.evaluation_tails.to(self.device)
        
        with th.no_grad():
            for batch, in dataloader: #tqdm(dataloader):
                batch = batch.to(self.device)

                rel = batch[:, 0].unsqueeze(1)
                head = batch[:, 1].unsqueeze(1)
                tail = batch[:, 2].unsqueeze(1)
                next_tail = batch[:, 3].unsqueeze(1)

                train_triples = th.cat([head, rel, tail], dim=1)
                assert len(train_triples) == len(batch), f"Length of train triples: {len(train_triples)} does not match batch length: {len(batch)}"

                test_triples = th.cat([tail, rel, next_tail], dim=1)
                extra_triples = th.cat([head, rel, next_tail], dim=1)

                aux_heads = th.cat([head, tail, head], dim=0).squeeze()
                tails = th.cat([tail, next_tail, next_tail], dim=0).squeeze()
                
                 
                logits_heads_train, _ = self.get_logits(model, train_triples, *kwargs)
                logits_heads_test, _ = self.get_logits(model, test_triples, *kwargs)
                logits_heads_extra, _ = self.get_logits(model, extra_triples, *kwargs)

                all_logits = th.cat([logits_heads_train, logits_heads_test, logits_heads_extra], dim=0)
                
                batch_ranks = list()
                
                for i, head in enumerate(aux_heads):
                    tail = tails[i]

                    perm = th.randperm(num_tails)
                    tail = th.where(self.evaluation_tails[perm] == tail)[0].item()
                    head_perm = th.where(self.evaluation_heads[perm] == head)[0].item()
                    
                    preds = all_logits[i][perm]
                    preds[head_perm] = 10000

                    ded_labels = deductive_labels[head][perm].to(preds.device)
                    ded_labels[tail] = 1
                    ded_labels[head_perm] = 1
                    # preds = preds * ded_labels

                    
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == tail)[0].item() + 1
                                        
                    batch_ranks.append(rank)

                batch_ranks = th.tensor(batch_ranks, dtype=th.long).reshape(-1, 3)
                assert len(batch_ranks) == len(batch), f"Length of batch ranks: {len(batch_ranks)} does not match batch length: {len(batch)}"

                final_ranks += batch_ranks.tolist()

            
        return final_ranks

    def evaluate_by_property(self, model, transitive_properties):
        model.eval()
        metrics = self.evaluate_overall(model, relations_to_evaluate=transitive_properties, mode="test")

        return metrics


    
def compute_rank_roc(ranks, num_entities):
    n_tails = num_entities
                    
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_tails)
    auc_y.append(1)
    auc = np.trapz(auc_y, auc_x) / n_tails
    return auc



