from mowl.projection import TaxonomyProjector, TaxonomyWithRelationsProjector, Edge
from mowl.utils.data import FastTensorDataLoader
import torch as th
from tqdm import tqdm
import numpy as np
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Evaluator:
    def __init__(self, dataset, device, batch_size=48, evaluate_with_deductive_closure=False, filter_deductive_closure=False):

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
            
            eval_tuples = th.cat([eval_tuples, deductive_closure_tuples], dim=0)
            # eval_tuples = deductive_closure_tuples
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

        
    
    
class SubsumptionEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyProjector()
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()

        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_logits(self, model, batch):
        heads, tails = batch[:, 0], batch[:, 1]
        num_heads, num_tails = len(heads), len(tails)

        heads = heads.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        eval_tails = th.arange(len(self.evaluation_tails), device=heads.device).repeat(num_heads).unsqueeze(1)
        logits_heads = model(th.cat([heads, eval_tails], dim=-1), "gci0")
        logits_heads = logits_heads.view(-1, len(self.evaluation_tails))
        
        tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        eval_heads = th.arange(len(self.evaluation_heads), device=tails.device).repeat(num_tails).unsqueeze(1)
        logits_tails = model(th.cat([eval_heads, tails], dim=-1), "gci0")
        logits_tails = logits_tails.view(-1, len(self.evaluation_heads))
        # print(logits_heads, logits_tails)
        return logits_heads, logits_tails

    
    def get_filtering_labels(self, num_heads, num_tails):

        if self.filter_deductive_closure:
            # take deductive closure tuples that are not in the testing tuples

            mask = (self.deductive_closure_tuples.unsqueeze(1) == self.test_tuples).all(dim=-1).any(dim=-1)
            deductive_closure_tuples = self.deductive_closure_tuples[~mask]
            
            
            filtering_tuples = th.cat([self.train_tuples, self.valid_tuples, deductive_closure_tuples], dim=0)
        else:
            filtering_tuples = th.cat([self.train_tuples, self.valid_tuples], dim=0)

        filtering_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, tail in filtering_tuples:
            filtering_labels[head, tail] = 10000
        
        return filtering_labels
    


    def get_deductive_labels(self, num_heads, num_tails):
        deductive_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, tail in self.deductive_closure_tuples:
            deductive_labels[head, tail] = 10000
        
        return deductive_labels



class PPIEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_tuples(self, ontology):
        projector = TaxonomyWithRelationsProjector(relations=["http://interacts_with"])
        edges = projector.project(ontology)

        classes, relations = Edge.get_entities_and_relations(edges)

        class_str2owl = self.dataset.classes.to_dict()
        class_owl2idx = self.dataset.classes.to_index_dict()
        relation_str2owl = self.dataset.object_properties.to_dict()
        relation_owl2idx = self.dataset.object_properties.to_index_dict()
        
        edges_indexed = []
        
        for e in edges:
            head = class_owl2idx[class_str2owl[e.src]]
            relation = relation_owl2idx[relation_str2owl[e.rel]]
            tail = class_owl2idx[class_str2owl[e.dst]]
            edges_indexed.append((head, relation, tail))
        
        return th.tensor(edges_indexed, dtype=th.long)

    def get_logits(self, model, batch):
        heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        num_heads, num_tails = len(heads), len(tails)

        heads = heads.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        rels = rels.repeat_interleave(len(self.evaluation_tails)).unsqueeze(1)
        eval_tails = th.arange(len(self.evaluation_tails), device=heads.device).repeat(num_heads).unsqueeze(1)
        logits_heads = model(th.cat([heads, rels, eval_tails], dim=-1), "gci2")
        logits_heads = logits_heads.view(-1, len(self.evaluation_tails))
        
        tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        eval_heads = th.arange(len(self.evaluation_heads), device=tails.device).repeat(num_tails).unsqueeze(1)
        logits_tails = model(th.cat([eval_heads, rels, tails], dim=-1), "gci2")
        logits_tails = logits_tails.view(-1, len(self.evaluation_heads))
        # print(logits_heads, logits_tails)
        return logits_heads, logits_tails

    
    def get_filtering_labels(self, num_heads, num_tails):

        filtering_tuples = th.cat([self.train_tuples, self.valid_tuples], dim=0)
        filtering_labels = th.ones((num_heads, num_tails), dtype=th.float)

        for head, rel, tail in filtering_tuples:
            filtering_labels[head, tail] = 10000
        
        return filtering_labels
    

class RelationEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        

    def get_relation_properties(self):
        rbox_axioms = self.dataset.ontology.getRBoxAxioms(Imports.fromBoolean(True))
        subproperties = []
        superproperties = []
        inverse_properties = []
        transitive_properties = []
        
        for axiom in rbox_axioms:
            axiom_type = axiom.getAxiomType()
            if axiom_type == Ax.SUB_OBJECT_PROPERTY:
                
                sub = axiom.getSubProperty()
                sup = axiom.getSuperProperty()
                subproperties.append(str(sub.toStringID()))
                superproperties.append(str(sup.toStringID()))
                                
            elif axiom_type == Ax.INVERSE_OBJECT_PROPERTIES:
                first = axiom.getFirstProperty()
                second = axiom.getSecondProperty()

                inverse_properties.append(str(first.toStringID()))
                inverse_properties.append(str(second.toStringID()))
                                
            elif axiom_type == Ax.TRANSITIVE_OBJECT_PROPERTY:
                property_ = axiom.getProperty()
                transitive_properties.append(str(property_.toStringID()))
                         
        return subproperties, superproperties, inverse_properties, transitive_properties

        
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

        number_of_filtered_values = th.sum(filtering_labels == 10000)
        return filtering_labels

    def get_deductive_labels(self, num_heads, num_tails, relation_id=-1):
        deductive_labels = th.ones((num_heads, num_tails), dtype=th.float)

        deductive_tuples = self.deductive_closure_tuples[self.deductive_closure_tuples[:, 1] == relation_id]
        
        for head, rel, tail in self.deductive_closure_tuples:
             deductive_labels[head, tail] = 10000
        
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

        # tails = tails.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        # rels = aux_rels.repeat_interleave(len(self.evaluation_heads)).unsqueeze(1)
        # eval_heads = self.evaluation_heads.repeat(num_tails).to(tails.device).unsqueeze(1)
        # logits_tails = model(th.cat([eval_heads, rels, tails], dim=-1), "gci2")
        # logits_tails = logits_tails.view(-1, len(self.evaluation_heads))

        return logits_heads, None

    def evaluate(self, model, mode="test", relations_to_evaluate=None):
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")

        
        # if relations_to_evaluate is None:
            # relations_to_evaluate = range(len(self.dataset.object_properties))
        # elif isinstance(relations_to_evaluate[0], str):
            # relations_to_evaluate = [self.relation_to_id[r] for r in relations_to_evaluate]

        results = dict()

        if mode == "valid":
            eval_tuples = self.valid_tuples
        elif mode == "test":
            eval_tuples = self.test_tuples

        for rel in relations_to_evaluate:
            rel_str = self.id_to_relation[rel]
            logger.debug(f"Evaluating relation {rel_str}. Id: {rel}")
            logger.debug(f"Shape of eval_tuples: {eval_tuples.shape}")
            mask = eval_tuples[:, 1] == rel
            rel_eval_tuples = eval_tuples[mask]
            if len(rel_eval_tuples) == 0:
                logger.debug(f"No evaluation tuples for relation {rel_str}. Skipping...")
                continue
            metrics = self.evaluate_base(model, rel_eval_tuples, mode=mode, relation_id=rel)
            metrics["num_points"] = len(rel_eval_tuples)
            logger.debug(f"Assigned {len(rel_eval_tuples)} points to relation {rel_str}")
            results[rel_str] = metrics

        return results
        # average_metrics = dict()

        # for rel, metrics in results.items():
            # for k, v in metrics.items():
                # if k not in average_metrics:
                    # average_metrics[k] = 0
                # average_metrics[k] += v

        # for k, v in average_metrics.items():
            # average_metrics[k] = v / len(results)

                
        # return average_metrics

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
            
            eval_tuples = th.cat([eval_tuples, deductive_closure_tuples], dim=0)
            # eval_tuples = deductive_closure_tuples
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
                logits_heads, _ = self.get_logits(model, batch, *kwargs)
    
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

                # for i, tail in enumerate(aux_tails):
                    # head = aux_heads[i]
                    # head = th.where(self.evaluation_heads == head)[0].item()
                    # preds = logits_tails[i]

                    # if self.evaluate_with_deductive_closure:
                        # ded_labels = deductive_labels[:, tail].to(preds.device)
                        # ded_labels[head] = 1
                        # preds = preds * ded_labels
                    
                    # order = th.argsort(preds, descending=False)
                    # rank = th.where(order == head)[0].item() + 1
                    # mr += rank
                    # mrr += 1 / rank

                    # if mode == "test":
                        # f_preds = preds * filtering_labels[:, tail].to(preds.device)

                        # if self.evaluate_with_deductive_closure:
                            # ded_labels = deductive_labels[:, tail].to(preds.device)
                            # ded_labels[head] = 1
                            # f_preds = f_preds * ded_labels

                        
                        # f_order = th.argsort(f_preds, descending=False)
                        # f_rank = th.where(f_order == head)[0].item() + 1
                        # fmr += f_rank
                        # fmrr += 1 / f_rank
                    

                    # if mode == "test":
                        # for k in hits_k:
                            # if rank <= int(k):
                                # hits_k[k] += 1

                        # for k in f_hits_k:
                            # if f_rank <= int(k):
                                # f_hits_k[k] += 1

                        # if rank not in ranks:
                            # ranks[rank] = 0
                        # ranks[rank] += 1
                                
                        # if f_rank not in franks:
                            # franks[f_rank] = 0
                        # franks[f_rank] += 1
                                
            mr = mr / (len(eval_tuples))
            mrr = mrr / (len(eval_tuples))

            metrics["mr"] = mr
            metrics["mrr"] = mrr

            if mode == "test":
                fmr = fmr / (len(eval_tuples))
                fmrr = fmrr / (len(eval_tuples))
                auc = compute_rank_roc(ranks, num_tails)
                f_auc = compute_rank_roc(franks, num_tails)

                metrics["f_mr"] = fmr
                metrics["f_mrr"] = fmrr
                metrics["auc"] = auc
                metrics["f_auc"] = f_auc
                
                for k in hits_k:
                    hits_k[k] = hits_k[k] / (len(eval_tuples))
                    metrics[f"hits@{k}"] = hits_k[k]
                    
                for k in f_hits_k:
                    f_hits_k[k] = f_hits_k[k] / (len(eval_tuples))
                    metrics[f"f_hits@{k}"] = f_hits_k[k]

            metrics = {f"{mode}_{k}": v for k, v in metrics.items()}
            return metrics


    def evaluate_by_property(self, model, transitive_properties):
        model.eval()
        # subproperties, superproperties, inverse_properties, transitive_properties = self.get_relation_properties()
        metrics = dict()
        # metrics["subproperties"] = self.evaluate(model, relations_to_evaluate=subproperties, mode="test")
        # metrics["superproperties"] = self.evaluate(model, relations_to_evaluate=superproperties, mode="test")
        # metrics["inverse_properties"] = self.evaluate(model, relations_to_evaluate=inverse_properties, mode="test")
        metrics["transitive_properties"] = self.evaluate(model, relations_to_evaluate=transitive_properties, mode="test")

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



