from mowl.projection import TaxonomyWithRelationsProjector, Edge
from utils import FastTensorDataLoader
import torch as th
from tqdm import tqdm
import numpy as np
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import AxiomType as Ax

class Evaluator:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.train_triples = self.create_triples(dataset.ontology)
        self.valid_triples = self.create_triples(dataset.validation)
        self.test_triples = self.create_triples(dataset.testing)

        self.class_to_id = {c: i for i, c in enumerate(self.dataset.classes.as_str)}
        self.id_to_class = {i: c for c, i in self.class_to_id.items()}

        self.relation_to_id = {r: i for i, r in enumerate(self.dataset.object_properties.as_str)}
        self.id_to_relation = {i: r for r, i in self.relation_to_id.items()}

        
    def create_triples(self, ontology):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    
    
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

        
    def create_triples(self, ontology):
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
        


    def get_filtering_labels(self, relation_id, num_entities):

        filtering_triples = th.cat([self.train_triples, self.valid_triples], dim=0)
        filtering_triples = filtering_triples[filtering_triples[:, 1] == relation_id]

        filtering_labels = th.ones((num_entities, num_entities), dtype=th.float)

        for head, rel, tail in filtering_triples:
            filtering_labels[head, tail] = 10000
        
        return filtering_labels
    
    def evaluate_single_relation(self, model, relation_id, num_entities, mode="test"):
        if mode == "valid":
            eval_triples = self.valid_triples
        else:
            eval_triples = self.test_triples

        eval_triples = eval_triples[eval_triples[:, 1] == relation_id]
        if len(eval_triples) == 0:
            return None

        dataloader = FastTensorDataLoader(eval_triples, batch_size=32, shuffle=False)

        metrics = dict()
        mrr, fmrr = 0, 0
        mr, fmr = 0, 0
        ranks, franks = dict(), dict()

        if mode == "test":
            hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
            f_hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        

        filtering_labels = self.get_filtering_labels(relation_id, num_entities)
            
        with th.no_grad():
            for batch, in dataloader:
                batch = batch.to(self.device)
                heads, rels, tails = batch[:, 0], batch[:, 1], batch[:, 2]
                aux_heads = heads.clone()
                heads = heads.repeat_interleave(num_entities).unsqueeze(1)
                rels = rels.repeat_interleave(num_entities).unsqueeze(1)
                eval_tails = th.arange(num_entities, device=rels.device).repeat(len(tails)).unsqueeze(1)
                data = th.cat([heads, rels, eval_tails], dim=-1)
                logits = model(th.cat([heads, rels, eval_tails], dim=-1), "gci2")
                logits = logits.view(-1, num_entities)

                for i, head in enumerate(aux_heads):
                    tail = tails[i]
                    preds = logits[i]
                    order = th.argsort(preds, descending=False)
                    rank = th.where(order == tail)[0].item() + 1
                    mr += rank
                    mrr += 1 / rank

                    if mode == "test":
                        f_preds = preds * filtering_labels[head].to(preds.device)
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

                                
            mr = mr / len(eval_triples)
            mrr = mrr / len(eval_triples)

            metrics["mr"] = mr
            metrics["mrr"] = mrr

            if mode == "test":
                fmr = fmr / len(eval_triples)
                fmrr = fmrr / len(eval_triples)
                auc = compute_rank_roc(ranks, num_entities)
                f_auc = compute_rank_roc(franks, num_entities)

                metrics["fmr"] = fmr
                metrics["fmrr"] = fmrr
                metrics["auc"] = auc
                metrics["f_auc"] = f_auc
                
                for k in hits_k:
                    hits_k[k] = hits_k[k] / len(eval_triples)
                    metrics[f"hits@{k}"] = hits_k[k]
                    
                for k in f_hits_k:
                    f_hits_k[k] = f_hits_k[k] / len(eval_triples)
                    metrics[f"f_hits@{k}"] = f_hits_k[k]

            metrics = {f"{mode}_{k}": v for k, v in metrics.items()}
            return metrics
        

    def evaluate(self, model, relations_to_evaluate=None, mode="test"):
        model.eval()
        if not mode in ["valid", "test"]:
            raise ValueError(f"Mode must be either 'valid' or 'test', not {mode}")


        if relations_to_evaluate is None:
            relations_to_evaluate = range(len(self.dataset.object_properties))
        else:
            relations_to_evaluate = [self.relation_to_id[r] for r in relations_to_evaluate]

        results = dict()

        for rel in relations_to_evaluate:
            rel_str = self.id_to_relation[rel]
            metrics = self.evaluate_single_relation(model, rel, len(self.dataset.classes), mode=mode)
            if metrics is None:
                continue
            results[rel_str] = metrics

        average_metrics = dict()

        for rel, metrics in results.items():
            for k, v in metrics.items():
                if k not in average_metrics:
                    average_metrics[k] = 0
                average_metrics[k] += v

        for k, v in average_metrics.items():
            average_metrics[k] = v / len(results)

                
        return average_metrics


    def evaluate_by_property(self, model):
        model.eval()
        subproperties, superproperties, inverse_properties, transitive_properties = self.get_relation_properties()
        metrics = dict()
        metrics["subproperties"] = self.evaluate(model, relations_to_evaluate=subproperties, mode="test")
        metrics["superproperties"] = self.evaluate(model, relations_to_evaluate=superproperties, mode="test")
        metrics["inverse_properties"] = self.evaluate(model, relations_to_evaluate=inverse_properties, mode="test")
        metrics["transitive_properties"] = self.evaluate(model, relations_to_evaluate=transitive_properties, mode="test")

        print("Subproperties")
        print(metrics["subproperties"])
        print("Superproperties")
        print(metrics["superproperties"])
        print("Inverse properties")
        print(metrics["inverse_properties"])
        print("Transitive properties")
        print(metrics["transitive_properties"])
        
        

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
