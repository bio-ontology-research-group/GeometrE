import numpy as np
import torch as th

from mowl.utils.data import FastTensorDataLoader
from mowl.error import messages as msg
from tqdm import tqdm
import logging
import time
from collections import Counter
from deprecated.sphinx import versionchanged
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'}

anchor_nodes_dict = {'1p': [0],
                '2p': [0],
                '3p': [0],
                '2i': [0, 2],
                '3i': [0, 2, 4],
                'ip': [0, 2],
                'pi': [0, 3],
                '2in': [0, 2],
                '3in': [0, 2, 4],
                'inp': [0, 2],
                'pin': [0, 3],
                'pni': [0, 4],
                '2u-DNF': [0, 2],
                'up-DNF': [0, 2],
                '2u-DM': [0, 3],
                'up-DM': [0, 3]}

anchor_nodes_dict = {k: th.tensor(v) for k, v in anchor_nodes_dict.items()}

def time_profile(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Time taken by {func.__name__}: {end - start}")
        return output
    return wrapper


class BaseRankingEvaluator():
    """
    Base class for ranking evaluation of ontology embedding methods.
    """

    
    def __init__(self, entities, batch_size, device):
        """
        :param heads: The indices of the head entities.
        :type heads: :class:`torch.Tensor`
        :param tails: The indices of the tail entities.
        :type tails: :class:`torch.Tensor`
        :param batch_size: The batch size for evaluation.
        :type batch_size: int
        :param device: The device to use for evaluation.
        :type device: str
        """

        self.batch_size = batch_size
        self.device = device
 
        self.entities = entities.to(self.device)
        sorted_entities = th.sort(entities)[0]
        assert (entities == sorted_entities).all(), "Heads must be sorted."
        
        entity_idx = th.arange(len(entities), dtype=th.long, device=self.device)
        
        if not (entities == entity_idx).all():
            logger.info(f"Head indices are incomplete. This is normal if you are predicting over a subset of entities.")
            max_head = entities.max().item() + 1
            self.mapped_entities = - th.ones(max_head, dtype=th.long, device=self.device)
            self.mapped_entities[entities] = entity_idx
        else:
            self.mapped_entities = entities
            
    def get_scores(self, evaluation_model, batch_init, eval_tails, task_name):
        return evaluation_model(batch_init, eval_tails, task_name, test=True)


    def get_expanded_scores(self, evaluation_model, batch_init, task_name):
        # batch does not contain the tail entities
        num_samples = len(batch_init)
        logger.debug(f"Expanding scores for {num_samples} samples.")
        logger.debug(f"Batch shape: {batch_init.shape}.")
        logger.debug(f"Number of entities: {len(self.entities)}.")
        # batch_init_rep = batch.repeat_interleave(len(self.entities), dim=0)
        # logger.debug(f"Batch init rep shape: {batch_init_rep.shape}.")
        repeated_eval_tails = th.arange(len(self.entities), device=self.device).repeat(num_samples).unsqueeze(1)
        eval_tails = th.arange(len(self.entities), device=self.device).unsqueeze(1)
        logger.debug(f"Eval tails shape: {eval_tails.shape}.")
        # data = th.cat([batch_init_rep, eval_tails], dim=1)
        scores = self.get_scores(evaluation_model, batch_init, eval_tails, task_name)
        scores = scores.view(-1, len(self.entities))

        return scores

    # @time_profile
    @th.no_grad()
    def compute_ranking_metrics(self, evaluation_model, test_queries_dataloader, easy_answers, hard_answers, task_name):
        num_queries = test_queries_dataloader.dataset_len
        logger.debug(f"Evaluating {task_name} on {num_queries} samples.")
        
        evaluation_model.to(self.device)
        evaluation_model.eval()

        num_entities = len(self.entities)
        
        # dataloader = FastTensorDataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        metrics = dict()
        mrr, fmrr = 0, 0
        mr, fmr = 0, 0
        # ranks, franks = dict(), dict()
        ranks, franks = Counter(), Counter()
        
        hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        f_hits_k = dict({"1": 0, "3": 0, "10": 0, "50": 0, "100": 0})
        
        num_samples = 0

        filtered_labels = th.ones(num_entities, device=self.device)
        
        for idxs, batch in tqdm(test_queries_dataloader, leave=False, desc=f"Evaluating {task_name}"):
            batch = batch.to(self.device)
            scores = self.get_expanded_scores(evaluation_model, batch, task_name)

            anchor_nodes = batch[:, anchor_nodes_dict[task_name]]
            
            
            for i, head in enumerate(range(len(batch))):
                idx = idxs[i]
                anchor_node = anchor_nodes[i]
                easy_batch_answers = list(easy_answers[idx])
                hard_batch_answers = list(hard_answers[idx])
                all_answers = easy_batch_answers + hard_batch_answers
                assert set(easy_batch_answers) & set(hard_batch_answers) == set(), "Easy and hard answers must be disjoint."

                num_samples += len(hard_batch_answers)
                
                logger.debug("Number of hard answers: %d", len(hard_batch_answers))
                logger.debug("Preds shape: %s", scores[i].shape)
                preds = scores[i].repeat(len(hard_batch_answers), 1)
                preds[:, anchor_node] = 10000
                logger.debug("Preds shape: %s", preds.shape)
                
                filtered_labels.fill_(1)
                mask = th.tensor(all_answers, device=self.device, dtype=th.long)
                filtered_labels[mask] = 10000
                filtered_labels[anchor_node] = 1
                logger.debug(f"Filtered labels shape: {filtered_labels.shape}.")

                idx_dim_0 = th.arange(len(hard_batch_answers), device=self.device)
                
                all_filtered_labels = filtered_labels.repeat(len(hard_batch_answers), 1)
                logger.debug(f"All filtered labels shape: {all_filtered_labels.shape}.")
                all_filtered_labels[idx_dim_0, hard_batch_answers] = 1

                f_preds = preds * all_filtered_labels

                target_scores = preds[idx_dim_0, hard_batch_answers].unsqueeze(1)
                f_target_scores = f_preds[idx_dim_0, hard_batch_answers].unsqueeze(1)
                logger.debug(f"Target scores shape: {target_scores.shape}.")
                logger.debug(f"Filtered target scores shape: {f_target_scores.shape}.")
                batch_ranks = (preds <= target_scores).sum(dim=1)
                batch_f_ranks = (f_preds <= target_scores).sum(dim=1)
                assert (batch_ranks > 0).all(), "All ranks should be greater than 0."
                assert (batch_f_ranks > 0).all(), "All filtered ranks should be greater than 0."
                
                mr += batch_ranks.sum().item()
                mrr += (1 / batch_ranks).sum().item()

                fmr += batch_f_ranks.sum().item()
                fmrr += (1 / batch_f_ranks).sum().item()

                for k in hits_k:
                    hits_k[k] += (batch_ranks <= int(k)).sum().item()
                    f_hits_k[k] += (batch_f_ranks <= int(k)).sum().item()

                ranks.update(batch_ranks.tolist())
                franks.update(batch_f_ranks.tolist())

                
                # for answer in hard_batch_answers:
                    # num_samples += 1
                    # aux_filtered_labels = filtered_labels.clone()
                    # aux_filtered_labels[answer] = 1
                    # f_preds = preds * aux_filtered_labels

                    # rank = (preds <= preds[answer]).sum().item()
                    
                    ##### order = th.argsort(preds, descending=False)
                    ###### rank = th.where(order == answer)[0].item() + 1
                    # mr += rank
                    # mrr += 1 / rank

                    # f_rank = (f_preds <= f_preds[answer]).sum().item()
                    
                    ###### f_order = th.argsort(f_preds, descending=False)
                    ####### f_rank = th.where(f_order == answer)[0].item() + 1
                    # fmr += f_rank
                    # fmrr += 1 / f_rank

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
                
        divisor = 1

        mr = mr / (divisor * num_samples)
        mrr = mrr / (divisor * num_samples)

        metrics["mr"] = mr
        metrics["mrr"] = mrr

        fmr = fmr / (divisor * num_samples)
        fmrr = fmrr / (divisor * num_samples)

        num_entities_for_auc = num_entities
                    
        auc = compute_rank_roc(ranks, num_entities_for_auc)
        f_auc = compute_rank_roc(franks, num_entities_for_auc)

        metrics["f_mr"] = fmr
        metrics["f_mrr"] = fmrr
        metrics["auc"] = auc
        metrics["f_auc"] = f_auc

        for k in hits_k:
            hits_k[k] = hits_k[k] / (divisor * num_samples)
            metrics[f"hits@{k}"] = hits_k[k]

        for k in f_hits_k:
            f_hits_k[k] = f_hits_k[k] / (divisor * num_samples)
            metrics[f"f_hits@{k}"] = f_hits_k[k]

        return metrics
 
class QAEvaluator(BaseRankingEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def evaluate(self, model, test_queries_datasets):
        metrics = dict()

        query_types = list(test_queries_datasets.keys())

        for query_name, queries_dataset in test_queries_datasets.items():
            data = [queries_dataset[i] for i in range(len(queries_dataset))]
            idxs, queries, easy_answers, hard_answers = zip(*data)
            idxs = th.tensor(idxs, device=self.device)
            queries = th.tensor(queries, device=self.device)
            queries = FastTensorDataLoader(idxs, queries, batch_size=self.batch_size, shuffle=False)
            metrics[query_name] = self.compute_ranking_metrics(model, queries, easy_answers, hard_answers, query_name)
        return metrics
            
        
        
def compute_rank_roc(ranks, num_entities, method="riemann"):
    if method == "riemann":
        fn = riemann_sum
    elif method == "trapz":
        fn = np.trapz
    else:
        raise ValueError(f"Method {method} not recognized.")
    
    num_entities = int(num_entities)
    ranks = {k-1: v for k, v in ranks.items()}
    min_rank = min(ranks.keys())
    assert min_rank >= 0
    
    all_ranks = {k: 0 for k in range(min_rank, num_entities)}
    all_ranks.update(ranks)
    
    ranks = all_ranks

    auc_x = list(ranks.keys())
    auc_x.sort()
    
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
 
    auc = fn(auc_y, auc_x) / (num_entities - 1)
    return auc

def riemann_sum(y, x):
    dx = np.diff(x)
    heights = y[:-1]  # Use left endpoints for rectangle heights
    integral = np.sum(heights * dx)
    return integral
