#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyArrow
from tqdm import tqdm
import embeddings as E
from box import Box
from util import transitive_roles_dict
from scipy.stats import spearmanr
import numpy as np
from typing import Dict, List, Tuple

from scipy.stats import rankdata

# argsort = torch.argsort(negative_logit, dim=1, descending=True)
def compute_ranks_with_min_tie_breaking(negative_logit):
    assert len(negative_logit.shape) == 2  # (batch_size, num_entities)
    ranks = []
    for i in range(negative_logit.shape[0]):
        # Convert to numpy, compute ranks, convert back
        batch_ranks = rankdata(negative_logit[i].cpu().numpy(), method='min') - 1  # -1 to make 0-indexed
        ranks.append(torch.tensor(batch_ranks, device=negative_logit.device))
    return torch.stack(ranks)

def Identity(x):
    return x

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, alpha,
                 geo, test_batch_size=1, box_mode=None,
                 use_cuda=False, query_name_dict=None, beta_mode=None, transitive_ids=None, inverse_ids=None, with_answer_embedding=False):
        super(KGReasoning, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        
        self.transitive_ids = nn.Parameter(torch.LongTensor(transitive_ids), requires_grad=False)
        self.inverse_ids = nn.Parameter(torch.LongTensor(inverse_ids), requires_grad=False)
        
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        self.alpha = nn.Parameter(
            torch.Tensor([alpha]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        
        
        self.center_embedding = self.init_embedding(nentity, self.entity_dim)
        self.offset_embedding = self.init_embedding(nentity, self.entity_dim)

        self.with_answer_embedding = with_answer_embedding

        if self.with_answer_embedding:
            self.answer_embedding = self.init_embedding(nentity, self.entity_dim)

        self.center_mul = self.init_embedding(nrelation, self.relation_dim)
        self.center_add = self.init_embedding(nrelation, self.relation_dim)
        self.offset_mul = self.init_embedding(nrelation, self.relation_dim)
        self.offset_add = self.init_embedding(nrelation, self.relation_dim)

        self.center_neg_mul = self.init_embedding(nrelation, self.relation_dim)
        self.center_neg_add = self.init_embedding(nrelation, self.relation_dim)
        self.offset_neg_mul = self.init_embedding(nrelation, self.relation_dim)
        self.offset_neg_add = self.init_embedding(nrelation, self.relation_dim)

        
    def init_embedding(self, num_embeddings, dimension):
        embedding = nn.Embedding(num_embeddings, dimension)
        nn.init.uniform_(
            tensor=embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        return embedding
    
    def compute_spearman_and_violations(self, args):
        transitive_roles = transitive_roles_dict["WN18RR-QA"]
        transitive_ids = [0,1,2,3,12,13] #self.transitive_ids.cpu().numpy().tolist()

        spearman_scores = {i: list() for i in transitive_ids}
        violation_counts = {i: list() for i in transitive_ids}

        for id in transitive_ids:
            embeddings = self.center_embedding.weight[:, id].detach().cpu()
            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.split(",") for c in chains], key=lambda x: (len(x), x))
                chains = [c for c in chains if len(c)>2]

            for seq in chains:
                values = [-embeddings[int(e)] for e in seq]
                positions = list(range(len(seq)))

                # Spearman's rho
                rho, _ = spearmanr(positions, values)
                spearman_scores[id].append(rho)

                # Count order violations (for increasing order)
                violations = sum(1 for i in range(len(values)-1) if values[i] > values[i+1])
                violation_counts[id].append(violations)

        avg_spearman = {id: np.mean(scores) for id, scores in spearman_scores.items()}
        avg_violations = {id: np.mean(violations) for id, violations in violation_counts.items()}
        return avg_spearman, avg_violations
                                                    
    def plot_chain_arrows(self, args):
        spearman_scores, violation_counts = self.compute_spearman_and_violations(args)
        print(spearman_scores)
        print(violation_counts)
        transitive_roles = transitive_roles_dict["WN18RR-QA"]
        transitive_ids = [0,1,2,3,12,13] #self.transitive_ids.cpu().numpy().tolist()

        for id in transitive_ids:
            right_color = 'green'
            left_color = 'red'

            embeddings = self.center_embedding.weight[:, id].detach().cpu()

            filename = os.path.join(args.data_path, f"chains_{id}.txt")
            with open(filename) as f:
                chains = f.readlines()
                chains = sorted([c.split(",") for c in chains], key=lambda x: (len(x), x))
                chains = [c for c in chains if len(c)>2]
            max_length = max(len(seq) for seq in chains)

            fig, ax = plt.subplots(figsize=(12, 10))
            for idx, seq in enumerate(chains):
                values = [-embeddings[int(e)] for e in seq]
                y_gap = 1  # Vertical space between sequences
                y = idx * y_gap  # Sequence row

                for i in range(len(values) - 1):
                    x_start = i
                    x_end = i + 1


                    if values[i] <= values[i + 1]:
                        color = right_color  # Order preserved
                        direction = 0.8
                    else:
                        color = left_color  # Order violated
                        direction = -0.8

                    ax.add_patch(FancyArrow(x_start, y, 0.9, 0, width=0.05, color=color))


            # Formatting
            ax.set_ylim(-1, len(chains) * y_gap)
            ax.set_xlim(-2, max_length + 1)
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(range(max_length))
            ax.set_xlabel("Position in Sequence")
            ax.set_title("Order Preservation Visualization (Green: Preserved, Red: Violated)")

            plt.tight_layout()
            outfilename = os.path.join(args.save_path, f"chains_arrows_plot_{id}.png")
            plt.savefig(outfilename, dpi=300)
            plt.close()

    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=transitive)

    def get_box_data(self):
        return self.center_embedding, self.offset_embedding

    def get_role_data(self):
        positive_data = self.center_mul, self.center_add, self.offset_mul, self.offset_add
        negative_data = self.center_neg_mul, self.center_neg_add, self.offset_neg_mul, self.offset_neg_add
        return positive_data, negative_data

            
    def embedding_1p(self, data, transitive):
        return E.embedding_1p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2p(self, data, transitive):
        return E.embedding_2p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_3p(self, data, transitive):
        return E.embedding_3p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2i(self, data, transitive):
        return E.embedding_2i(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_3i(self, data, transitive):
        return E.embedding_3i(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2in(self, data, transitive):
        return E.embedding_2in(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
    
    def embedding_3in(self, data, transitive):
        return E.embedding_3in(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_pi(self, data, transitive):
        return E.embedding_pi(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_ip(self, data, transitive):
        return E.embedding_ip(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_inp(self, data, transitive):
        return E.embedding_inp(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
         
    def embedding_pin(self, data, transitive):
        return E.embedding_pin(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_pni(self, data, transitive):
        return E.embedding_pni(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)
    
    def get_embedding_fn(self, task_name):
        """
        This chooses the corresponding embedding fuction given the name of the task.
        """

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
    
    
    def embed_query_box(self, queries, query_type, transitive):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        
        embedding_fn = self.get_embedding_fn(query_type)
        return embedding_fn(queries, transitive)


    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))

    def cal_membership_logit(self, entity_embedding, box_embedding):
        return Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha)

    def cal_transitive_relation_logit(self, transitive_ids, inverse_ids):
        inverse_mask = torch.isin(transitive_ids, inverse_ids)
        
        cen_mul = self.center_mul(transitive_ids)
        cen_add = self.center_add(transitive_ids)
        off_mul = self.offset_mul(transitive_ids)
        off_add = self.offset_add(transitive_ids)

        cen_mul_loss = torch.linalg.norm(cen_mul - 1, ord=1) + torch.linalg.norm(cen_mul -1, dim=-1, ord=1)
        cen_add_loss = torch.linalg.norm(cen_add, ord=1)
        off_mul_loss = torch.linalg.norm(off_mul - 1, ord=1) + torch.linalg.norm(off_mul -1, dim=-1, ord=1)
        off_add_loss = torch.linalg.norm(off_add, ord=1)

        loss = cen_mul_loss + cen_add_loss + off_mul_loss + off_add_loss
        return loss

        
        projection_dims = torch.arange(len(transitive_ids))
        n, dim = cen_mul.shape
        mask = torch.ones((n, dim), dtype=torch.bool)
        mask[torch.arange(n), projection_dims] = False

        cen_mul_non_trans = cen_mul[mask].reshape(n, dim - 1)
        cen_mul_trans = cen_mul[torch.arange(n), projection_dims]
        cen_add_non_trans = cen_add[mask].reshape(n, dim - 1)
        cen_add_trans = cen_add[torch.arange(n), projection_dims]
        off_mul_non_trans = off_mul[mask].reshape(n, dim - 1)
        off_mul_trans = off_mul[torch.arange(n), projection_dims]
        off_add_non_trans = off_add[mask].reshape(n, dim - 1)
        off_add_trans = off_add[torch.arange(n), projection_dims]

        cen_mul_loss = torch.linalg.norm(cen_mul_non_trans - 1, ord=1) + torch.linalg.norm(cen_mul_trans-1, dim=-1, ord=1)
        cen_add_loss = torch.linalg.norm(cen_add_non_trans, ord=1)
        off_mul_loss = torch.linalg.norm(off_mul_non_trans - 1, ord=1) + torch.linalg.norm(off_mul_trans-1, dim=-1, ord=1)
        off_add_loss = torch.linalg.norm(off_add_non_trans, ord=1)

        loss = cen_mul_loss + cen_add_loss + off_mul_loss + off_add_loss
        return loss

    def cal_logit_box(self, entity_embedding, box_embedding, trans_inv, trans_not_inv, projection_dims, transitive=False, negative=False, negative_box=None, negation_indices=None):
        if transitive:
            logit = Box.box_composed_score_with_projection(box_embedding, entity_embedding, self.alpha, trans_inv, trans_not_inv, projection_dims, negative=negative, transitive=transitive)
        else:
            logit = Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha, negative=negative)

        # Add exclusion score for queries with negative boxes
        if negative_box is not None and negation_indices is not None:
            # Extract entity embeddings and boxes for negation queries only
            negation_entity_embedding = Box(entity_embedding.center[negation_indices], entity_embedding.offset[negation_indices])
            exclusion_logit = Box.box_exclusion_score(negative_box, negation_entity_embedding, self.alpha, negative=negative)
            # Add exclusion scores back to the full logit tensor at the correct indices
            logit[negation_indices] = logit[negation_indices] + exclusion_logit

        return self.gamma - logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        all_boxes, all_idxs, all_trans_masks, all_inv_masks, all_projection_dims, all_negative_boxes = [], [], [], [], [], []
        all_union_boxes, all_union_idxs, all_union_trans_masks, all_union_inv_masks, all_union_projection_dims, all_union_negative_boxes = [], [], [], [], [], []

        # Track the cumulative count of queries to map negative boxes to correct indices
        query_count = 0
        negation_indices = []  # Indices of queries with negation
        negation_boxes_list = []  # Corresponding negative boxes

        for query_structure in batch_queries_dict:
            query_type = self.query_name_dict[query_structure]
            batch_size = len(batch_queries_dict[query_structure])

            if 'u' in self.query_name_dict[query_structure]:
                query_type = self.query_name_dict[self.transform_union_structure(query_structure)]
                boxes, inv_mask, trans_mask, projection_dims, negative_box = self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], query_structure), query_type, transitive)
                all_union_boxes.append(boxes)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_trans_masks.append(trans_mask)
                all_union_inv_masks.append(inv_mask)
                all_union_projection_dims.append(projection_dims)
                all_union_negative_boxes.append(negative_box)
            else:
                boxes, inv_mask, trans_mask, projection_dims, negative_box = self.embed_query_box(batch_queries_dict[query_structure], query_type, transitive)
                all_boxes.append(boxes)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_trans_masks.append(trans_mask)
                all_inv_masks.append(inv_mask)
                all_projection_dims.append(projection_dims)

                # Track negation queries
                if negative_box is not None:
                    for i in range(len(boxes)):
                        negation_indices.append(query_count + i)
                    negation_boxes_list.append(negative_box)

                query_count += len(boxes)

        if len(all_boxes) > 0:
            all_boxes = Box.cat(all_boxes, dim=0)
            all_boxes.center = all_boxes.center.unsqueeze(1)
            all_boxes.offset = all_boxes.offset.unsqueeze(1)
            all_trans_masks = torch.cat(all_trans_masks, dim=0)
            all_inv_masks = torch.cat(all_inv_masks, dim=0)
            all_projection_dims = torch.cat(all_projection_dims, dim=0).long()

            # Concatenate negative boxes
            if len(negation_boxes_list) > 0:
                all_negative_boxes = Box.cat(negation_boxes_list, dim=0)
                all_negative_boxes.center = all_negative_boxes.center.unsqueeze(1)
                all_negative_boxes.offset = all_negative_boxes.offset.unsqueeze(1)
                negation_indices = torch.tensor(negation_indices, device=all_boxes.center.device)
            else:
                all_negative_boxes = None
                negation_indices = None
        if len(all_union_boxes) > 0:
            all_union_boxes = Box.cat(all_union_boxes, dim=0)
            all_union_boxes.center = all_union_boxes.center.unsqueeze(1)
            all_union_boxes.offset = all_union_boxes.offset.unsqueeze(1)
            all_union_trans_masks = torch.cat(all_union_trans_masks, dim=0)
            all_union_inv_masks = torch.cat(all_union_inv_masks, dim=0)
            all_union_projection_dims = torch.cat(all_union_projection_dims, dim=0).long()
            # Union queries don't have negation (for now)
            all_union_negative_boxes = None
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_boxes) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                if self.with_answer_embedding:
                    positive_center_embedding = self.answer_embedding(positive_sample_regular).unsqueeze(1)
                else:
                    positive_center_embedding = self.center_embedding(positive_sample_regular).unsqueeze(1)
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_logit = self.cal_logit_box(positive_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, transitive=transitive, negative_box=all_negative_boxes, negation_indices=negation_indices)
            else:
                positive_logit = torch.Tensor([]).to(self.center_embedding.weight.device)
                
            if len(all_union_boxes) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                if self.with_answer_embedding:
                    positive_center_embedding = self.answer_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                else:
                    positive_center_embedding = self.center_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_union_logit = self.cal_logit_box(positive_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, all_union_projection_dims, transitive=transitive, negative_box=all_union_negative_boxes, negation_indices=None)
                positive_union_logit = positive_union_logit.unsqueeze(1).view(positive_union_logit.shape[0]//2, 2)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.center_embedding.weight.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None
            
        if type(negative_sample) != type(None):
            if len(all_boxes) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                if self.with_answer_embedding:
                    negative_center_embedding = self.answer_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                else:
                    negative_center_embedding = self.center_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_logit = self.cal_logit_box(negative_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, negative=True, transitive=transitive, negative_box=all_negative_boxes, negation_indices=negation_indices)
            else:
                negative_logit = torch.Tensor([]).to(self.center_embedding.weight.device)

            if len(all_union_boxes) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                if self.with_answer_embedding:
                    negative_center_embedding = self.answer_embedding(negative_sample_union.view(-1)).view(batch_size, negative_size, -1).repeat(2, 1, 1)
                else:
                    negative_center_embedding = self.center_embedding(negative_sample_union.view(-1)).view(batch_size, negative_size, -1).repeat(2, 1, 1)
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_union_logit = self.cal_logit_box(negative_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, all_union_projection_dims, negative=True, transitive=transitive, negative_box=all_union_negative_boxes, negation_indices=None)
                negative_union_logit = negative_union_logit.unsqueeze(1).view(negative_union_logit.shape[0]//2, 2, -1)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.center_embedding.weight.device)

            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        all_query_boxes = Box(self.center_embedding.weight, self.offset_embedding.weight)
        if self.with_answer_embedding:
            all_answer_boxes = Box(self.answer_embedding.weight, as_point=True)
        else:
            all_answer_boxes = Box(self.center_embedding.weight, as_point=True)
        membership_logit = self.cal_membership_logit(all_answer_boxes, all_query_boxes)
        transitive_relation_logit = self.cal_transitive_relation_logit(self.transitive_ids, self.inverse_ids)

        return positive_logit, negative_logit, membership_logit, transitive_relation_logit, subsampling_weight, all_idxs+all_union_idxs
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator)
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        # Apply negation query reweighting to compensate for 10:1 imbalance
        # if hasattr(args, 'negation_weight') and args.negation_weight != 1.0:
            # for i, query_structure in enumerate(query_structures):
                # query_name = model.query_name_dict[query_structure]
                # if 'n' in query_name:  # negation query (2in, 3in, inp, pin, pni)
                    # subsampling_weight[i] *= args.negation_weight

        positive_logit, negative_logit, membership_logit, transitive_relation_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        membership_loss = -F.logsigmoid(membership_logit).mean()
        relation_loss = -F.logsigmoid(transitive_relation_logit).mean()

        loss = (positive_sample_loss + negative_sample_loss)/2 + relation_loss # + membership_loss
        loss.backward()
        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'membership_loss': membership_loss.item(),
            'transitive_rel_loss': relation_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, transitive_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, _, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
 
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)

                    if args.filter_deductive_triples:
                        transitive_answer = transitive_answers[query]
                        num_transitive = len(transitive_answer)
                        assert len(hard_answer.intersection(transitive_answer)) == 0
                        assert len(easy_answer.intersection(transitive_answer)) == 0

                    else:
                        transitive_answer = set()
                        num_transitive = 0
                        
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    
                    cur_ranking = ranking[idx, list(easy_answer) + list(transitive_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy + num_transitive
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics




    @staticmethod
    def test_step_with_min(model, easy_answers, hard_answers, transitive_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                _, negative_logit, _, _, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                # argsort = torch.argsort(negative_logit, dim=1, descending=True)
                argsort = compute_ranks_with_min_tie_breaking(negative_logit)
                ranking = argsort.clone().to(torch.float)
                if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else: # otherwise, create a new torch Tensor for batch_entity_range
                    if args.cuda:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1).cuda()
                                                   ) # achieve the ranking of all entities
                    else:
                        ranking = ranking.scatter_(1, 
                                                   argsort, 
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
 
                    hard_answer = hard_answers[query]
                    easy_answer = easy_answers[query]
                    
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)

                    if args.filter_deductive_triples:
                        transitive_answer = transitive_answers[query]
                        num_transitive = len(transitive_answer)
                        assert len(hard_answer.intersection(transitive_answer)) == 0
                        assert len(easy_answer.intersection(transitive_answer)) == 0

                    else:
                        transitive_answer = set()
                        num_transitive = 0
                        
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    
                    cur_ranking = ranking[idx, list(easy_answer) + list(transitive_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy + num_transitive
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy + num_transitive).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics
