#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import random
import pickle
import math
import collections
import itertools
import time
from tqdm import tqdm
import os
import embeddings as E
from box import Box

def Identity(x):
    return x

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma, alpha,
                 geo, test_batch_size=1, box_mode=None,
                 use_cuda=False, query_name_dict=None, beta_mode=None, transitive_ids=None, inverse_ids=None):
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
        
        self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        nn.init.uniform_(
            tensor=self.entity_embedding.weight,
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.offset_embedding = nn.Embedding(nentity, self.entity_dim)
        nn.init.uniform_(
            tensor=self.offset_embedding.weight,
            a=0., 
            b=self.embedding_range.item()
        )

        self.answer_embedding = nn.Embedding(nentity, self.entity_dim)
        nn.init.uniform_(
            tensor=self.answer_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.translation_mul = nn.Embedding(nrelation, self.relation_dim)
        nn.init.uniform_(
            tensor=self.translation_mul.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.translation_add = nn.Embedding(nrelation, self.relation_dim)
        nn.init.uniform_(
            tensor=self.translation_add.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.scaling_mul = nn.Embedding(nrelation, self.relation_dim)
        nn.init.uniform_(
            tensor=self.scaling_mul.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.scaling_add = nn.Embedding(nrelation, self.relation_dim)
        nn.init.uniform_(
            tensor=self.scaling_add.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.inter_translation = nn.Embedding(nrelation, self.relation_dim)
        nn.init.uniform_(
            tensor=self.inter_translation.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=transitive)


    def embedding_1p(self, data):
        return E.embedding_1p(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids)

    def embedding_2p(self, data):
        return E.embedding_2p(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids)

    def embedding_3p(self, data):
        return E.embedding_3p(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids)

    def embedding_2i(self, data):
        return E.embedding_2i(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)

    def embedding_3i(self, data):
        return E.embedding_3i(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)

    def embedding_2in(self, data):
        return E.embedding_2in(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)
    
    def embedding_3in(self, data):
        return E.embedding_3in(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)

    def embedding_pi(self, data):
        return E.embedding_pi(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)

    def embedding_ip(self, data):
        return E.embedding_ip(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)

    def embedding_inp(self, data):
        return E.embedding_inp(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)
         
    def embedding_pin(self, data):
        return E.embedding_pin(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)

    def embedding_pni(self, data):
        return E.embedding_pni(data, self.entity_embedding, self.offset_embedding, self.translation_mul, self.translation_add, self.scaling_mul, self.scaling_add, self.transitive_ids, self.inverse_ids, self.inter_translation)
    
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
    
    
    def embed_query_box(self, queries, query_type):
        '''
        Iterative embed a batch of queries with same structure using Query2box
        queries: a flattened batch of queries
        '''
        
        embedding_fn = self.get_embedding_fn(query_type)
        return embedding_fn(queries)


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

    def cal_logit_box(self, entity_embedding, box_embedding, trans_inv, trans_not_inv, transitive=False, negative=False):
        if transitive:
            logit = Box.box_composed_score(box_embedding, entity_embedding, self.alpha, trans_inv, trans_not_inv, negative=negative)
        else:
            logit = Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha, negative=negative)
        
        return self.gamma - logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        all_boxes, all_idxs, all_corner_logits, all_disjoints, all_total_boxes, all_trans_masks, all_inv_masks = [], [], [], 0, 0, [], []
        all_union_boxes, all_union_idxs, all_union_corner_logits, all_union_disjoints, all_union_total_boxes, all_union_trans_masks, all_union_inv_masks = [], [], [], 0, 0, [], []
        for query_structure in batch_queries_dict:
            query_type = self.query_name_dict[query_structure]
            if 'u' in self.query_name_dict[query_structure]:
                query_type = self.query_name_dict[self.transform_union_structure(query_structure)]
                (boxes, corner_logits, union_disjoints, union_total_boxes), (inv_mask, trans_mask) = self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], query_structure), query_type)
                corner_logits = corner_logits.to(self.entity_embedding.weight.device)
                all_union_boxes.append(boxes)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_corner_logits.append(corner_logits)
                all_union_disjoints += union_disjoints
                all_union_total_boxes += union_total_boxes
                all_union_trans_masks.append(trans_mask)
                all_union_inv_masks.append(inv_mask)
            else:
                (boxes, corner_logits, disjoints, total_boxes), (inv_mask, trans_mask) = self.embed_query_box(batch_queries_dict[query_structure], 
                                             query_type)
                corner_logits = corner_logits.to(self.entity_embedding.weight.device)
                all_boxes.append(boxes)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_corner_logits.append(corner_logits)
                all_disjoints += disjoints
                all_total_boxes += total_boxes
                all_trans_masks.append(trans_mask)
                all_inv_masks.append(inv_mask)

        if len(all_boxes) > 0:
            all_boxes = Box.cat(all_boxes, dim=0)
            all_boxes.center = all_boxes.center.unsqueeze(1)
            all_boxes.offset = all_boxes.offset.unsqueeze(1)
            all_corner_logits = torch.cat(all_corner_logits, dim=0)
            all_trans_masks = torch.cat(all_trans_masks, dim=0)
            all_inv_masks = torch.cat(all_inv_masks, dim=0)
        if len(all_union_boxes) > 0:
            all_union_boxes = Box.cat(all_union_boxes, dim=0)
            all_union_boxes.center = all_union_boxes.center.unsqueeze(1).view(all_union_boxes.center.shape[0]//2, 2, 1, -1)
            all_union_boxes.offset = all_union_boxes.offset.unsqueeze(1).view(all_union_boxes.offset.shape[0]//2, 2, 1, -1)
            all_union_corner_logits = torch.cat(all_union_corner_logits, dim=0)
            all_union_trans_masks = torch.cat(all_union_trans_masks, dim=0)
            all_union_trans_masks = all_union_trans_masks.unsqueeze(1).view(all_union_trans_masks.shape[0]//2, 2)[:, 0]
            all_union_inv_masks = torch.cat(all_union_inv_masks, dim=0)
            all_union_inv_masks = all_union_inv_masks.unsqueeze(1).view(all_union_inv_masks.shape[0]//2, 2)[:, 0]
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_boxes) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_center_embedding = self.answer_embedding(positive_sample_regular).unsqueeze(1)
                # positive_offset_embedding = torch.abs(self.offset_embedding(positive_sample_regular).unsqueeze(1))
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_logit = self.cal_logit_box(positive_box, all_boxes, all_inv_masks, all_trans_masks, transitive=transitive)
                corner_logit = self.gamma - all_corner_logits
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)
                corner_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)

            if len(all_union_boxes) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_center_embedding = self.answer_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                # positive_offset_embedding = torch.abs(self.offset_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1))
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_union_logit = self.cal_logit_box(positive_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, transitive=transitive)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
                corner_union_logit = self.gamma - all_union_corner_logits
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)
                corner_union_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)
                
                
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
            corner_logit = torch.cat([corner_logit, corner_union_logit], dim=0)
        else:
            positive_logit = None
            corner_logit = None

        if type(negative_sample) != type(None):
            if len(all_boxes) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_center_embedding = self.answer_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                # negative_offset_embedding = torch.abs(self.offset_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1))
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_logit = self.cal_logit_box(negative_box, all_boxes, all_inv_masks, all_trans_masks, negative=True, transitive=transitive)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)

            if len(all_union_boxes) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_center_embedding = self.answer_embedding(negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1)
                # negative_offset_embedding = torch.abs(self.offset_embedding(negative_sample_union.view(-1)).view(batch_size, 1, negative_size, -1))
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_union_logit = self.cal_logit_box(negative_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, negative=True, transitive=transitive)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)

            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        # print(all_disjoints, all_total_boxes, all_union_disjoints, all_union_total_boxes)
        return positive_logit, negative_logit, subsampling_weight, all_idxs+all_union_idxs, corner_logit, all_disjoints+all_union_disjoints, all_total_boxes+all_union_total_boxes
    
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

        positive_logit, negative_logit, subsampling_weight, _, corner_logit, all_disjoints, all_total_boxes = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)

        # print(all_disjoints, all_total_boxes)
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        # mask = corner_logit > 0
        # if torch.sum(mask) == 0:
            # corner_loss = torch.zeros(1).to(positive_sample.device)
        # else:
            # corner_loss = corner_logit[mask].mean()
            
        corner_score = -F.logsigmoid(corner_logit) * subsampling_weight
        corner_loss = corner_score.sum() / subsampling_weight.sum()

        # print(corner_loss.item(), positive_sample_loss.item(), negative_sample_loss.item())
        loss = (positive_sample_loss + negative_sample_loss)/2  + corner_loss
        loss.backward()
        optimizer.step()
        
        
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'corner_loss': corner_loss.item(),
            'disjoint': all_disjoints,
            'total_boxes': all_total_boxes
        }
        
        # if corner_loss.item() > 0:
            # log['corner_loss'] = corner_loss.item()
        # if all_disjoints > 0:
            # log['disjoint'] = all_disjoints
            # log['total_boxes'] = all_total_boxes
        # else:
            # log['disjoint'] = 1
            # log['total_boxes'] = 1
        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
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

                _, negative_logit, _, idxs, _, _, _ = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)
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
                    assert len(hard_answer.intersection(easy_answer)) == 0
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
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
