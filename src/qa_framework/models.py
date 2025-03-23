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
        
        self.entity_embedding = self.init_embeddings(nentity, self.entity_dim)

        self.offset_embedding = self.init_embeddings(nentity, self.entity_dim)

        self.answer_embedding = self.init_embeddings(nentity, self.entity_dim)

        self.center_mul = self.init_embeddings(nrelation, self.relation_dim)
        self.center_add = self.init_embeddings(nrelation, self.relation_dim)
        self.offset_mul = self.init_embeddings(nrelation, self.relation_dim)
        self.offset_add = self.init_embeddings(nrelation, self.relation_dim)

        self.inter_center_mul = self.init_embeddings(nrelation, self.relation_dim)
        self.inter_center_add = self.init_embeddings(nrelation, self.relation_dim)
        self.inter_offset_mul = self.init_embeddings(nrelation, self.relation_dim)
        self.inter_offset_add = self.init_embeddings(nrelation, self.relation_dim)

    def init_embeddings(self, num_embeddings, dim, init="default"):
        embedding = nn.Embedding(num_embeddings, dim)
        if init == "default":
            nn.init.uniform_(
                tensor=embedding.weight,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
        elif init == "ones":
            nn.init.ones_(
                tensor=embedding.weight
            )
        elif init == "zeros":
                nn.init.zeros_(
                        tensor=embedding.weight
                )
        else:
            raise ValueError("Unknown init method")
        return embedding
        
    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        return self.forward_box(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=transitive)

    def get_box_data(self):
        return self.entity_embedding, self.offset_embedding

    def get_role_data(self):
        return self.center_mul, self.center_add, self.offset_mul, self.offset_add

    def get_inter_data(self):
        return self.inter_center_mul, self.inter_center_add, self.inter_offset_mul, self.inter_offset_add
    
    def embedding_1p(self, data, transitive):
        return E.embedding_1p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2p(self, data, transitive):
        return E.embedding_2p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_3p(self, data, transitive):
        return E.embedding_3p(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive)

    def embedding_2i(self, data, transitive):
        return E.embedding_2i(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())

    def embedding_3i(self, data, transitive):
        return E.embedding_3i(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())

    def embedding_2in(self, data, transitive):
        return E.embedding_2in(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())
    
    def embedding_3in(self, data, transitive):
        return E.embedding_3in(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())

    def embedding_pi(self, data, transitive):
        return E.embedding_pi(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())

    def embedding_ip(self, data, transitive):
        return E.embedding_ip(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())

    def embedding_inp(self, data, transitive):
        return E.embedding_inp(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())
         
    def embedding_pin(self, data, transitive):
        return E.embedding_pin(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())

    def embedding_pni(self, data, transitive):
        return E.embedding_pni(data, self.get_box_data(), self.get_role_data(), self.transitive_ids, self.inverse_ids, transitive, self.get_inter_data())
    
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
        
    def cal_logit_box(self, entity_embedding, box_embedding, trans_inv, trans_not_inv, projection_dims, transitive=False, negative=False):
        if transitive:
            # logit = Box.box_composed_score(box_embedding, entity_embedding, self.alpha, trans_inv, trans_not_inv, negative=negative)
            logit = Box.box_composed_score_with_projection(box_embedding, entity_embedding, self.alpha, trans_inv, trans_not_inv, projection_dims, negative=negative)
        else:
            logit = Box.box_inclusion_score(box_embedding, entity_embedding, self.alpha, negative=negative)
        
        return self.gamma - logit

    def forward_box(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=False):
        all_boxes, all_idxs, all_trans_masks, all_inv_masks, all_projection_dims = [], [], [], [], []
        all_union_boxes, all_union_idxs, all_union_trans_masks, all_union_inv_masks, all_union_projection_dims = [], [], [], [], []
        for query_structure in batch_queries_dict:
            query_type = self.query_name_dict[query_structure]
            if 'u' in self.query_name_dict[query_structure]:
                query_type = self.query_name_dict[self.transform_union_structure(query_structure)]
                boxes, inv_mask, trans_mask, projection_dims = self.embed_query_box(self.transform_union_query(batch_queries_dict[query_structure], query_structure), query_type, transitive)
                all_union_boxes.append(boxes)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
                all_union_trans_masks.append(trans_mask)
                all_union_inv_masks.append(inv_mask)
                all_union_projection_dims.append(projection_dims)
            else:
                boxes, inv_mask, trans_mask, projection_dims = self.embed_query_box(batch_queries_dict[query_structure], query_type, transitive)
                all_boxes.append(boxes)
                all_idxs.extend(batch_idxs_dict[query_structure])
                all_trans_masks.append(trans_mask)
                all_inv_masks.append(inv_mask)
                all_projection_dims.append(projection_dims)

        if len(all_boxes) > 0:
            all_boxes = Box.cat(all_boxes, dim=0)
            all_boxes.center = all_boxes.center.unsqueeze(1)
            all_boxes.offset = all_boxes.offset.unsqueeze(1)
            all_trans_masks = torch.cat(all_trans_masks, dim=0)
            all_inv_masks = torch.cat(all_inv_masks, dim=0)
            all_projection_dims = torch.cat(all_projection_dims, dim=0).long()
        if len(all_union_boxes) > 0:
            all_union_boxes = Box.cat(all_union_boxes, dim=0)
            all_union_boxes.center = all_union_boxes.center.unsqueeze(1)
            all_union_boxes.offset = all_union_boxes.offset.unsqueeze(1)
            all_union_trans_masks = torch.cat(all_union_trans_masks, dim=0)
            all_union_inv_masks = torch.cat(all_union_inv_masks, dim=0)
            all_union_projection_dims = torch.cat(all_union_projection_dims, dim=0).long()
        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs+all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_boxes) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_center_embedding = self.answer_embedding(positive_sample_regular).unsqueeze(1)
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_logit = self.cal_logit_box(positive_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, transitive=transitive)
            else:
                positive_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)
                
            if len(all_union_boxes) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_center_embedding = self.answer_embedding(positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_box = Box(positive_center_embedding, as_point=True)
                positive_union_logit = self.cal_logit_box(positive_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, all_union_projection_dims, transitive=transitive)
                positive_union_logit = positive_union_logit.unsqueeze(1).view(positive_union_logit.shape[0]//2, 2)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None
            
        if type(negative_sample) != type(None):
            if len(all_boxes) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_center_embedding = self.answer_embedding(negative_sample_regular.view(-1)).view(batch_size, negative_size, -1)
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_logit = self.cal_logit_box(negative_box, all_boxes, all_inv_masks, all_trans_masks, all_projection_dims, negative=True, transitive=transitive)
            else:
                negative_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)

            if len(all_union_boxes) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                negative_center_embedding = self.answer_embedding(negative_sample_union.view(-1)).view(batch_size, negative_size, -1).repeat(2, 1, 1)
                negative_box = Box(negative_center_embedding, as_point=True)
                negative_union_logit = self.cal_logit_box(negative_box, all_union_boxes, all_union_inv_masks, all_union_trans_masks, all_union_projection_dims, negative=True, transitive=transitive)
                negative_union_logit = negative_union_logit.unsqueeze(1).view(negative_union_logit.shape[0]//2, 2, -1)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.entity_embedding.weight.device)

            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        all_query_boxes = Box(self.entity_embedding.weight, self.offset_embedding.weight)
        all_answer_boxes = Box(self.answer_embedding.weight, as_point=True)
        membership_logit = self.cal_membership_logit(all_answer_boxes, all_query_boxes)
        
        return positive_logit, negative_logit, membership_logit, subsampling_weight, all_idxs+all_union_idxs
    
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

        positive_logit, negative_logit, membership_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)

        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        membership_loss = -F.logsigmoid(membership_logit).mean()

        # lambda_reg = 0.1
        # reg = lambda_reg * (((model.center_mul.weight - 1.0) ** 2).mean() + ((model.center_add.weight) ** 2).mean() + ((model.offset_mul.weight - 1.0) ** 2).mean() + ((model.offset_add.weight) ** 2).mean())
        
        loss = (positive_sample_loss + negative_sample_loss)/2 + membership_loss
        loss.backward()
        optimizer.step()
        
        
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'membership_loss': membership_loss.item(),
            'loss': loss.item()
        }
        
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

                _, negative_logit, _, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict, transitive=args.transitive)
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
