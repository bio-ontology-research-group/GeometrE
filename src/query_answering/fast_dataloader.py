#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from multiprocessing import get_context
from functools import partial
from tqdm import tqdm

# from util import list2tuple, tuple2list, 
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class FastTensorWithNegativesDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, nentity, num_negs, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. All tensors must have the same size at dimension 0.
        :param batch_size: batch size to load. Defaults to 32.
        :type batch_size: int, optional
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object. Defaults to False.
        :type shuffle: bool, optional
        """

        # Type checking
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            types = [type(t) for t in tensors]
            raise TypeError(f"All non-optional parameters must be Tensors. Got {types}")
        
        if not isinstance(batch_size, int):
            raise TypeError("Optional parameter batch_size must be of type int")

        if not isinstance(shuffle, bool):
            raise TypeError("Optional parameter shuffle must be of type bool")

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nentity = nentity
        self.num_negs = num_negs

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        negs = torch.LongTensor(np.random.choice(range(self.nentity), size=(len(batch[0]), self.num_negs)))
        batch = batch + (negs,)
        assert all(t.shape[0] == batch[0].shape[0] for t in batch)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

class TestDatasetOld(Dataset):
    def __init__(self, queries, nentity, nrelation):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))
        return negative_sample, flatten(query), query, query_structure
    
    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = [_[1] for _ in data]
        query_unflatten = [_[2] for _ in data]
        query_structure = [_[3] for _ in data]
        return negative_sample, query, query_unflatten, query_structure

def preprocess_queries(queries, answers, count):
    pairs = []

    for query in queries:
        for tail in answers[query]:
            pairs.append((query, tail))
    return pairs

def count_frequency(queries, answer, start=4):
    count = {}
    for query in queries:
        count[query] = start + len(answer[query])
    return count


def construct_train_dataset(queries, answers ):
    logger.info(f"Constructing TrainDataset with {len(queries)} queries.")
    count = count_frequency(queries, answers)
    pairs = preprocess_queries(queries, answers, count)
    

    output_queries = []
    output_tails = []
    output_subsampling_weights = []

    for query, tail in tqdm(pairs):
        subsampling_weight = count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        flattened_query = flatten(query)
        output_queries.append(flattened_query)
        output_tails.append(tail)
        output_subsampling_weights.append(subsampling_weight)

    output_queries = torch.tensor(output_queries, dtype=torch.long)
    output_tails = torch.tensor(output_tails, dtype=torch.long)
    output_subsampling_weights = torch.stack(output_subsampling_weights)
    
    return output_queries, output_tails, output_subsampling_weights
                    
class TrainDataset(Dataset):
    def __init__(self, queries, answers, nentity, negative_sample_size):
        # queries is a list of (query, query_structure) pairs
        self.nentity = nentity
        self.queries = queries
        self.negative_sample_size = negative_sample_size
        self.answers = answers
        self.pairs = self.preprocess_queries()
        self.count = self.count_frequency(queries, answers)
    def preprocess_queries(self):
        pairs = []
        
        for query in self.queries:
            for tail in self.answers[query]:
                pairs.append((query, tail))
        return pairs
        

    def __len__(self):
        return len(self.pairs)
        
    def count_frequency(self, queries, answer, start=4):
        count = {}
        for query in queries:
            count[query] = start + len(answer[query])
        return count

        
    def __getitem__(self, idx):
        query, tail = self.pairs[idx]
        subsampling_weight = self.count[query]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_tail = np.random.choice(range(self.nentity), size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_tail,
                self.answers[query],
                assume_unique=True,
                invert=True
            )
            negative_tail = negative_tail[mask]
            negative_sample_list.append(negative_tail)
            negative_sample_size += negative_tail.size
        negative_tail = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_tail = torch.from_numpy(negative_tail)
            
        flattened_query = torch.tensor(flatten(query), dtype=torch.long)
        tail = torch.tensor(tail, dtype=torch.long)
        return flattened_query, tail, negative_tail, subsampling_weight
        
        return tuple(flattened_query), tail, negative_tail
        data = flattened_query + [tail] + list(negative_tail)
        
        return tuple(data)


    
class TestDataset(Dataset):
    def __init__(self, queries, easy_answers, hard_answers):
        logger.debug(f"Creating TestDataset with {len(queries)} queries.")
        self.queries = queries
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # print(f"Type of queries: {type(self.queries)}. Len of queries: {len(self.queries)}")
        query = self.queries[idx]
        # print(f"Type of query: {type(query)}. Len of query: {len(query)}")
        flattened_query = flatten(query)
        # print(self.easy_answers[query])
        return idx, tuple(flattened_query), set(self.easy_answers[query]), set(self.hard_answers[query])
    
class TestNPDataset(Dataset):
    def __init__(self, queries, easy_answers, hard_answers):
        logger.debug(f"Creating TestNPDataset with {len(queries)} queries.")
        self.queries = queries
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # print(f"Type of queries: {type(self.queries)}. Len of queries: {len(self.queries)}")
        query = self.queries[idx]
        # print(f"Type of query: {type(query)}. Len of query: {len(query)}")
        flattened_query = flatten(query)
        # print(self.easy_answers[query])
        return idx, tuple(flattened_query), set(self.easy_answers[query]), set(self.hard_answers[query])

class TestNIDataset(Dataset):
    def __init__(self, queries, easy_answers, hard_answers):
        logger.debug(f"Creating TestNIDataset with {len(queries)} queries.")
        self.queries = queries
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        flattened_query = flatten(query)
        return idx, tuple(flattened_query), set(self.easy_answers[query]), set(self.hard_answers[query])


    
class TrainDatasetOld(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        # queries is a list of (query, query_structure) pairs
        self.queries = self.flatten_queries(queries)
        self.len = len(self.queries)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(self.queries, answer)
        self.answer = answer
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query] 
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.answer[query], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor([tail])
        return flatten(query), positive_sample, negative_sample, subsampling_weight, query_structure
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure
    
    @staticmethod
    def collate_fn(data):
        query = [_[0] for _ in data]
        positive_sample = torch.cat([_[1] for _ in data], dim=0)
        negative_sample = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        query_structure = [_[4] for _ in data]
        return query, positive_sample, negative_sample, subsample_weight, query_structure
        # return positive_sample, negative_sample, subsample_weight, query, query_structure

    def flatten_queries(self, queries):
        """assign query structure to each sample"""
        all_queries = []
        for query_structure in queries:
            tmp_queries = list(queries[query_structure])
            all_queries.extend([(query, query_structure) for query in tmp_queries])
        return all_queries
        
    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count

class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
