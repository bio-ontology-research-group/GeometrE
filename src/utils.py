import torch as th
import random
import os
import numpy as np
from itertools import chain, combinations, product




def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def pairs(iterable):
    num_items = len(iterable)
    power_set = list(powerset(iterable))
    product_set = list(product(power_set, power_set))

    curated_set = []
    for i1, i2 in product_set:
        if i1 == i2:
            continue
        if len(i1) + len(i2) != num_items:
            continue
        if len(i1) == 0 or len(i1) == num_items:
            continue
        if len(i2) == 0 or len(i2) == num_items:
            continue
        curated_set.append((i1, i2))

    return curated_set

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
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
        if not all(isinstance(t, th.Tensor) for t in tensors):
            raise TypeError("All non-optional parameters must be Tensors")

        if not isinstance(batch_size, int):
            raise TypeError("Optional parameter batch_size must be of type int")

        if not isinstance(shuffle, bool):
            raise TypeError("Optional parameter shuffle must be of type bool")

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def print_as_md(overall_metrics, key=None):
    metrics = ["test_mr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10", "test_hits@50", "test_hits@100", "test_auc"]
    filt_metrics = [k.replace("_", "_f_") for k in metrics]

    string_metrics = "| Property | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | AUC | \n"
    string_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    string_filtered_metrics = "| Property | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | AUC | \n"
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
        
    


    
transitive_roles = {"wn18rr": ["_hypernym", "_has_part"],
                    "fb15k237":["http://www.w3.org/medicine/symptom/symptom_of", "http://www.w3.org/location/location/contains"],
                    "yago310": ["isLocatedIn"]
             }
