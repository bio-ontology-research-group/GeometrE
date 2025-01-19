import click as ck
import pickle as pkl
from util import FastTensorDataLoader, seed_everything
from box import Box
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import gc
from itertools import cycle
from dataloader import TrainDatasetOld, TestDataset, SingledirectionalOneShotIterator
from module import ELBEQAModule
from tqdm import tqdm
from evaluators import QAEvaluator
from util import transitive_roles
import wandb
import collections
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

th.multiprocessing.set_sharing_strategy('file_system')

seed_everything(42)

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

all_tasks = list(query_name_dict.keys()) # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

def collate_fn(batch):
    init, pos_tail, neg_tails, subsampling_weights = zip(*batch)

    init = th.vstack(init)
    pos_tail = th.hstack(pos_tail).unsqueeze(1)
    neg_tails = th.vstack(neg_tails)
    subsampling_weights = th.hstack(subsampling_weights).unsqueeze(1)
    logger.debug(f"init: {init.shape}, pos_tail: {pos_tail.shape}, neg_tails: {neg_tails.shape}")
    
    return init, pos_tail, neg_tails, subsampling_weights

def log_to_wandb(metrics, epoch, wandb_logger, prefix):
    prefix = prefix[0]
    for query_type, query_metrics in metrics.items():
        name = f"{prefix}.{query_type}.fmrr"
        wandb_logger.log({f"{name}": query_metrics["f_mrr"]})
             
            
def to_str(metrics, metric_names=["mrr", "f_mrr", "hits@1", "f_hits@1", "hits@3", "f_hits@3", "hits@10", "f_hits@10", "mr", "f_mr", "auc", "f_auc"]):
    string = "| Query\t| MRR\t| F-MRR\t| H1\t| F-H1\t| H3\t| F-H3\t| H10\t| F-H10\t| MR\t| F-MR\t| AUC\t| F-AUC\t|\n"
    for query_type, query_metrics in metrics.items():
        string += f"| {query_type}\t| "
        for metric_name in metric_names:
            metric_value = query_metrics[metric_name]
            if metric_name in ["mr", "f_mr"]:
                metric_value = int(metric_value)
            else:
                metric_value = round(metric_value, 3)
            string += f"{metric_value}\t| "
        string += "\n"
    return string

def group_queries_and_answers_by_task(queries, *answers_list):
    queries_dict = dict()
    all_answers_dict = [dict() for _ in range(len(answers_list))]

    for query_structure in queries:
        short_name = query_name_dict[query_structure]

        queries_dict[short_name] = sorted(list(queries[query_structure]))
        for i, answers in enumerate(answers_list):
            all_answers_dict[i][short_name] = dict()
            for query in queries[query_structure]:
                all_answers_dict[i][short_name][query] = sorted(list(answers[query]))
            
    query_keys = list(queries_dict.keys())
    all_answer_keys = [list(answers_dict.keys()) for answers_dict in all_answers_dict]

    for answer_keys in all_answer_keys:
        assert set(query_keys) == set(answer_keys), f"Queries and answers do not match: {query_keys} != {answer_keys}"
    return queries_dict, *all_answers_dict

def load_data(data_path):
    '''
    Load queries and remove queries not in tasks
    '''
    logger.info("loading data")
    ent2id = pkl.load(open(os.path.join(data_path, "ent2id.pkl"), 'rb'))
    id2ent = pkl.load(open(os.path.join(data_path, "id2ent.pkl"), 'rb'))
    rel2id = pkl.load(open(os.path.join(data_path, "rel2id.pkl"), 'rb'))
    id2rel = pkl.load(open(os.path.join(data_path, "id2rel.pkl"), 'rb'))

    train_queries = pkl.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
    train_answers = pkl.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pkl.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pkl.load(open(os.path.join(data_path, "saturated-valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pkl.load(open(os.path.join(data_path, "saturated-valid-easy-answers.pkl"), 'rb'))
    test_queries = pkl.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pkl.load(open(os.path.join(data_path, "saturated-test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pkl.load(open(os.path.join(data_path, "saturated-test-easy-answers.pkl"), 'rb'))
    
    # remove tasks not in args.tasks

    allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in", "inp", "pin", "pni"]
    allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in"]
    allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in"]
    for name in all_tasks:
        short_name = query_name_dict[name]
        if short_name not in allowed_tasks:
            if name in train_queries:
                del train_queries[name]
            # if name in valid_queries:
                # del valid_queries[name]
            # if name in test_queries:
                # del test_queries[name]
            continue
        logger.info(f"Including {short_name} in tasks")

    # train_queries, train_answers = group_queries_and_answers_by_task(train_queries, train_answers)
    valid_queries, valid_easy_answers, valid_hard_answers = group_queries_and_answers_by_task(valid_queries, valid_easy_answers, valid_hard_answers)
    test_queries, test_easy_answers, test_hard_answers = group_queries_and_answers_by_task(test_queries, test_easy_answers, test_hard_answers)
    
    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, ent2id, id2ent, rel2id, id2rel

# @ck.option('--data-path', '-d', default='../../use_cases/query_answering_data/FB15k-237-betae', help='Path to data directory')
@ck.command()
@ck.option('--data-path', '-d', default='../../use_cases/query_answering_data/WN18RR-QA', help='Path to data directory')
@ck.option('--embed_dim', '-dim', default=400, help='Embedding dimension')
@ck.option('--batch_size', '-bs', default=512, help='Batch size')
@ck.option('--learning_rate', '-lr', default=0.001, help='Learning rate')
@ck.option('--margin', '-m', default=0.01, help='Margin')
@ck.option('--num_negs', '-negs', default=1, help='Number of negative samples')
@ck.option('--device', '-dev', default='cuda', help='Device to use')
@ck.option('--no_sweep', '-ns', is_flag=True, help='Do not use wandb sweep')
@ck.option('--wandb_description', "-desc", default="transEL-QA", help="Description for wandb")
def main(data_path, embed_dim, batch_size, learning_rate, margin, num_negs, device, no_sweep, wandb_description):

    wandb_logger = wandb.init(entity="ferzcam", project="transEL-QA", name=wandb_description)
                     
    if no_sweep:
        wandb_logger.log({"embed_dim": embed_dim,
                          "margin": margin,
                          "learning_rate": learning_rate,
                          "batch_size": batch_size,
                          "num_negs": num_negs,
                          
                          })
    else:
        embed_dim = wandb.config.embed_dim
        margin = wandb.config.margin
        learning_rate = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        num_negs = wandb.config.num_negs
        
    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, ent2id, id2ent, rel2id, id2rel = load_data(data_path)

    dataset_name = data_path.split("/")[-1]

    
    nentity = len(ent2id)
    nrelation = len(rel2id)
    logger.info(f"Number of entities: {nentity}. Number of relations: {nrelation}")

    
    entities_tensor = th.arange(nentity).to(device)
    evaluator = QAEvaluator(entities_tensor, 85, device)

    train_dataloader = DataLoader(TrainDatasetOld(train_queries, nentity, nrelation, num_negs, train_answers),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  collate_fn=TrainDatasetOld.collate_fn
                                  )
    # train_path_iterator = SingledirectionalOneShotIterator(train_dataloader)

    
    # train_dataset_1p = TrainDataset(train_queries['1p'], train_answers['1p'], nentity, num_negs)
    # train_dataset_2p = TrainDataset(train_queries['2p'], train_answers['2p'], nentity, num_negs)
    # train_dataset_3p = TrainDataset(train_queries['3p'], train_answers['3p'], nentity, num_negs)
    # train_dataset_2i = TrainDataset(train_queries['2i'], train_answers['2i'], nentity, num_negs)
    # train_dataset_3i = TrainDataset(train_queries['3i'], train_answers['3i'], nentity, num_negs)
    # train_dataset_2in = TrainDataset(train_queries['2in'], train_answers['2in'], nentity, num_negs)
    # train_dataset_3in = TrainDataset(train_queries['3in'], train_answers['3in'], nentity, num_negs)
    # train_dataset_inp = TrainDataset(train_queries['inp'], train_answers['inp'], nentity, num_negs)
    # train_dataset_pin = TrainDataset(train_queries['pin'], train_answers['pin'], nentity, num_negs)
    # train_dataset_pni = TrainDataset(train_queries['pni'], train_answers['pni'], nentity, num_negs)
    
    # train_dataloader_1p = DataLoader(train_dataset_1p, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    # train_dataloader_2p = cycle(DataLoader(train_dataset_2p, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_3p = cycle(DataLoader(train_dataset_3p, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_2i = cycle(DataLoader(train_dataset_2i, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_3i = cycle(DataLoader(train_dataset_3i, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_2in = cycle(DataLoader(train_dataset_2in, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_3in = cycle(DataLoader(train_dataset_3in, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_inp = cycle(DataLoader(train_dataset_inp, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_pin = cycle(DataLoader(train_dataset_pin, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))
    # train_dataloader_pni = cycle(DataLoader(train_dataset_pni, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8))

    
    # dataloaders = {"2p": train_dataloader_2p, "3p": train_dataloader_3p,
                   # "2i": train_dataloader_2i, "3i": train_dataloader_3i,
                   # "2in": train_dataloader_2in, "3in": train_dataloader_3in,
                   # "inp": train_dataloader_inp, "pin": train_dataloader_pin, "pni": train_dataloader_pni
                   # }

    valid_dataset_1p = TestDataset(valid_queries['1p'], valid_easy_answers['1p'], valid_hard_answers['1p'])
    valid_dataset_2p = TestDataset(valid_queries['2p'], valid_easy_answers['2p'], valid_hard_answers['2p'])
    valid_dataset_3p = TestDataset(valid_queries['3p'], valid_easy_answers['3p'], valid_hard_answers['3p'])
    valid_dataset_2i = TestDataset(valid_queries['2i'], valid_easy_answers['2i'], valid_hard_answers['2i'])
    valid_dataset_3i = TestDataset(valid_queries['3i'], valid_easy_answers['3i'], valid_hard_answers['3i'])
    valid_dataset_2in = TestDataset(valid_queries['2in'], valid_easy_answers['2in'], valid_hard_answers['2in'])
    valid_dataset_3in = TestDataset(valid_queries['3in'], valid_easy_answers['3in'], valid_hard_answers['3in'])
    valid_dataset_pi = TestDataset(valid_queries['pi'], valid_easy_answers['pi'], valid_hard_answers['pi'])
    valid_dataset_ip = TestDataset(valid_queries['ip'], valid_easy_answers['ip'], valid_hard_answers['ip'])
    valid_dataset_inp = TestDataset(valid_queries['inp'], valid_easy_answers['inp'], valid_hard_answers['inp'])
    valid_dataset_pin = TestDataset(valid_queries['pin'], valid_easy_answers['pin'], valid_hard_answers['pin'])
    valid_dataset_pni = TestDataset(valid_queries['pni'], valid_easy_answers['pni'], valid_hard_answers['pni'])
    
    valid_datasets = {"1p": valid_dataset_1p, "2p": valid_dataset_2p, "3p": valid_dataset_3p,
                      "2i": valid_dataset_2i, "3i": valid_dataset_3i,
                      "2in": valid_dataset_2in, "3in": valid_dataset_3in,
                      "pi": valid_dataset_pi, "ip": valid_dataset_ip,
                      "inp": valid_dataset_inp, "pin": valid_dataset_pin, "pni": valid_dataset_pni
                      }

    test_dataset_1p = TestDataset(test_queries['1p'], test_easy_answers['1p'], test_hard_answers['1p'])
    test_dataset_2p = TestDataset(test_queries['2p'], test_easy_answers['2p'], test_hard_answers['2p'])
    test_dataset_3p = TestDataset(test_queries['3p'], test_easy_answers['3p'], test_hard_answers['3p'])
    test_dataset_2i = TestDataset(test_queries['2i'], test_easy_answers['2i'], test_hard_answers['2i'])
    test_dataset_3i = TestDataset(test_queries['3i'], test_easy_answers['3i'], test_hard_answers['3i'])
    test_dataset_2in = TestDataset(test_queries['2in'], test_easy_answers['2in'], test_hard_answers['2in'])
    test_dataset_3in = TestDataset(test_queries['3in'], test_easy_answers['3in'], test_hard_answers['3in'])
    test_dataset_pi = TestDataset(test_queries['pi'], test_easy_answers['pi'], test_hard_answers['pi'])
    test_dataset_ip = TestDataset(test_queries['ip'], test_easy_answers['ip'], test_hard_answers['ip'])
    test_dataset_inp = TestDataset(test_queries['inp'], test_easy_answers['inp'], test_hard_answers['inp'])
    test_dataset_pin = TestDataset(test_queries['pin'], test_easy_answers['pin'], test_hard_answers['pin'])
    test_dataset_pni = TestDataset(test_queries['pni'], test_easy_answers['pni'], test_hard_answers['pni'])

    test_datasets = {"1p": test_dataset_1p, "2p": test_dataset_2p, "3p": test_dataset_3p,
                     "2i": test_dataset_2i, "3i": test_dataset_3i,
                     "2in": test_dataset_2in, "3in": test_dataset_3in,
                     "pi": test_dataset_pi, "ip": test_dataset_ip,
                     "inp": test_dataset_inp, "pin": test_dataset_pin, "pni": test_dataset_pni
                     }
    

    transitive_dataset_roles = transitive_roles[dataset_name]
    logger.info(f"Transitive roles: {transitive_dataset_roles}")
    transitive_ids = th.tensor([rel2id[rel] for rel in transitive_dataset_roles]).to(device)

    
    model = ELBEQAModule(nentity, nrelation, embed_dim=embed_dim, transitive_ids=transitive_ids, margin=margin).to(device)

    optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    criterion = th.nn.MSELoss()
    
    prev_pos_loss = float('inf')
    prev_neg_loss = float('inf')
    curr_pos_loss = prev_pos_loss
    curr_neg_loss = prev_neg_loss
    
    with tqdm(range(1000)) as pbar:
        for epoch in tqdm(range(1000)):
            optimizer.zero_grad()
            
            epoch_loss = 0
            prev_pos_loss = curr_pos_loss
            prev_neg_loss = curr_neg_loss
            curr_pos_loss = 0
            curr_neg_loss = 0


            for batch_data in tqdm(train_dataloader, leave=False):
                loss, pos_loss, neg_loss = 0, 0, 0
                # logger.info(f"num queries: {len(batch_data[0])}")
                batch_queries, pos_tail, neg_tails, subsampling_weights, query_structures = batch_data
                assert pos_tail.shape[0] == neg_tails.shape[0] == subsampling_weights.shape[0], f"Shapes do not match: {pos_tail.shape[0]} != {neg_tails.shape[0]} != {subsampling_weights.shape[0]}"
                
                pos_tail = pos_tail.to(device)
                neg_tails = neg_tails.to(device)
                subsampling_weights = subsampling_weights.to(device)

                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                batch_inits = collections.defaultdict(list)

                for i, query in enumerate(batch_queries):  # group queries with same structure
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)

                for query_structure, queries in batch_queries_dict.items():
                    short_name = query_name_dict[query_structure]
                    queries = th.tensor(queries).to(device)
                    inits = model.get_query_embedding(queries, short_name)
                    batch_inits[query_structure] = inits

                indexed_batch_inits = []
                for query_structure in batch_inits:
                    indexed_batch_inits += zip(batch_idxs_dict[query_structure], batch_inits[query_structure].center, batch_inits[query_structure].offset)
                indexed_batch_inits = sorted(indexed_batch_inits, key=lambda x: x[0])

                idxs, centers, offsets = zip(*indexed_batch_inits)
                centers = th.vstack(centers)
                offsets = th.vstack(offsets)
                query_boxes = Box(centers, offsets)

                tail_center = model.class_embed(pos_tail)
                tail_offset = th.abs(model.class_offset(pos_tail))
                tail_box = Box(tail_center, tail_offset, normalize=True)
                pos_logits = Box.box_inclusion_score(query_boxes, tail_box, margin)
                
                tail_center = model.class_embed(neg_tails)
                tail_offset = th.abs(model.class_offset(neg_tails))
                tail_box = Box(tail_center, tail_offset, normalize=True)
                neg_logits = Box.box_inclusion_score(query_boxes, tail_box, margin)
                
                
                                                
                loss_margin = 20 #30 #10
                pos_loss = - F.logsigmoid(loss_margin - pos_logits) * subsampling_weights
                neg_loss = - F.logsigmoid(neg_logits - loss_margin) * subsampling_weights.unsqueeze(1)

                # pos_loss = pos_loss.mean()
                # neg_loss = neg_loss.mean()
                pos_loss = pos_loss.sum()/subsampling_weights.sum()
                neg_loss = neg_loss.sum()/subsampling_weights.sum()

                loss += pos_loss + neg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                curr_pos_loss += pos_loss.item()
                curr_neg_loss += neg_loss.item()
                epoch_loss += loss.item()

            
            scheduler.step()
            epoch_loss /= len(train_dataloader)
            curr_pos_loss /= len(train_dataloader)
            curr_neg_loss /= len(train_dataloader)
            
            pbar.set_description(f"Prev losses: {prev_pos_loss:.4f}, {prev_neg_loss:.4f}. Curr losses: {curr_pos_loss:.4f}, {curr_neg_loss:.4f}")
            
            if epoch % 10 == 0:
                # validation_metrics = evaluator.evaluate(model, valid_datasets)
                # log_to_wandb(validation_metrics, epoch, wandb_logger, "valid")
                # valid_str = to_str(validation_metrics)
                test_metrics = evaluator.evaluate(model, test_datasets)
                log_to_wandb(test_metrics, epoch, wandb_logger, "test")
                test_str = to_str(test_metrics)

                logger.info(f"Epoch {epoch} Loss: {epoch_loss:6f}")
                # print("\nValidation")
                # print(valid_str)
                print("\nTest")
                print(test_str)

            # print(th.cuda.memory_summary(device='cuda'))
            del loss, pos_loss, neg_loss, pos_logits, neg_logits, query_boxes, tail_box
            gc.collect()
            th.cuda.empty_cache()
            

if __name__ == '__main__':
    main()
        
