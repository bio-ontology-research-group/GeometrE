import click as ck
import pickle as pkl
from util import FastTensorDataLoader, seed_everything
from box import Box
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from itertools import cycle
from fast_dataloader import construct_train_dataset, TestDataset, FastTensorWithNegativesDataLoader
from module import ELBEQAModule
from tqdm import tqdm
from evaluators import QAEvaluator
from util import transitive_roles, inverse_roles
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

def group_queries_and_answers_by_task(queries, *answers_list, transitive_only=False):
    logger.info(f"Grouping queries and answers by task. Transitive only: {transitive_only}")
    queries_dict = dict()
    all_answers_dict = [dict() for _ in range(len(answers_list))]

    transitive_tasks = ["1p", "2p", "3p", "ip", "inp"]
    for query_structure in queries:
        
        short_name = query_name_dict[query_structure]
        if transitive_only and not short_name in transitive_tasks:
            continue
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

def load_data(data_path, saturated_data):
    '''
    Load queries and remove queries not in tasks
    '''
    logger.info(f"Loading data. Saturated: {saturated_data}")
    ent2id = pkl.load(open(os.path.join(data_path, "ent2id.pkl"), 'rb'))
    id2ent = pkl.load(open(os.path.join(data_path, "id2ent.pkl"), 'rb'))
    rel2id = pkl.load(open(os.path.join(data_path, "rel2id.pkl"), 'rb'))
    id2rel = pkl.load(open(os.path.join(data_path, "id2rel.pkl"), 'rb'))

    train_queries = pkl.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
    train_answers = pkl.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
    
    
    if saturated_data == "yes":
        valid_queries = pkl.load(open(os.path.join(data_path, "saturated-valid-queries.pkl"), 'rb'))
        valid_hard_answers = pkl.load(open(os.path.join(data_path, "saturated-valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pkl.load(open(os.path.join(data_path, "saturated-valid-easy-answers.pkl"), 'rb'))
        test_queries = pkl.load(open(os.path.join(data_path, "saturated-test-queries.pkl"), 'rb'))
        test_hard_answers = pkl.load(open(os.path.join(data_path, "saturated-test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pkl.load(open(os.path.join(data_path, "saturated-test-easy-answers.pkl"), 'rb'))

    elif saturated_data == "no":
        valid_queries = pkl.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pkl.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pkl.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pkl.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pkl.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pkl.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))
    else:
        raise ValueError("Saturated data must be either 'yes' or 'no'")
    # remove tasks not in args.tasks

    allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in", "inp", "pin", "pni"]
    # allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in"]
    # allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in"]
    # allowed_tasks = ["1p", "2i", "2p", "3p", "3i", "inp", "pin", "pni"]
    allowed_tasks = ["1p", "2p", "3p", "2i", "3i", "inp", "pin", "pni", "2in", "3in"]
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

    train_queries, train_answers = group_queries_and_answers_by_task(train_queries, train_answers)
    transitive_only = False
    if saturated_data == "yes":
        transitive_only = True
    valid_queries, valid_easy_answers, valid_hard_answers = group_queries_and_answers_by_task(valid_queries, valid_easy_answers, valid_hard_answers, transitive_only=transitive_only)
    test_queries, test_easy_answers, test_hard_answers = group_queries_and_answers_by_task(test_queries, test_easy_answers, test_hard_answers, transitive_only=transitive_only)
    
    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, ent2id, id2ent, rel2id, id2rel

# @ck.option('--data-path', '-d', default='../../use_cases/query_answering_data/FB15k-237-betae', help='Path to data directory')
@ck.command()
@ck.option('--data-path', '-d', default='../../use_cases/query_answering_data/WN18RR-QA', help='Path to data directory')
@ck.option('--embed_dim', '-dim', default=400, help='Embedding dimension')
@ck.option('--batch_size', '-bs', default=512, help='Batch size')
@ck.option('--learning_rate', '-lr', default=0.001, help='Learning rate')
@ck.option('--loss_margin', '-m', default=0.01, help='Loss Margin')
@ck.option('--module_margin', '-mm', default=0.01, help='Module Margin')
@ck.option('--num_negs', '-negs', default=1, help='Number of negative samples')
@ck.option('--device', '-dev', default='cuda', help='Device to use')
@ck.option('--no_sweep', '-ns', is_flag=True, help='Do not use wandb sweep')
@ck.option('--wandb_description', "-desc", default="transEL-QA", help="Description for wandb")
@ck.option('--transitive', '-t', type=ck.Choice(['yes', 'no']), help='Use transitive relations')
@ck.option('--saturated', '-sat', type=ck.Choice(['yes', 'no']), help='Use saturated data')
@ck.option('--only_test', '-ot', is_flag=True, help='Only test')
def main(data_path, embed_dim, batch_size, learning_rate, loss_margin, module_margin, num_negs, device, no_sweep, wandb_description, transitive, saturated, only_test):

    wandb_logger = wandb.init(entity="ferzcam", project="transEL-QA", name=wandb_description)
                     
    if no_sweep:
        wandb_logger.log({"embed_dim": embed_dim,
                          "loss_margin": loss_margin,
                          "module_margin": module_margin,
                          "learning_rate": learning_rate,
                          "batch_size": batch_size,
                          "num_negs": num_negs,
                          "transitive": transitive,
                          "saturated": saturated
                          })
    else:
        embed_dim = wandb.config.embed_dim
        loss_margin = wandb.config.loss_margin
        module_margin = wandb.config.module_margin
        learning_rate = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        num_negs = wandb.config.num_negs
        # transitive = wandb.config.transitive
        # saturated = wandb.config.saturated
        transitive = "no"
        saturated = "no"
        
    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers, ent2id, id2ent, rel2id, id2rel = load_data(data_path, saturated)

    dataset_name = data_path.split("/")[-1]

    
    nentity = len(ent2id)
    nrelation = len(rel2id)
    logger.info(f"Number of entities: {nentity}. Number of relations: {nrelation}")

    
    entities_tensor = th.arange(nentity).to(device)
    evaluator = QAEvaluator(entities_tensor, 70, device)

    train_dataset_1p = construct_train_dataset(train_queries['1p'], train_answers['1p'])
    train_dataset_2p = construct_train_dataset(train_queries['2p'], train_answers['2p'])
    train_dataset_3p = construct_train_dataset(train_queries['3p'], train_answers['3p'])
    train_dataset_2i = construct_train_dataset(train_queries['2i'], train_answers['2i'])
    train_dataset_3i = construct_train_dataset(train_queries['3i'], train_answers['3i'])
    train_dataset_2in = construct_train_dataset(train_queries['2in'], train_answers['2in'])
    train_dataset_3in = construct_train_dataset(train_queries['3in'], train_answers['3in'])
    train_dataset_inp = construct_train_dataset(train_queries['inp'], train_answers['inp'])
    train_dataset_pin = construct_train_dataset(train_queries['pin'], train_answers['pin'])
    train_dataset_pni = construct_train_dataset(train_queries['pni'], train_answers['pni'])

    dataset_sizes = {"1p": train_dataset_1p[0].shape[0], "2p": train_dataset_2p[0].shape[0], "3p": train_dataset_3p[0].shape[0],
                     "2i": train_dataset_2i[0].shape[0], "3i": train_dataset_3i[0].shape[0],
                     "2in": train_dataset_2in[0].shape[0], "3in": train_dataset_3in[0].shape[0],
                     "inp": train_dataset_inp[0].shape[0], "pin": train_dataset_pin[0].shape[0], "pni": train_dataset_pni[0].shape[0]
                     }
            
    logger.info(f"Dataset sizes: {dataset_sizes}")

    
    train_dataloader_1p = FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_1p, batch_size=batch_size, shuffle=True)
    train_dataloader_2p = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_2p, batch_size=batch_size, shuffle=True))
    train_dataloader_3p = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_3p, batch_size=batch_size, shuffle=True))
    train_dataloader_2i = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_2i, batch_size=batch_size, shuffle=True))
    train_dataloader_3i = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_3i, batch_size=batch_size, shuffle=True))
    train_dataloader_2in = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_2in, batch_size=batch_size, shuffle=True))
    train_dataloader_3in = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_3in, batch_size=batch_size, shuffle=True))
    train_dataloader_inp = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_inp, batch_size=batch_size, shuffle=True))
    train_dataloader_pin = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_pin, batch_size=batch_size, shuffle=True))
    train_dataloader_pni = cycle(FastTensorWithNegativesDataLoader(nentity, num_negs, *train_dataset_pni, batch_size=batch_size, shuffle=True))


    dataloaders = dict()
    dataloaders = {
        "2p": train_dataloader_2p, "3p": train_dataloader_3p,
        "2i": train_dataloader_2i, "3i": train_dataloader_3i,
        "2in": train_dataloader_2in, "3in": train_dataloader_3in,
        "inp": train_dataloader_inp, "pin": train_dataloader_pin, "pni": train_dataloader_pni
                   }

    data_loader_sizes = {"1p": train_dataset_1p[0].shape[0], "2p": train_dataset_2p[0].shape[0], "3p": train_dataset_3p[0].shape[0],
                         "2i": train_dataset_2i[0].shape[0], "3i": train_dataset_3i[0].shape[0],
                         "2in": train_dataset_2in[0].shape[0], "3in": train_dataset_3in[0].shape[0],
                         "inp": train_dataset_inp[0].shape[0], "pin": train_dataset_pin[0].shape[0], "pni": train_dataset_pni[0].shape[0]
                         }
    total_count = sum(data_loader_sizes.values())
    train_data_weights = {key: value/total_count for key, value in data_loader_sizes.items()}

    # valid_dataset_1p = TestDataset(valid_queries['1p'], valid_easy_answers['1p'], valid_hard_answers['1p'])
    # valid_dataset_2p = TestDataset(valid_queries['2p'], valid_easy_answers['2p'], valid_hard_answers['2p'])
    # valid_dataset_3p = TestDataset(valid_queries['3p'], valid_easy_answers['3p'], valid_hard_answers['3p'])
    # valid_dataset_2i = TestDataset(valid_queries['2i'], valid_easy_answers['2i'], valid_hard_answers['2i'])
    # valid_dataset_3i = TestDataset(valid_queries['3i'], valid_easy_answers['3i'], valid_hard_answers['3i'])
    # valid_dataset_2in = TestDataset(valid_queries['2in'], valid_easy_answers['2in'], valid_hard_answers['2in'])
    # valid_dataset_3in = TestDataset(valid_queries['3in'], valid_easy_answers['3in'], valid_hard_answers['3in'])
    # valid_dataset_pi = TestDataset(valid_queries['pi'], valid_easy_answers['pi'], valid_hard_answers['pi'])
    # valid_dataset_ip = TestDataset(valid_queries['ip'], valid_easy_answers['ip'], valid_hard_answers['ip'])
    # valid_dataset_inp = TestDataset(valid_queries['inp'], valid_easy_answers['inp'], valid_hard_answers['inp'])
    # valid_dataset_pin = TestDataset(valid_queries['pin'], valid_easy_answers['pin'], valid_hard_answers['pin'])
    # valid_dataset_pni = TestDataset(valid_queries['pni'], valid_easy_answers['pni'], valid_hard_answers['pni'])
    
    # valid_datasets = {"1p": valid_dataset_1p#, "2p": valid_dataset_2p, "3p": valid_dataset_3p,
                      # "2i": valid_dataset_2i, "3i": valid_dataset_3i,
                      # "2in": valid_dataset_2in, "3in": valid_dataset_3in,
                      # "pi": valid_dataset_pi, "ip": valid_dataset_ip,
                      # "inp": valid_dataset_inp, "pin": valid_dataset_pin, "pni": valid_dataset_pni
                      # }

    test_dataset_1p = TestDataset(test_queries['1p'], test_easy_answers['1p'], test_hard_answers['1p'])
    test_dataset_2p = TestDataset(test_queries['2p'], test_easy_answers['2p'], test_hard_answers['2p'])
    test_dataset_3p = TestDataset(test_queries['3p'], test_easy_answers['3p'], test_hard_answers['3p'])
    test_dataset_ip = TestDataset(test_queries['ip'], test_easy_answers['ip'], test_hard_answers['ip'])
    test_dataset_inp = TestDataset(test_queries['inp'], test_easy_answers['inp'], test_hard_answers['inp'])
    test_datasets = {"1p": test_dataset_1p, "2p": test_dataset_2p, "3p": test_dataset_3p,
                     "ip": test_dataset_ip,
                     "inp": test_dataset_inp
                     }

    if saturated == "no":
        test_dataset_2i = TestDataset(test_queries['2i'], test_easy_answers['2i'], test_hard_answers['2i'])
        test_dataset_3i = TestDataset(test_queries['3i'], test_easy_answers['3i'], test_hard_answers['3i'])
        test_dataset_2in = TestDataset(test_queries['2in'], test_easy_answers['2in'], test_hard_answers['2in'])
        test_dataset_3in = TestDataset(test_queries['3in'], test_easy_answers['3in'], test_hard_answers['3in'])
        test_dataset_pi = TestDataset(test_queries['pi'], test_easy_answers['pi'], test_hard_answers['pi'])
        test_dataset_pin = TestDataset(test_queries['pin'], test_easy_answers['pin'], test_hard_answers['pin'])
        test_dataset_pni = TestDataset(test_queries['pni'], test_easy_answers['pni'], test_hard_answers['pni'])
        
        test_datasets["2i"] = test_dataset_2i
        test_datasets["3i"] = test_dataset_3i
        test_datasets["2in"] = test_dataset_2in
        test_datasets["3in"] = test_dataset_3in
        test_datasets["pi"] = test_dataset_pi
        test_datasets["pin"] = test_dataset_pin
        test_datasets["pni"] = test_dataset_pni
    

    transitive_dataset_roles = transitive_roles[dataset_name]
    inverse_dataset_roles = inverse_roles[dataset_name]
    logger.info(f"Transitive roles: {transitive_dataset_roles}")
    if transitive == "yes":
        transitive_ids = th.tensor([rel2id[rel] for rel in transitive_dataset_roles]).to(device)
        inverse_ids = th.tensor([rel2id[rel] for rel in inverse_dataset_roles]).to(device)
    elif transitive == "no":
        transitive_ids = None
        inverse_ids = None
    else:
        raise ValueError("Transitive must be either 'yes' or 'no'")
    
    model = ELBEQAModule(nentity, nrelation, embed_dim=embed_dim, transitive_ids=transitive_ids, inverse_ids=inverse_ids, gamma=loss_margin).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    criterion = th.nn.MSELoss()
    
    prev_pos_loss = float('inf')
    prev_neg_loss = float('inf')
    curr_pos_loss = prev_pos_loss
    curr_neg_loss = prev_neg_loss

    
    step = 0
    num_warmup_steps = 1000
    initial_lr = learning_rate
    
    with tqdm(range(101)) as pbar:
        for epoch in tqdm(range(1000)):
            epoch_loss = 0
            epoch_trans_score = 0
            epoch_non_trans_score = 0
            prev_pos_loss = curr_pos_loss
            prev_neg_loss = curr_neg_loss
            curr_pos_loss = 0
            curr_neg_loss = 0
            
            test_evaluation_part = False

            if test_evaluation_part:
                if epoch % 10 == 0:
                    # validation_metrics = evaluator.evaluate(model, valid_dataloaders, valid_easy_answers, valid_hard_answers)
                    validation_metrics = evaluator.evaluate(model, valid_datasets)
                    valid_str = to_str(validation_metrics)
                    test_metrics = evaluator.evaluate(model, test_datasets)
                    test_str = to_str(test_metrics)

                    logger.info(f"Epoch {epoch} Loss: {epoch_loss:6f}")
                    print("\nValidation")
                    print(valid_str)
                    print("\nTest")
                    print(test_str)


            for batch_data in tqdm(train_dataloader_1p, leave=False):

                # if step < num_warmup_steps:
                    # step += 1
                    # lr = initial_lr * (step / num_warmup_steps)
                    # for param_group in optimizer.param_groups:
                        # param_group['lr'] = lr

                
                init, pos_tail, subsampling_weights, neg_tails = batch_data
                
                init = init.to(device)
                pos_tail = pos_tail.to(device)
                neg_tails = neg_tails.to(device)
                neg_tails = th.randint(0, nentity, neg_tails.shape, device=device)
                subsampling_weights = subsampling_weights.to(device)


                loss = 0 

                logger.debug(f"\n\n\nEntering pos logits")
                pos_logits = model(init, pos_tail, "1p").unsqueeze(1)
                logger.debug(f"\n\n\nEntering neg logits")
                neg_logits = model(init, neg_tails, "1p")

                # loss_margin = 20 #0.01 #20 #10
                old = False
                if old:
                    loss_1p = - F.logsigmoid(neg_logits - pos_logits - loss_margin) * subsampling_weights
                    loss += loss_1p.sum()/subsampling_weights.sum()
                else:
                    negative_score = F.logsigmoid(-neg_logits).mean(dim=1)
                    positive_score = F.logsigmoid(pos_logits).squeeze(dim=1)
                    positive_sample_loss = - (subsampling_weights * positive_score).sum()
                    negative_sample_loss = - (subsampling_weights * negative_score).sum()
                    positive_sample_loss /= subsampling_weights.sum()
                    negative_sample_loss /= subsampling_weights.sum()

                    loss_1p = (positive_sample_loss + negative_sample_loss)/2
                    loss += loss_1p
                    
                
                
                
                
                # pos_loss = - F.logsigmoid(loss_margin - pos_logits) * subsampling_weights
                # logger.debug(f"Pos logits: {pos_logits.shape} - subsampling_weights: {subsampling_weights.shape}")
                # logger.debug(f"Neg logits: {neg_logits.shape} - subsampling_weights: {subsampling_weights.shape}")
                # neg_loss = - F.logsigmoid(neg_logits - loss_margin) * subsampling_weights#.unsqueeze(1)
                # logger.debug(f"Pos loss: {pos_loss.shape} - Neg loss: {neg_loss.shape}")
                # pos_loss = pos_loss.sum()/subsampling_weights.sum()
                # neg_loss = neg_loss.sum()/subsampling_weights.sum()
                # curr_pos_loss += pos_loss.item()
                # curr_neg_loss += neg_loss.item()
                # loss += pos_loss + neg_loss

                for task_name, dataloader in dataloaders.items():

                    init, pos_tail, subsampling_weights, neg_tails = next(dataloader)
                    init = init.to(device)
                    pos_tail = pos_tail.to(device)
                    neg_tails = neg_tails.to(device)
                    neg_tails = th.randint(0, nentity, neg_tails.shape, device=device)
                    subsampling_weights = subsampling_weights.to(device)

                    pos_logits = model(init, pos_tail, task_name).unsqueeze(1)
                    neg_logits = model(init, neg_tails, task_name)

                    logger.debug(f"pos_logits: {pos_logits.shape}, neg_logits: {neg_logits.shape}")
                    logger.debug(f"Pos logits: {pos_logits.shape} - subsampling_weights: {subsampling_weights.shape}")
                    logger.debug(f"Neg logits: {neg_logits.shape} - subsampling_weights: {subsampling_weights.shape}")

                    # if task_name in ["2p", "3p", "inp"]:
                        # loss_margin = 0.01
                    # else:
                        # loss_margin = 0.01 #20 #10
                    if old:
                        loss_task = - F.logsigmoid(neg_logits - pos_logits - loss_margin) * subsampling_weights
                        loss += loss_task.sum()/subsampling_weights.sum()
                    else:
                        negative_score = F.logsigmoid(-neg_logits).mean(dim=1)
                        positive_score = F.logsigmoid(pos_logits).squeeze(dim=1)
                        positive_sample_loss = - (subsampling_weights * positive_score).sum()
                        negative_sample_loss = - (subsampling_weights * negative_score).sum()
                        positive_sample_loss /= subsampling_weights.sum()
                        negative_sample_loss /= subsampling_weights.sum()
                        loss += (positive_sample_loss + negative_sample_loss)/2
                        
                    
                    # pos_loss = - F.logsigmoid(loss_margin - pos_logits) * subsampling_weights
                    # neg_loss = - F.logsigmoid(neg_logits - loss_margin) * subsampling_weights#.unsqueeze(1)
                    # pos_loss = pos_loss.sum()/subsampling_weights.sum()
                    # neg_loss = neg_loss.sum()/subsampling_weights.sum()
                    # curr_pos_loss += pos_loss.item()
                    # curr_neg_loss += neg_loss.item()    
                    # loss += pos_loss #.sum() / subsampling_weights.sum()
                    # loss += neg_loss #.sum() / subsampling_weights.sum()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # curr_pos_loss += pos_loss.item()
                # curr_neg_loss += neg_loss.item()
                epoch_loss += loss.item()

            scheduler.step()
            epoch_loss /= len(train_dataloader_1p)
            epoch_trans_score /= len(train_dataloader_1p)
            epoch_non_trans_score /= len(train_dataloader_1p)
            curr_pos_loss /= len(train_dataloader_1p)
            curr_neg_loss /= len(train_dataloader_1p)
            pbar.set_description(f"Prev losses: {prev_pos_loss:.4f}, {prev_neg_loss:.4f}. Curr losses: {curr_pos_loss:.4f}, {curr_neg_loss:.4f}")
            if epoch % 10 == 0:
                # validation_metrics = evaluator.evaluate(model, valid_datasets)
                # log_to_wandb(validation_metrics, epoch, wandb_logger, "valid")
                # valid_str = to_str(validation_metrics)
                test_metrics = evaluator.evaluate(model, test_datasets)
                log_to_wandb(test_metrics, epoch, wandb_logger, "test")
                test_str = to_str(test_metrics)

                logger.info(f"Epoch {epoch}: Loss: {epoch_loss:6f}")
                # print("\nValidation")
                # print(valid_str)
                print("\nTest")
                print(test_str)


if __name__ == '__main__':
    main()
        
