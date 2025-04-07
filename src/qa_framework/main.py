#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGReasoning
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple, transitive_roles_dict, inverse_roles_dict
import wandb

query_name_dict = {('e',('r',)): '1p', 
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
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM',
                   ('e', ('r', 't')): '1pto',
                   ('e', ('r', 'r', 't')): '2pto',
                   ('e', ('r', 'r', 'r', 't')): '3pto',
                   ((('e', ('r',)), ('e', ('r',))), ('r', 't')): 'ipto',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r', 't')): 'inpto',
                   ('e', ('r', 'i')): '1pti',
                   ('e', ('r', 'r', 'i')): '2pti',
                   ('e', ('r', 'r', 'r', 'i')): '3pti',
                   ((('e', ('r',)), ('e', ('r',))), ('r', 'i')): 'ipti',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r', 'i')): 'inpti',
                   
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys()) # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")
    parser.add_argument('--do_test_full_tr', action='store_true', help="do test on saturated transitive set")
    parser.add_argument('--plot_embeddings', action='store_true', help="plot embeddings")
    parser.add_argument('--data_path', type=str, default="data/WN18RR-QA", help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=12.0, type=float, help="margin in the loss")
    parser.add_argument('-a', '--alpha', default=0.5, type=float, help="balance the in_box dist and out_box dist")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=3, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default=None, type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=200001, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    
    parser.add_argument('--save_checkpoint_steps', default=5000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=5000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    parser.add_argument('--geo', default='pure_box', type=str, choices=['pure_box'], help='the reasoning model')
    parser.add_argument('--print_on_screen', action='store_true')
    
    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')
    parser.add_argument('-ts', '--transitive', type=str, choices=['yes', 'no'], default='no', help='whether to include transitive queries')
    parser.add_argument('-ns', '--no_sweep', action='store_true', help='Not a W&B sweep')
    parser.add_argument('-desc', '--description', default='default', type=str, help='description of the run')
    return parser.parse_args(args)

def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer, wb_logger):
    '''
    Evaluate queries in dataloader
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            query_type = query_name_dict[query_structure]
            wb_logger.log({f"{mode}_{query_type}_{metric}": metrics[query_structure][metric]})
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        wb_logger.log({f"{mode}_avg_{metric}": average_metrics[metric]})
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)

    return all_metrics
        
def load_data(args, tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    logging.info("loading data")
    # if args.transitive == 'yes':
        # train_queries = pickle.load(open(os.path.join(args.data_path, "new-train-queries.pkl"), 'rb'))
        # valid_queries = pickle.load(open(os.path.join(args.data_path, "new-valid-queries.pkl"), 'rb'))
        # test_queries = pickle.load(open(os.path.join(args.data_path, "new-test-queries.pkl"), 'rb'))
    # else:
    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))

    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    valid_tr_queries = pickle.load(open(os.path.join(args.data_path, "transitive-valid-queries.pkl"), 'rb'))
    test_tr_queries = pickle.load(open(os.path.join(args.data_path, "transitive-test-queries.pkl"), 'rb'))
        
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))

    if args.do_test_full_tr:
        valid_queries = valid_tr_queries
        test_queries = test_tr_queries
        valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "saturated-valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "saturated-valid-easy-answers.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(args.data_path, "saturated-test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(args.data_path, "saturated-test-easy-answers.pkl"), 'rb'))

    else:
        valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))
        
    rel2id = pickle.load(open(os.path.join(args.data_path, "rel2id.pkl"), 'rb'))
    
    # remove tasks not in args.tasks
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union
        if name not in tasks or evaluate_union != args.evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]
            if query_structure in valid_tr_queries:
                del valid_tr_queries[query_structure]
            if query_structure in test_tr_queries:
                del test_tr_queries[query_structure]

    dataset_name = args.data_path.split('/')[-1]
    transitive_roles = transitive_roles_dict[dataset_name]
    inverse_roles = inverse_roles_dict[dataset_name]
    transitive_ids = torch.tensor([rel2id[role] for role in transitive_roles], dtype=torch.long)
    inverse_ids = torch.tensor([rel2id[role] for role in inverse_roles], dtype=torch.long)
                
    return train_queries, train_answers, valid_queries, valid_tr_queries, valid_hard_answers, valid_easy_answers, test_queries, test_tr_queries, test_hard_answers, test_easy_answers, transitive_ids, inverse_ids

def main(args):
    args.cuda = True
    wandb_logger = wandb.init(entity="ferzcam", project="box_qa", name=args.description)

    
    if args.no_sweep:
        wandb_logger.log({"hidden_dim": args.hidden_dim,
                          "gamma": args.gamma,
                          "alpha": args.alpha,
                          "learning_rate": args.learning_rate,
                          "batch_size": args.batch_size,
                          "negative_sample_size": args.negative_sample_size,
                          "transitive": args.transitive,
                          })
    else:
        args.hidden_dim = wandb.config.hidden_dim
        args.gamma = wandb.config.gamma
        args.alpha = wandb.config.alpha
        args.learning_rate = wandb.config.learning_rate
        args.batch_size = wandb.config.batch_size
        args.negative_sample_size = wandb.config.negative_sample_size
        args.transitive = wandb.config.transitive

    if args.transitive == 'yes':
        args.transitive = True
    elif args.transitive == 'no':
        args.transitive = False
    else:
        raise ValueError("transitive should be 'yes' or 'no'")
        
    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix

    print ("overwritting args.save_path")
    args.save_path = os.path.join(prefix, args.data_path.split('/')[-1], args.tasks, args.geo)
    if args.geo in ['pure_box']:
        tmp_str = "hid-{}-g-{}-a-{}-lr-{}-bs-{}-ns-{}-trans-{}".format(args.hidden_dim, args.gamma, args.alpha, args.learning_rate, args.batch_size, args.negative_sample_size, args.transitive)
    elif args.geo in ['vec']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo == 'beta':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)

    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path
    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print ("logging to", args.save_path)
    if not args.do_train: # if not training, then create tensorboard files in some tmp location
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)

    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('-------------------------------'*3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)

    train_queries, train_answers, valid_queries, valid_tr_queries, valid_hard_answers, valid_easy_answers, test_queries, test_tr_queries, test_hard_answers, test_easy_answers, transitive_ids, inverse_ids = load_data(args, tasks)        

    logging.info("Training info:")
    if args.do_train:
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
                                    TrainDataset(train_path_queries, nentity, nrelation, args.negative_sample_size, train_answers),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.cpu_num,
                                    collate_fn=TrainDataset.collate_fn,
                                ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                                        TrainDataset(train_other_queries, nentity, nrelation, args.negative_sample_size, train_answers),
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.cpu_num,
                                        collate_fn=TrainDataset.collate_fn,
                                    ))
        else:
            train_other_iterator = None
    
    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries, 
                args.nentity, 
                args.nrelation, 
            ), 
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num, 
            collate_fn=TestDataset.collate_fn
        )


    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries, 
                args.nentity, 
                args.nrelation, 
            ), 
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num, 
            collate_fn=TestDataset.collate_fn
        )

        for query_structure in test_tr_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(test_tr_queries[query_structure])))
        test_tr_queries = flatten_query(test_tr_queries)
        test_tr_dataloader = DataLoader(
            TestDataset(
                test_tr_queries, 
                args.nentity, 
                args.nrelation, 
            ), 
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num, 
            collate_fn=TestDataset.collate_fn
        )

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        alpha=args.alpha,
        geo=args.geo,
        use_cuda=args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode=eval_tuple(args.beta_mode),
        test_batch_size=args.test_batch_size,
        query_name_dict=query_name_dict,
        transitive_ids=transitive_ids,
        inverse_ids=inverse_ids
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)

    if args.cuda:
        model = model.cuda()
    
    if args.do_train:
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=current_learning_rate
        )
        warm_up_steps = args.max_steps // 2
        logging.info('Warming up for %d steps' % warm_up_steps)
        
    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0


    if args.plot_embeddings:
        # model.plot_embeddings(args, outfilename=os.path.join(args.save_path, 'embeddings.png'))
        model.plot_chains(args)
        sys.exit(0)
        
    step = init_step 
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    
    if args.do_train:
        training_logs = []
        # #Training Loop
        for step in range(init_step, args.max_steps):
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_'+metric, log[metric], step)
            training_logs.append(log)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                training_logs.append(log)
                log = model.train_step(model, optimizer, train_path_iterator, args, step)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)

            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer, wandb_logger)

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer, wandb_logger)
                    logging.info('Evaluation on Transitive Test Dataset:')
                    test_tr_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_tr_dataloader, query_name_dict, 'TestTr', step, writer, wandb_logger)
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)
        
    try:
        print (step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer, wandb_logger)
        logging.info('Evaluation on Transitive Test Dataset:')
        test_tr_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_tr_dataloader, query_name_dict, 'TestTr', step, writer, wandb_logger)

    logging.info("Training finished!!")

if __name__ == '__main__':
    main(parse_args())
