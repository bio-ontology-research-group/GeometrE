import os
import pickle as pkl
import click as ck
import sys
from util import transitive_roles

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
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


query_name_dict_inv = {v: k for k, v in query_name_dict.items()}


@ck.command()
@ck.option("--dataset", "-ds", type=ck.Choice(["WN18RR-QA"]))
def main(dataset):
    root_dir = f"../../use_cases/query_answering_data/{dataset}"

    transitive_closure_file = os.path.join(root_dir, "train_saturated.txt")
    transitive_triples = load_transitive_closure(transitive_closure_file)

    role2id_file = os.path.join(root_dir, "rel2id.pkl")
    role_to_id = pkl.load(open(role2id_file, "rb"))
    
    valid_queries_file = os.path.join(root_dir, "valid-queries.pkl")
    test_queries_file = os.path.join(root_dir, "test-queries.pkl")
    
    valid_easy_answers_file = os.path.join(root_dir, "valid-easy-answers.pkl")
    valid_hard_answers_file = os.path.join(root_dir, "valid-hard-answers.pkl")
    test_easy_answers_file = os.path.join(root_dir, "test-easy-answers.pkl")
    test_hard_answers_file = os.path.join(root_dir, "test-hard-answers.pkl")

    saturated_valid_easy_answers_file = os.path.join(root_dir, "saturated-valid-easy-answers.pkl")
    saturated_valid_hard_answers_file = os.path.join(root_dir, "saturated-valid-hard-answers.pkl")
    saturated_test_easy_answers_file = os.path.join(root_dir, "saturated-test-easy-answers.pkl")
    saturated_test_hard_answers_file = os.path.join(root_dir, "saturated-test-hard-answers.pkl")

    saturated_valid_easy_answers = saturate(transitive_triples, valid_queries_file, valid_easy_answers_file, dataset, role_to_id)
    saturated_valid_hard_answers = saturate(transitive_triples, valid_queries_file, valid_hard_answers_file, dataset, role_to_id)
    saturated_test_easy_answers =  saturate(transitive_triples, test_queries_file, test_easy_answers_file, dataset, role_to_id)
    saturated_test_hard_answers =  saturate(transitive_triples, test_queries_file, test_hard_answers_file, dataset, role_to_id)

    pkl.dump(saturated_valid_easy_answers, open(saturated_valid_easy_answers_file, "wb"))
    pkl.dump(saturated_valid_hard_answers, open(saturated_valid_hard_answers_file, "wb"))
    pkl.dump(saturated_test_easy_answers, open(saturated_test_easy_answers_file, "wb"))
    pkl.dump(saturated_test_hard_answers, open(saturated_test_hard_answers_file, "wb"))

    logger.info("Saturated answers saved.")
    
def load_transitive_closure(transitive_closure_file):    
    transitive_triples = dict()
    with open(transitive_closure_file) as f:
        lines = f.readlines()
        for line in lines:
            triple = line.strip().split("\t")
            head, relation, tail = triple
            head = int(head)
            relation = int(relation)
            tail = int(tail)
            
            if not (head, relation) in transitive_triples:
                transitive_triples[(head, relation)] = set()
            transitive_triples[(head, relation)].add(tail)
            
    return transitive_triples

def saturate(transitive_triples, queries_file, answers_file, dataset, role_to_id):
    roles_to_saturate = [str(role_to_id[t]) for t in transitive_roles[dataset]]

    patterns_to_saturate = ['1p', '2p', '3p', 'inp', 'ip']
    patterns_to_saturate = [query_name_dict_inv[p] for p in patterns_to_saturate]
    
    queries_dict = pkl.load(open(queries_file, "rb"))
    answers_dict = pkl.load(open(answers_file, "rb"))
    saturated_answers = dict()
    saturation_cases = 0
    for query_type, queries in queries_dict.items():
        if query_type not in patterns_to_saturate:
            logger.info(f"Skipping query type: {query_type}")
            for query in queries:
                saturated_answers[query] = answers_dict[query]
        else:
            logger.info(f"Saturating query type: {query_type}")
            for query in queries:
                relation = str(query[-1][-1])
                assert type(relation) == type(roles_to_saturate[0]), f"Type of relation: {type(relation)} and type of roles_to_saturate[0]: {type(roles_to_saturate[0])}"
                if relation not in roles_to_saturate:
                    saturated_answers[query] = answers_dict[query]
                    continue
                else:
                    prev_answers = answers_dict[query]
                    new_answers = set() | prev_answers
                    for answer in prev_answers:
                        pair = (answer, int(relation))
                        if pair in transitive_triples:
                            saturation_cases += 1
                            new_answers |= set(transitive_triples[pair])
                    saturated_answers[query] = new_answers

    logger.info(f"Saturation cases: {saturation_cases}")
    return saturated_answers

if __name__ == "__main__":
    main()
