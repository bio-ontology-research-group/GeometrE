import os
import pickle as pkl
import click as ck
import sys
from util import transitive_roles_dict, inverse_roles_dict
import copy
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
    root_dir = f"data/{dataset}"

    role2id_file = os.path.join(root_dir, "rel2id.pkl")
    role_to_id = pkl.load(open(role2id_file, "rb"))

    train_queries_file = os.path.join(root_dir, "train-queries.pkl")
    valid_queries_file = os.path.join(root_dir, "valid-queries.pkl")
    test_queries_file = os.path.join(root_dir, "test-queries.pkl")
    new_train_queries_file = os.path.join(root_dir, "new-train-queries.pkl")
    new_valid_queries_file = os.path.join(root_dir, "new-valid-queries.pkl")
    new_test_queries_file = os.path.join(root_dir, "new-test-queries.pkl")
    
    train_answers_file = os.path.join(root_dir, "train-answers.pkl")
    valid_easy_answers_file = os.path.join(root_dir, "valid-easy-answers.pkl")
    valid_hard_answers_file = os.path.join(root_dir, "valid-hard-answers.pkl")
    test_easy_answers_file = os.path.join(root_dir, "test-easy-answers.pkl")
    test_hard_answers_file = os.path.join(root_dir, "test-hard-answers.pkl")

    new_train_queries = process_queries(train_queries_file, train_answers_file, train_answers_file, dataset, role_to_id)
    new_valid_queries = process_queries(valid_queries_file, valid_easy_answers_file, valid_hard_answers_file, dataset, role_to_id)
    new_test_queries = process_queries(test_queries_file, test_easy_answers_file, test_hard_answers_file, dataset, role_to_id)
    
    pkl.dump(new_train_queries, open(new_train_queries_file, "wb"))
    pkl.dump(new_valid_queries, open(new_valid_queries_file, "wb"))
    pkl.dump(new_test_queries, open(new_test_queries_file, "wb"))
        
    logger.info("New answers saved.")


def process_queries(queries_file, easy_answers_file, hard_answers_file, dataset, role_to_id):
    transitive_roles = [str(role_to_id[t]) for t in transitive_roles_dict[dataset]]
    inverse_roles = [str(role_to_id[t]) for t in inverse_roles_dict[dataset]]

    id_to_role = {v: k for k, v in role_to_id.items()}
    
    patterns_of_interest = ['1p', '2p', '3p', 'inp', 'ip']
    

    transitive_query_structures = {"1p": ('e', ('r', 't')),
                                   "2p": ('e', ('r', 'r', 't')),
                                   "3p": ('e', ('r', 'r', 'r', 't')),
                                   "inp": ((('e', ('r',)), ('e', ('r', 'n'))), ('r', 't')),
                                   "ip": ((('e', ('r',)), ('e', ('r',))), ('r', 't'))
                                   }

    inverse_query_structures = {"1p": ('e', ('r', 'i')),
                                   "2p": ('e', ('r', 'r', 'i')),
                                   "3p": ('e', ('r', 'r', 'r', 'i')),
                                   "inp": ((('e', ('r',)), ('e', ('r', 'n'))), ('r', 'i')),
                                   "ip": ((('e', ('r',)), ('e', ('r',))), ('r', 'i'))
                                   }
    

    queries_dict = pkl.load(open(queries_file, "rb"))
    easy_answers_dict = pkl.load(open(easy_answers_file, "rb"))
    hard_answers_dict = pkl.load(open(hard_answers_file, "rb"))
    
    print(f"Type of key and values in queries_dict: {type(list(queries_dict.keys())[0])}, {type(list(queries_dict.values())[0])}")
    print(f"Type of key and values in answers_dict: {type(list(easy_answers_dict.keys())[0])}, {type(list(easy_answers_dict.values())[0])}")
    print(f"Type of key and values in answers_dict: {type(list(hard_answers_dict.keys())[0])}, {type(list(hard_answers_dict.values())[0])}")
    
    num_transitive_queries = {role: 0 for role in transitive_roles}
    num_inverse_queries = {role: 0 for role in inverse_roles}
    num_transitive_answers = {role: 0 for role in transitive_roles}
    num_inverse_answers = {role: 0 for role in inverse_roles}

    aux_queries_dict = copy.deepcopy(queries_dict)
    for query_structure, queries in aux_queries_dict.items():
        query_type = query_name_dict[query_structure]
        if query_type not in patterns_of_interest:
            logger.info(f"Skipping query type: {query_type}")
            continue
        
        logger.info(f"Processing query type: {query_type}")
        for query in queries:
            relation = query[-1][-1]
            relation = str(relation)
            
            if relation in transitive_roles:
                num_transitive_queries[relation] += 1
                num_transitive_answers[relation] += len(easy_answers_dict[query]) + len(hard_answers_dict[query])

                if relation in inverse_roles:
                    num_inverse_queries[relation] += 1
                    num_inverse_answers[relation] += len(easy_answers_dict[query]) + len(hard_answers_dict[query])
                    new_query_structure = inverse_query_structures[query_type]
                else:
                    new_query_structure = transitive_query_structures[query_type]

                queries_dict[query_structure].remove(query)
                if not new_query_structure in queries_dict:
                    queries_dict[new_query_structure] = set()
                queries_dict[new_query_structure].add(query)
            

    logger.info(f"Transitive queries: {num_transitive_queries}")
    logger.info(f"Inverse queries: {num_inverse_queries}")
    logger.info(f"Transitive answers: {num_transitive_answers}")
    logger.info(f"Inverse answers: {num_inverse_answers}")

    return queries_dict

if __name__ == "__main__":
    main()
