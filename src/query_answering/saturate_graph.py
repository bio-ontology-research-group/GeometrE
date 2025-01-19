# This script saturates the input graph based on the transitive relations therein.

import os
import click as ck
from util import transitive_roles
import copy
import pickle as pkl
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def fst(tuple_):
    return tuple_[0]

def snd(tuple_):
    return tuple_[1]

def thd(tuple_):
    return tuple_[2]
    
@ck.command()
@ck.option("--dataset", "-ds", type=ck.Choice(["WN18RR-QA"]))
def main(dataset):
    path = f"../../use_cases/query_answering_data/{dataset}"
    triples_file = os.path.join(path, "train.txt")
    # triples_file = os.path.join(path, "test.txt")
    assert os.path.exists(triples_file), f"Cannot find {triples_file}"

    triples = set()
    with open(triples_file) as f:
        lines = f.readlines()
        for line in lines:
            triple = line.strip().split("\t")
            assert len(triple) == 3, f"Triple {triple} with {len(triple)} elements found."
            triple = tuple(triple)
            triples.add(triple)
    logger.info(f"Graph contains {len(triples)}.")


    role_to_id = pkl.load(open(os.path.join(path, "rel2id.pkl"), "rb"))
    print(role_to_id)
    roles_to_saturate = [str(role_to_id[role]) for role in transitive_roles[dataset]]
    
    transitive_triples = set([t for t in triples if snd(t) in roles_to_saturate])
    logger.info(f"Graph contains {len(transitive_triples)}.")

    continue_ = input("Continue? (y/n): ")

    if continue_ == "n":
        return

    
    saturated = False
    saturated_rel = {rel: False for rel in roles_to_saturate}
    out_edges_per_rel = dict()
    for h, r, t in transitive_triples:
        if not r in out_edges_per_rel:
            out_edges_per_rel[r] = dict()
        if not h in out_edges_per_rel[r]:
            out_edges_per_rel[r][h] = set()

        out_edges_per_rel[r][h].add(t)

    checkpoint_dict = copy.deepcopy(out_edges_per_rel)

    saturation_round = 0
    while not saturated:
        saturation_round += 1
        logger.info(f"Saturation round {saturation_round}")
        for rel, out_edges in out_edges_per_rel.items():
            added_rel = 0
            if saturated_rel[rel]:
                continue
            logger.info(f"Saturating relation {rel}")
            for h, ts in out_edges.items():
                # if not h in out_edges_per_rel[rel]:
                    # continue
                added = 0
                initial_len = len(checkpoint_dict[rel][h])
                for t in ts:
                    if not t in out_edges:
                        continue
           
                    checkpoint_dict[rel][h] |= (out_edges_per_rel[rel][t])
                    added += len(out_edges_per_rel[rel][t])
                    added_rel += len(out_edges_per_rel[rel][t])
                final_len = len(checkpoint_dict[rel][h])
                logger.debug(f"Added {added} triples with {h}, {rel}. Initial length {initial_len}, final length {final_len}")
            logger.info(f"Added {added_rel} triples for rel {rel}.")
            if checkpoint_dict[rel] == out_edges_per_rel[rel]:
                saturated_rel[rel] = True
        out_edges_per_rel = copy.deepcopy(checkpoint_dict)

        saturated = True
        for sat_rel, value in saturated_rel.items():
            saturated = saturated and value

    flattened_triples = set()
    for rel, out_edges in out_edges_per_rel.items():
        for head, tails in out_edges.items():
            triples = set([(head, rel, t) for t in tails])
            flattened_triples |= triples

    logger.info(f"Saturated graph contains {len(flattened_triples)}")
            
    only_closure = flattened_triples - transitive_triples
    logger.info(f"Only closure contains {len(only_closure)}")

    with open(os.path.join(path, "train_saturated.txt"), "w") as f:
    # with open(os.path.join(path, "test_saturated.txt"), "w") as f:
        for triple in only_closure:
            f.write("\t".join(triple) + "\n")

    
    logging.info("Done!")
            
if __name__ == "__main__":
    main()
