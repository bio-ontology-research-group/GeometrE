# This script generates triples of triples in the following way: For a
# triple (a, r, b) in the training set where r is a transitive
# relation, it queries triples (b, r, c) in the testing set such that
# the triple (a, r, c) is not in the testing set. Then it outputs the
# triples (a, r, b), (b, r, c), (a, r, c) in the form: "r,a,b,c"

import sys
import os
import click as ck
import logging

from tqdm import tqdm

from utils import transitive_roles

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@ck.command()
@ck.option("--dataset", "-ds", type=ck.Choice(["wn18rr", "fb15k237"]))
def main(dataset):
    path = f"../use_cases/{dataset}/data"

    training_file = os.path.join(path, "train.txt")
    testing_file = os.path.join(path, "test.txt")
    out_file = os.path.join(path, "chains.txt")
    
    training_data = set()
    with open(training_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            assert len(line) == 3, f"Unrecognized line: {line}"
            training_data.add(tuple(line))


    testing_data = dict()
    with open(testing_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            assert len(line) == 3, f"Unrecognized line: {line}"
            h, r, t = tuple(line)

            if not (h, r) in testing_data:
                testing_data[(h, r)] = set()

            testing_data[(h, r)].add(t)

    chains = set()

    trans_roles = transitive_roles[dataset]
    
    for h, r, t in tqdm(training_data, desc="Starting search of 2-hop chains"):
        if not r in trans_roles:
            continue
        
        next_tails = testing_data.get((t, r), list())

        
        
        for next_t in next_tails:
            chains.add((r, h, t, next_t))

    print(f"Found {len(chains)} chains")
            
    with open(out_file, "w") as f:
        for chain in chains:
            string = "\t".join(chain)
            f.write(f"{string}\n")
            
    print("Done!")
            
            
if __name__ == "__main__":
    main()
