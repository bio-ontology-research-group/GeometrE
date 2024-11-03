# This script preprocesses .txt files of KGs such as WordNet18RR
# The input is three files: train.txt, valid.txt and test.txt
# The output is three files: train.owl, valid.owl and test.owl


import mowl
mowl.init_jvm("10g")
from mowl.ontology.create import create_from_triples

import sys
import os
import click as ck

owl_prefix = "http://www.w3.org/2002/07/"


@ck.command()
@ck.option("--dataset", "-ds", type=ck.Choice(["wn18rr", "fb15k237", "yago310"]))
def main(dataset):
    path = f"../use_cases/{dataset}/data"
    train_triples_file = os.path.join(path, "train.txt")
    valid_triples_file = os.path.join(path, "valid.txt")
    test_triples_file = os.path.join(path, "test.txt")
    train_deductive_triples = os.path.join(path, "train_deductive_closure.txt")
    test_trans_only = os.path.join(path, "test_trans_only.txt")
    
    assert os.path.exists(f"{train_triples_file}"), f"File {train_triples_file} not found"
    assert os.path.exists(f"{valid_triples_file}"), f"File {valid_triples_file} not found"
    assert os.path.exists(f"{test_triples_file}"), f"File {test_triples_file} not found"
    assert os.path.exists(f"{train_deductive_triples}"), f"File {train_deductive_triples} not found"
    assert os.path.exists(f"{test_trans_only}"), f"File {test_trans_only} not found"

    train_owl_file = train_triples_file.replace(".txt", ".owl")
    valid_owl_file = valid_triples_file.replace(".txt", ".owl")
    test_owl_file = test_triples_file.replace(".txt", ".owl")
    train_deductive_owl_file = train_deductive_triples.replace(".txt", ".owl")
    test_trans_only_owl_file = test_trans_only.replace(".txt", ".owl")

    
    create_from_triples(train_triples_file, train_owl_file)
    create_from_triples(valid_triples_file, valid_owl_file)
    create_from_triples(test_triples_file, test_owl_file)
    create_from_triples(train_deductive_triples, train_deductive_owl_file)
    create_from_triples(test_trans_only, test_trans_only_owl_file)
    print("Done!")


if __name__ == "__main__":
    main()
