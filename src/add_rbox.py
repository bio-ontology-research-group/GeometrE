# This scripts adds transitive rbox axioms to KGs such as WordNet18RR

import mowl
mowl.init_jvm("10g")

from mowl.owlapi import OWLAPIAdapter
from mowl.datasets import PathDataset
import click as ck
import os
import logging

from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
from java.util import HashSet

from utils import transitive_roles

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

owl_prefix = "http://www.w3.org/2002/07/"

@ck.command()
@ck.option("--dataset", "-ds", type=ck.Choice(["wn18rr"]), default="wn18rr")
def main(dataset):
    path = f"../use_cases/{dataset}/data"
    train_owl_file = os.path.join(path, "train.owl")
    assert os.path.exists(train_owl_file), f"File {train_owl_file} not found"

    trans_roles = transitive_roles[dataset]
    trans_roles = [owl_prefix + r for r in trans_roles]

    ds = PathDataset(train_owl_file)

    ontology = ds.ontology
    roles = ontology.getObjectPropertiesInSignature()

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    factory = adapter.data_factory

    axioms = HashSet()
    for role in roles:
        role_str = str(role.toStringID())
        if role_str in trans_roles:
            logger.info(f"Transitive role {role_str} found")

            axiom = factory.getOWLTransitiveObjectPropertyAxiom(role)
            axioms.add(axiom)

    manager.addAxioms(ontology, axioms)

    out_file = train_owl_file.replace(".owl", "_with_rbox.owl")
    manager.saveOntology(ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(out_file)))

    print("Done!")


if __name__ == "__main__":
    main()
