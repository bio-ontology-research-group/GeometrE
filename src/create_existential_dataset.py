import mowl
mowl.init_jvm("20g")
import click as ck
from mowl.datasets import PathDataset
from mowl.owlapi import OWLAPIAdapter
import os
from tqdm import tqdm
import random
from org.semanticweb.owlapi.model import ClassExpressionType as CT
from org.semanticweb.owlapi.model import AxiomType, IRI
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.formats import RDFXMLDocumentFormat
from java.util import HashSet

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
    
@ck.command()
@ck.option("--input_ontology", "-i", type=ck.Path(exists=True), required=True)
@ck.option("--percentage", "-p", type=float, default=0.3)
@ck.option("--random_seed", "-seed", type=int, default=0)
def main(input_ontology, percentage, random_seed):
    """Remove axioms from an ontology. It will remove subclass axioms of
        the form C subclassof some R. D. C and D are concept names and
        R is a role. The percentage value indicates the amount of
        axioms to be used for validation and testing set. If
        percentage = 0.1, 10% of the axioms will be used for and 10%
        for testing.
    """

    if not input_ontology.endswith(".owl"):
        raise ValueError("The input ontology must be in OWL format")
    
    random.seed(random_seed)
    manager = OWLAPIAdapter().owl_manager

    new_directory = os.path.dirname(input_ontology)
    train_file_name = os.path.join(new_directory, "train.owl")
    valid_file_name = os.path.join(new_directory, "valid.owl")
    test_file_name = os.path.join(new_directory, "test.owl")
    os.makedirs(new_directory, exist_ok=True)

    ds = PathDataset(input_ontology)
    ontology = ds.ontology
    tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
    logger.info("Number of initial axioms: {}".format(len(tbox_axioms)))
        
    print("Removing subclass axioms of the form C subclassof some R.D")
    relations_axioms = dict()

    for axiom in tqdm(tbox_axioms, desc="Getting C subclassof R.D axioms"):
        if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
            continue
        if axiom.getSubClass().getClassExpressionType() != CT.OWL_CLASS:
            continue
        if axiom.getSuperClass().getClassExpressionType() != CT.OBJECT_SOME_VALUES_FROM:
            continue
        filler = axiom.getSuperClass().getFiller()
        if filler.getClassExpressionType() != CT.OWL_CLASS:
            continue
        relation = axiom.getSuperClass().getProperty()
        relation_str = str(relation.toStringID())
        if relation_str not in relations_axioms:
            relations_axioms[relation_str] = []
        relations_axioms[relation_str].append(axiom)

    removed_axioms = dict()
        
    for rel_str, axioms in relations_axioms.items():
        num_axioms = len(axioms)
        random.shuffle(axioms)
        axioms_to_remove = axioms[:int(num_axioms*percentage)]
        removed_axioms[rel_str] = axioms_to_remove
        axioms_to_remove_j = HashSet()
        axioms_to_remove_j.addAll(axioms_to_remove)
        print(f"Relation {rel_str}: Removing {len(axioms_to_remove)} axioms from a total of {num_axioms}")
        manager.removeAxioms(ontology, axioms_to_remove_j)

    
    valid_axioms = HashSet()
    test_axioms = HashSet()
    for rel_str, axioms in removed_axioms.items():
        num_axioms = len(axioms)//3
        valid_axioms.addAll(axioms[:num_axioms])
        test_axioms.addAll(axioms[num_axioms:])


    num_axioms = len(ontology.getTBoxAxioms(Imports.fromBoolean(True)))
    print(f"Number of training axioms: {num_axioms}")
                     
        
    valid_ontology = manager.createOntology(valid_axioms)
    num_axioms = len(valid_ontology.getTBoxAxioms(Imports.fromBoolean(True)))
    print(f"Number of validation axioms: {num_axioms}")
    test_ontology = manager.createOntology(test_axioms)
    num_axioms = len(test_ontology.getTBoxAxioms(Imports.fromBoolean(True)))
    print(f"Number of testing axioms: {num_axioms}")
        
    print("Saving ontologies")
    manager.saveOntology(ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(train_file_name)))
    manager.saveOntology(valid_ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(valid_file_name)))
    manager.saveOntology(test_ontology, RDFXMLDocumentFormat(), IRI.create("file:" + os.path.abspath(test_file_name)))
    
    print(f"Done.")

if __name__ == "__main__":
    main()
