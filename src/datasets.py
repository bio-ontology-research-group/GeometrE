import mowl
from mowl.datasets import PathDataset
from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.datasets.base import OWLClasses
from mowl.owlapi import OWLAPIAdapter

class SubsumptionDataset(PathDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir + "train.owl", root_dir + "valid.owl", root_dir + "test.owl")

        self.root_dir = root_dir
        self._deductive_closure_ontology = None
        
    @property
    def deductive_closure_ontology(self):
        if self._deductive_closure_ontology is None:
            self._deductive_closure_ontology = PathDataset(self.root_dir + "train_deductive_closure.owl").ontology

        return self._deductive_closure_ontology

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:

            train_classes = self.ontology.getClassesInSignature()
            valid_classes = self.validation.getClassesInSignature()
            test_classes = self.testing.getClassesInSignature()
            
            valid_not_in_train = set(valid_classes) - set(train_classes)
            test_not_in_train = set(test_classes) - set(train_classes)
            if len(valid_not_in_train) > 0:
                print(f"Ignoring {len(valid_not_in_train)} classes in validation set that are not in the training set")
            if len(test_not_in_train) > 0:
                print(f"Ignoring {len(test_not_in_train)} classes in test set that are not in the training set")
            
            not_in_train = valid_not_in_train.union(test_not_in_train)

            # assert set(valid_classes) - set(train_classes) == set(), f"Valid classes not in train: {set(valid_classes) - set(train_classes)}"
            # assert set(test_classes) - set(train_classes) == set(), f"Test classes not in train: {set(test_classes) - set(train_classes)}"

            
            classes = self.ontology.getClassesInSignature()

            bot_in_classes = False
            top_in_classes = False

            for cls in classes:
                if cls.isOWLNothing():
                    bot_in_classes = True
                if cls.isOWLThing():
                    top_in_classes = True

            if not bot_in_classes:
                print("Did not find owl:Nothing in ontology classes. Adding it.")
                classes.add(self.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLNothing())
            if not top_in_classes:
                print("Did not find owl:Thing in ontology classes. Adding it.")
                classes.add(self.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLThing())

            classes = OWLClasses(classes)
            self._evaluation_classes = classes, classes

        return self._evaluation_classes

class KGDataset(PathDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir + "train_with_rbox.owl", root_dir + "valid.owl", root_dir + "test.owl")

        self.root_dir = root_dir
        self._deductive_closure_ontology = None
        self._transitive_test_ontology = None
        
    @property
    def deductive_closure_ontology(self):
        if self._deductive_closure_ontology is None:
            self._deductive_closure_ontology = PathDataset(self.root_dir + "train_deductive_closure.owl").ontology
            # self._deductive_closure_ontology = PathDataset(self.root_dir + "test_trans_only.owl").ontology

        return self._deductive_closure_ontology

    @property
    def transitive_test_ontology(self):
        if self._transitive_test_ontology is None:
            self._transitive_test_ontology = PathDataset(self.root_dir + "test_trans_only.owl").ontology
            # self._deductive_closure_ontology = PathDataset(self.root_dir + "test_trans_only.owl").ontology

        return self._transitive_test_ontology

    
    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:

            train_classes = self.ontology.getClassesInSignature()
            valid_classes = self.validation.getClassesInSignature()
            test_classes = self.testing.getClassesInSignature()
            ded_classes = self.deductive_closure_ontology.getClassesInSignature()
            

            valid_not_in_train = set(valid_classes) - set(train_classes)
            test_not_in_train = set(test_classes) - set(train_classes)
            if len(valid_not_in_train) > 0:
                print(f"Ignoring {len(valid_not_in_train)} classes in validation set that are not in the training set")
            if len(test_not_in_train) > 0:
                print(f"Ignoring {len(test_not_in_train)} classes in test set that are not in the training set")
            
            not_in_train = valid_not_in_train.union(test_not_in_train)

            # assert set(valid_classes) - set(train_classes) == set(), f"Valid classes not in train: {set(valid_classes) - set(train_classes)}"
            # assert set(test_classes) - set(train_classes) == set(), f"Test classes not in train: {set(test_classes) - set(train_classes)}"

            
            classes = self.classes #ontology.getClassesInSignature()

            ded_classes = self.deductive_closure_ontology.getClassesInSignature()
            classes = set(classes) | set(ded_classes)

            
            bot_in_classes = False
            top_in_classes = False

            for cls in classes:
                if cls.isOWLNothing():
                    bot_in_classes = True
                if cls.isOWLThing():
                    top_in_classes = True

            # if not bot_in_classes:
                # print("Did not find owl:Nothing in ontology classes. Adding it.")
                # classes.add(self.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLNothing())
            # if not top_in_classes:
                # print("Did not find owl:Thing in ontology classes. Adding it.")
                # classes.add(self.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLThing())

            classes = OWLClasses(classes)
            self._evaluation_classes = classes, classes

        return self._evaluation_classes

                         
class PPIDataset(PathDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir + "ontology.owl", root_dir + "valid.owl", root_dir + "test.owl")

        self.root_dir = root_dir
        self._deductive_closure_ontology = None
        
    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://4932" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes
