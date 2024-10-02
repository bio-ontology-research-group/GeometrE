from org.semanticweb.owlapi.model import AxiomType as ax

IGNORED_AXIOM_TYPES = [ax.ANNOTATION_ASSERTION,
                       ax.ASYMMETRIC_OBJECT_PROPERTY,
                       ax.DECLARATION,
                       ax.EQUIVALENT_OBJECT_PROPERTIES,
                       ax.FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_FUNCTIONAL_OBJECT_PROPERTY,
                       ax.INVERSE_OBJECT_PROPERTIES,
                       ax.IRREFLEXIVE_OBJECT_PROPERTY,
                       ax.OBJECT_PROPERTY_DOMAIN,
                       ax.OBJECT_PROPERTY_RANGE,
                       ax.REFLEXIVE_OBJECT_PROPERTY,
                       ax.SUB_PROPERTY_CHAIN_OF,
                       ax.SUB_ANNOTATION_PROPERTY_OF,
                       ax.SUB_OBJECT_PROPERTY,
                       ax.SWRL_RULE,
                       ax.SYMMETRIC_OBJECT_PROPERTY,
                       ax.TRANSITIVE_OBJECT_PROPERTY
                       ]
