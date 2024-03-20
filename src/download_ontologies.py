import os
import wget

script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(script_dir, "../data/")

def main():
    goslim_url = "http://www.geneontology.org/ontology/subsets/goslim_generic.owl"
    go_plus_url = "http://purl.obolibrary.org/obo/go/extensions/go-plus.owl"
    go_url = "http://purl.obolibrary.org/obo/go.owl"

    os.makedirs(root_dir, exist_ok=True)

    wget.download(goslim_url, root_dir)
    wget.download(go_url, root_dir)
    wget.download(go_plus_url, root_dir)
    
if __name__ == "__main__":
    main()
    
