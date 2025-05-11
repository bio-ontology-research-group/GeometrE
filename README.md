# Fully Geometric Multi-Hop Reasoning on Knowledge Graphs With Transitive Relations
------------------
## Main Requirements and installation
* Python 3.10.15
* PyTorch 2.4.1

You can install all the requirements through `environment.yml` as follows

```
conda env create -f environment.yml
conda activate geometre
```
---
## Data
* WN18RR was obtained from from https://github.com/nelsonhuangzijian/WN18RR-QA
* NELL and FB15k-237 are obtained from http://snap.stanford.edu/betae/KG_data.zip:
  ```
  wget http://snap.stanford.edu/betae/KG_data.zip
  ```

Place the data under `src/data` such that the file structure looks like tis:
```
.
├── environment.yml
├── README.md
├── requirements.txt
└── src
    ├── data
        ├── FB15k-237-betae
	├── NELL-betae
        └── WN18RR-QA

```
----
## Reproduce the results

To reproduce the results please run the following commands:

* WN18RR
```
python main.py --do_train --do_test --alpha=0.5 --batch_size=1024 --data_path=data/WN18RR-QA --gamma=20 --hidden_dim=400 --learning_rate=0.001 --negative_sample_size=64 --transitive=no --with_answer_embedding
```

* NELL
```
python main.py --do_train --do_test --alpha=0.2 --batch_size=1024 --data_path=data/NELL-betae --gamma=10 --hidden_dim=400 --learning_rate=0.0005 --negative_sample_size=64 --transitive=no
```

* FB15k-237
```
python main.py --do_train --do_test --alpha=0.2 --batch_size=1024 --data_path=data/FB15k-237-betae --gamma=20 --hidden_dim=400 --learning_rate=0.0005 --negative_sample_size=64 --transitive=no
```
