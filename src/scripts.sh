# WN18RR
python main.py --do_train --do_test --alpha=0.5 --batch_size=1024 --data_path=data/WN18RR-QA --gamma=20 --hidden_dim=400 --learning_rate=0.001 --negative_sample_size=64 --transitive=yes --with_answer_embedding

# NELL
python main.py --do_train --do_test --alpha=0.2 --batch_size=1024 --data_path=data/NELL-betae --gamma=10 --hidden_dim=400 --learning_rate=0.0005 --negative_sample_size=64 --transitive=no

# FB15k-237
python main.py --do_train --do_test --alpha=0.2 --batch_size=1024 --data_path=data/FB15k-237-betae --gamma=20 --hidden_dim=400 --learning_rate=0.0005 --negative_sample_size=64 --transitive=no
