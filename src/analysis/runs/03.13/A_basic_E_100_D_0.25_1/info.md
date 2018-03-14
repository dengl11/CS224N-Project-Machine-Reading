———————————
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 100\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --hidden_size 160\
            --pred_hidden_sz 100\
            --dropout 0.25\
            --mode=train
————————————
Num params: 1596962

Epoch 10, Iter 8500, Train F1 score: 0.694336, Train EM score: 0.556000

Epoch 10, Iter 8500, Dev F1 score: 0.614509, Dev EM score: 0.467231
