————————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --batch_size 120\
            --embedding_size 200\
            --hidden_size 160\
            --context_len=500\
            --output_size 200\
            --output lstm\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25
————————————	
Num params: 1692962



[main] Epoch 10, Iter 6500, Train F1 score: 0.711927 Train EM score: 0.581000
[main] Epoch 10, Iter 6500, Dev F1 score: 0.615566,   Dev EM score: 0.464344
[main] Epoch 12, Iter 8000, Dev F1 score: 0.624488,                            Dev EM score: 0.473583
[main] Epoch 12, Iter 8000, Train F1 score: 0.752107,                            Train EM score: 0.614000
