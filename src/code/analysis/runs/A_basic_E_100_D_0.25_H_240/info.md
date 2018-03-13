————————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 100\
            --hidden_size 240\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25
————————————	
Num params: 2349442

Epoch 10, Iter 8000, Train F1 score: 0.723348, Train EM score: 0.580000
Epoch 10, Iter 8000, Dev F1 score: 0.610946,                            Dev EM score: 0.461650
