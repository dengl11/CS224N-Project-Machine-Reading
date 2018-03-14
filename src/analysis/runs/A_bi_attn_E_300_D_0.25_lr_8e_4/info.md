python  -W ignore main.py\
        --attn_layer=bi_attn\
        --embedding_size 300\
        --context_len=500\
        --batch_size 50\
        --output_size 200\
        --learning_rate 8e-4\
        --output lstm\
        --pred_layer dense+softmax\
        --hidden_size 200\
        --pred_hidden_sz 200\
        --mode=train\
        --dropout 0.25
[main] Epoch 8, Iter 12500, Train F1 score: 0.803237,                            Train EM score: 0.651000
[main] Epoch 8, Iter 12500, Dev F1 score: 0.651143,                            Dev EM score: 0.497690
