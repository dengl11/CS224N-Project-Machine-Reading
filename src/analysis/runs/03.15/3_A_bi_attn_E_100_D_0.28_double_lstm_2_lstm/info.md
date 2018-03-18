python  -W ignore main.py \
            --mode=train\
            --context_len=300\
            --embedding_size 100\
            --attn_layer=bi_attn\
            --batch_size 60\
                --output double_lstm_dense\
            --output_size 200\
                --pred_layer condition\
            --hidden_size 300\
            --pred_hidden_sz 200\
            --learning_rate 2e-4\
            --dropout 0.28


[main] Epoch 29, Iter 40000, Train F1 score: 0.733352,                            Train EM score: 0.576000
INFO:root:Epoch 29, Iter 40000, Dev F1 score: 0.663362,                            Dev EM score: 0.507507
[main] Epoch 29, Iter 40000, Dev F1 score: 0.663362,                            Dev EM score: 0.507507
