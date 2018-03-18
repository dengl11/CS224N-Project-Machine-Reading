python  -W ignore main.py \
            --attn_layer=bi_attn\
            --embedding_size 100\
            --context_len=300\
            --batch_size 60\
            --output_size 200\
            --learning_rate 2e-4\
            --output double_lstm_dense\
                --pred_layer condition\
            --hidden_size 300\
            --pred_hidden_sz 200\
            --mode=train\
            --dropout 0.22
———————————
[main] Epoch 35, Iter 24500, Train F1 score: 0.757969,                            Train EM score: 0.624000
INFO:root:Epoch 35, Iter 24500, Dev F1 score: 0.667951,                            Dev EM score: 0.518429
[main] Epoch 35, Iter 24500, Dev F1 score: 0.667951,                            Dev EM score: 0.518429
