python  -W ignore main.py \
                --attn_layer=bi_attn\
            --embedding_size 100\
            --context_len=400\
            --batch_size 70\
            --output_size 200\
                --learning_rate 8e-4\
            --output double_lstm_dense\
            --hidden_size 200\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25


Epoch 7, Iter 7500, Train F1 score: 0.731655,                            Train EM score: 0.584000
Calculating F1/EM for all examples in dev set...
Calculating F1/EM for 10391 examples in dev set took 143.61 seconds
Epoch 7, Iter 7500, Dev F1 score: 0.661584,                            Dev EM score: 0.514195
