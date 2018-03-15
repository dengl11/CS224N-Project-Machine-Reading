```
python  -W ignore main.py \
            --attn_layer=bi_attn\
            --embedding_size 100\
            --context_len=300\
            --batch_size 80\
            --output_size 200\
            --learning_rate 2e-4\
            --output double_lstm_dense\
                --pred_layer condition\
            --hidden_size 200\
            --pred_hidden_sz 200\
            --mode=train\
            --dropout 0.25
```

Epoch 13, Iter 13000, Train F1 score: 0.761232,                            Train EM score: 0.615000
Epoch 13, Iter 13000, Dev F1 score: 0.667412,                            Dev EM score: 0.514676
