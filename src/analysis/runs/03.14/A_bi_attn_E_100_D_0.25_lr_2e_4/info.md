
```  
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
            --dropout 0.25

```  

Epoch 12, Iter 14500, Train F1 score: 0.795666,                            Train EM score: 0.659000
Epoch 12, Iter 14500, Dev F1 score: 0.685293,                            Dev EM score: 0.539409
