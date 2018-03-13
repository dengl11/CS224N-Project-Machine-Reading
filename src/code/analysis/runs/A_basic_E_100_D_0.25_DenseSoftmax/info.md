————————————	
python  -W ignore main.py\
            --experiment_name=bi_attn\
            --embedding_size 100\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --hidden_size 160\
            --pred_layer dense+softmax\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25

————————————	
Num params: 1636462
                           

Epoch 8, Iter 6500, Train F1 score: 0.661380,Train EM score: 0.516000
Epoch 8, Iter 6500, Dev F1 score: 0.613298, Dev EM score: 0.467424
Epoch 9, Iter 7500, Dev F1 score: 0.619950,                            Dev EM score: 0.470696
Epoch 9, Iter 7500, Train F1 score: 0.688810,                            Train EM score: 0.543000
