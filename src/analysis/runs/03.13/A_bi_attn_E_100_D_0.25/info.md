———————————	
python  -W ignore main.py\
                --attn_layer=bi_attn\
            --embedding_size 100\
            --context_len=500\
            --batch_size 100\
            --output_size 200\
            --output lstm\
            --hidden_size 200\
            --pred_hidden_sz 100\
            --mode=train\
            --dropout 0.25
————————————	
Num params: 3884802

5k iter:
train: 0.67
dev:   0.61
