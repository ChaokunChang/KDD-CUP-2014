# !/bin/bash
python run.py	--train --model LSTMText --optim Adam \
		        --cuda --gpu 0 \
		        --vocab_dir ../../data/vocab --vocab_data vocab_sub10000_balanced.data \
		        --model_dir ../../data/models --model_suffix ModelTest \
		        --epochs 50 --batch_size 128 \
		        --learning_rate 0.001 --dropout 0.2 --weight_decay 0
