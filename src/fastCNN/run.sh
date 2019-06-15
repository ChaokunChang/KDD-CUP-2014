# !/bin/bash
python run.py	--train --model CNNText --optim Adam \
		        --cuda --gpu 1 \
		        --vocab_dir ../../data/vocab --vocab_data vocab_oversampling.data \
		        --model_dir ../../data/models --model_suffix oversampling \
		        --epochs 10 --batch_size 320 \
		        --learning_rate 0.001 --dropout 0.2 --weight_decay 0
