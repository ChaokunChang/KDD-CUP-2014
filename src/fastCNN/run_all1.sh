# !/bin/bash
text_var="essay"

target_var="at_least_1_teacher_referred_donor"
m_name=$text_var"_"$target_var
python run.py	--prepare --train --model CNNText --optim Adam \
		        --cuda --gpu 1 \
				--text_var $text_var --target_var $target_var \
		        --vocab_dir ../../data/vocab --vocab_data vocab_$m_name.data \
		        --model_dir ../../data/models --model_suffix $m_name \
		        --epochs 4 --batch_size 128 \
		        --learning_rate 0.001 --dropout 0.3 --weight_decay 0
