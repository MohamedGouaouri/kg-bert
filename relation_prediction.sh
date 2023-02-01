#!/bin/bash
python3 run_bert_relation_prediction.py \
--task_name kg  \
--do_train  \
--do_eval \
--data_dir ./data/linkdin \
--bert_model bert-base-cased    \
--max_seq_length 25 \
--train_batch_size 32 \
--learning_rate 5e-3    \
--num_train_epochs 10   \
--output_dir ./output_Linkedin/  \
--gradient_accumulation_steps 1 \
--eval_batch_size 512