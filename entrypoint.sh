#!/bin/bash

python run_bert_triple_classifier.py --task_name kg --do_train --do_eval --data_dir ./data/linkedin/ --bert_model bert-base-uncased --max_seq_length 60 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 200 --output_dir ./output_Linkedin/  --gradient_accumulation_steps 1