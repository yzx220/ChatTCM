SET PRE_SEQ_LEN=128
SET LR=2e-3

python main.py ^
    --do_train ^
    --train_file MED/train2.json ^
    --validation_file MED/dev1.json ^
    --preprocessing_num_workers 10 ^
    --prompt_column query ^
    --response_column response ^
    --overwrite_cache  False^
    --model_name_or_path D:\work\ChatGLM2-6B\model ^
    --output_dir output/adgen-chatglm2-6b-pt-%PRE_SEQ_LEN%-%LR% ^
    --overwrite_output_dir ^
    --max_source_length 512 ^
    --max_target_length 128 ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 4 ^
    --gradient_accumulation_steps 16 ^
    --predict_with_generate ^
    --max_steps 3000 ^
    --logging_steps 1 ^
    --save_steps 500 ^
    --learning_rate %LR% ^
    --pre_seq_len %PRE_SEQ_LEN% ^
    --quantization_bit 4

