set PRE_SEQ_LEN=128
set CHECKPOINT=adgen-chatglm2-6b-pt-LR3
set STEP=2000
set NUM_GPUS=1

python main.py ^
    --do_predict ^
    --validation_file MED\\dev1.json ^
    --test_file MED\\dev1.json ^
    --overwrite_cache ^
    --prompt_column query ^
    --response_column response ^
    --model_name_or_path D:\\work\\ChatGLM2-6B\\model ^
    --ptuning_checkpoint D:\\work\\ChatGLM2-6B\\ptuning\\output\\adgen-chatglm2-6b-pt-128-2e-3\\checkpoint-3000  ^
    --output_dir output/CHECKPOINT-3000 ^
    --overwrite_output_dir ^
    --max_source_length 256 ^
    --max_target_length 128 ^
    --per_device_eval_batch_size 1 ^
    --predict_with_generate ^
    --pre_seq_len %PRE_SEQ_LEN% ^
    --quantization_bit 4
