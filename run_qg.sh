python run_qg.py \
    --task e2e_qg \
    --valid_for_qg_only False\
    --model_type t5 \
    --qg_format highlight_qg_format \
    --max_source_length 512 \
    --max_target_length 32 \
    --model_name_or_path t5-small \
    --tokenizer_name_or_path t5_qg_tokenizer \
    --output_dir t5-small-e2e-qg-zino \
    --per_device_train_batch_size: 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train True \
    --evaluate_during_training True \
    --logging_steps 100
