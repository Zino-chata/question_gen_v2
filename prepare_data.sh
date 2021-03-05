python prepare_data.py \
    --task e2e_qg \
    --valid_for_qg_only False\
    --model_type t5 \
    --qg_format highlight_qg_format \
    --max_source_length 512 \
    --max_target_length 32 \
    --train_file_name train_data.pt \
    --valid_file_name valid_data.pt