from run_qg import run_qg

args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "t5_qg_tokenizer",
    "output_dir": "t5-small-e2e-qg-zino",
    "train_file_path": "data/train_data.pt",
    "valid_file_path": "data/valid_data.pt",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
    "seed": 42,
    "do_train": True,
    "evaluate_during_training": True,
    "logging_steps": 100,
    "save_total_limit": 10,
    "overwrite_output_dir": True
}

run_qg(args_dict)
