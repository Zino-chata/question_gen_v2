from run_qg import run_qg

'''
args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "t5_qg_tokenizer",
    "output_dir": "t5-small-e2e-qg-zino",
    "train_file_path": "data/train_data.pt",
    "valid_file_path": "data/valid_data.pt",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
    "seed": 42,
    "do_train": True,
    "evaluate_during_training": True,
    "logging_steps": 100
}
'''

args_dict = {
    "model_name_or_path": "t5-small",
    "model_type": "t5",
    "tokenizer_name_or_path": "t5_qg_tokenizer",
    "output_dir": "t5-small-e2e-qg-zino",
    "train_file_path": "data/train_data.pt",
    "valid_file_path": "data/valid_data.pt",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_train_epochs": 10,
    "seed": 42,
    "do_train": True,
    "evaluate_during_training": True,
    "logging_steps": 100,
    "task": "e2e_qg",
    "valid_for_qg_only": False,
    "qg_format": "highlight_qg_format",
    "max_source_length": 512,
    "max_target_length": 32,
    "train_file_name": "train_data.pt",
    "valid_file_name": "valid_data.pt"
}

run_qg(args_dict)
