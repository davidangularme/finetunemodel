import logging
import random
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model settings
SOURCE_MODEL = "YOUR_BASE_MODEL"  
FINE_TUNED_MODEL = "YOUR_FINETUNE_MODEL"

# Dataset file paths
DATASET_FILES = [
    'data for fine tuning',  
]

# Set random seed
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(42)

def setup_distributed_training():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and torch.cuda.is_available():
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank

from contextlib import contextmanager

@contextmanager
def initialize_model_and_tokenizer(model_name: str, local_rank: int, fine_tuned_model: str = None):
    model = None
    tokenizer = None
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        logger.info("Tokenizer padding_side set to 'left'.")

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            logger.info("Added pad_token to tokenizer.")

        if local_rank != -1 and torch.cuda.is_available():
            device_map = {"": local_rank}
        else:
            device_map = "auto"

        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_int8_training(model)

        # Load fine-tuned weights if available
        if fine_tuned_model and os.path.exists(fine_tuned_model):
            logger.info(f"Loading fine-tuned model from {fine_tuned_model}")
            model = get_peft_model(model, fine_tuned_model)
            model.eval()

        yield model, tokenizer
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()

def load_dataset_from_file(file_path: str) -> Dataset:
    encodings = ['utf-8', 'cp1255', 'ISO-8859-8', 'windows-1255']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = [line.strip() for line in file if line.strip()]
            if lines:
                data = {'text': lines}
                df = pd.DataFrame(data)
                return Dataset.from_pandas(df)
            else:
                logger.warning(f"No text data found in file: {file_path}")
                return None
        except UnicodeDecodeError:
            logger.warning(f"UnicodeDecodeError with encoding {encoding} for file {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path} with encoding {encoding}: {str(e)}")

    raise ValueError(f"All encoding attempts failed for file: {file_path}")

def fine_tune_model(
    source_model: str,
    fine_tuned_model: str,
    dataset_files: list,
    peft_config: LoraConfig,
    training_args: TrainingArguments,
) -> None:
    local_rank = setup_distributed_training()
    is_main_process = local_rank in [-1, 0]

    if is_main_process:
        datasets = []
        for file_path in dataset_files:
            try:
                logger.info(f"Loading dataset from file: {file_path}")
                dataset = load_dataset_from_file(file_path)
                if dataset is not None:
                    datasets.append(dataset)
                    logger.info(f"Successfully loaded {len(dataset)} examples from {file_path}")
                else:
                    logger.warning(f"Unable to load dataset from file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading dataset from {file_path}: {str(e)}")

        if not datasets:
            raise Exception("No datasets were successfully loaded")

        # Concatenate all datasets
        combined_dataset = concatenate_datasets(datasets)
        # Save combined dataset to disk for other processes
        combined_dataset.save_to_disk("combined_dataset")
        logger.info(f"Total combined dataset size: {len(combined_dataset)}")

    # Ensure all processes have the dataset
    if local_rank != -1:
        torch.distributed.barrier()

    # Load the dataset in all processes
    try:
        combined_dataset = Dataset.load_from_disk("combined_dataset")
        # Split into train and validation sets
        train_val_dataset = combined_dataset.train_test_split(test_size=0.1)
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return

    with initialize_model_and_tokenizer(source_model, local_rank) as (model, tokenizer):
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_val_dataset["train"],
            eval_dataset=train_val_dataset["test"],
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,  
            tokenizer=tokenizer,
            args=training_args,
        )

        trainer.train(resume_from_checkpoint=False)
        
        if is_main_process:
            trainer.model.save_pretrained(fine_tuned_model)
            tokenizer.save_pretrained(fine_tuned_model)
            logger.info(f"Model saved to {fine_tuned_model}")


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    # Configure training arguments
    training_args = {
        "output_dir": "./results",
        "overwrite_output_dir": True,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 1,  
        "per_device_eval_batch_size": 1,   
        "gradient_accumulation_steps": 8,  
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 50,
        "learning_rate": 1e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 500,
        "optim": "adamw_torch",
        "evaluation_strategy": "steps",
        "logging_dir": "./logs",
        "save_total_limit": 3,
        "fp16": True,
    }

    # Add distributed training arguments if running in distributed mode
    if local_rank != -1:
        training_args.update({
            "local_rank": local_rank,
            "ddp_backend": "nccl",
            "dataloader_num_workers": 4,
        })

    training_arguments = TrainingArguments(**training_args)

    # Fine-tune the model
    fine_tune_model(
        source_model=SOURCE_MODEL,
        fine_tuned_model=FINE_TUNED_MODEL,
        dataset_files=DATASET_FILES,
        peft_config=peft_config,
        training_args=training_arguments
    )


if __name__ == "__main__":
    main()
