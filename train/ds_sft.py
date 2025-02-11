import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset
import transformers
import trl
import deepspeed

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1-fast-7b")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def train():
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    
    # Add DeepSpeed config
    args.deepspeed = "ds_config.json"
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    logging.info(f"Training config: {asdict(config)} | Args: {asdict(args)}")

    kwargs = {"torch_dtype": "auto", "use_cache": False}
    if "70B" in config.model_name:
        kwargs["attn_implementation"] = "flash_attention_2"
    
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    dataset = load_dataset(config.train_file_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)

    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    else:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size

    # Initialize DeepSpeed Plugin
    args.ddp_find_unused_parameters = False
    args.gradient_checkpointing = True

    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
    )

    # Configure optimizer for DeepSpeed
    trainer.accelerator.state.deepspeed_plugin.deepspeed_config['optimizer']['params']['lr'] = args.learning_rate
    trainer.accelerator.state.deepspeed_plugin.deepspeed_config['scheduler']['params']['warmup_max_lr'] = args.learning_rate

    trainer.train()
    
    if trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()