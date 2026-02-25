"""Fine-tune a model with QLoRA using Unsloth."""

import json
from pathlib import Path

import yaml
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

CONFIG_PATH = Path("configs/train_config.yaml")


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_dataset_from_jsonl(path):
    """Load a JSONL file of chat messages into a HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def formatting_func(example, tokenizer):
    """Apply the chat template to a single example."""
    return [tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )]


def train():
    cfg = load_config()

    print(f"Loading model: {cfg['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name"],
        max_seq_length=cfg["model"]["max_seq_length"],
        dtype=cfg["model"]["dtype"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        use_gradient_checkpointing=cfg["lora"]["use_gradient_checkpointing"],
    )

    print("Loading datasets...")
    train_dataset = load_dataset_from_jsonl(cfg["data"]["train_path"])
    val_dataset = load_dataset_from_jsonl(cfg["data"]["val_path"])

    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val:   {len(val_dataset)} examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=cfg["output"]["dir"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            warmup_steps=cfg["training"]["warmup_steps"],
            optim=cfg["training"]["optim"],
            weight_decay=cfg["training"]["weight_decay"],
            fp16=cfg["training"]["fp16"],
            bf16=cfg["training"]["bf16"],
            logging_steps=cfg["training"]["logging_steps"],
            save_strategy=cfg["training"]["save_strategy"],
            seed=cfg["training"]["seed"],
            max_seq_length=cfg["model"]["max_seq_length"],
            dataset_text_field="text",
            eval_strategy="epoch",
        ),
        formatting_func=lambda example: formatting_func(example, tokenizer),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {cfg['output']['dir']}...")
    model.save_pretrained(cfg["output"]["dir"])
    tokenizer.save_pretrained(cfg["output"]["dir"])

    print("Done!")


if __name__ == "__main__":
    train()
