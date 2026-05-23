import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    df = pd.read_csv("data/processed_data/vanilla_dataset.csv")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")

    tokenized = tokenize_function({"text": df["prompt"].tolist()})

    training_args = TrainingArguments(
        output_dir="train/checkpoints/vanilla_llama3",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )
    trainer.train()
    model.save_pretrained("train/checkpoints/vanilla_final")

if __name__ == "__main__":
    main()