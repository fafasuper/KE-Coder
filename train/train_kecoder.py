import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import os
import ast

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

    df = pd.read_csv("data/processed_data/tuple_dataset.csv")
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        sys_prompt = f.read().strip()
    with open("prompts/ke_coder.txt", "r", encoding="utf-8") as f:
        template = f.read().strip()
    CORE_DISEASES = [
        "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Edema",
        "Lung Opacity", "Fracture", "Atelectasis", "Enlarged Cardiomediastinum"
    ]
    training_texts = []
    for _, row in df.iterrows():
        input_prompt = template.format(
            system_prompt=sys_prompt,
            knowledge_definitions=row['K_def'] if pd.notna(row['K_def']) else "",
            retrieved_exemplars=row['K_rag'] if pd.notna(row['K_rag']) else "",
            clinical_text=row['D']
        )
        y_list = ast.literal_eval(row['Y'])
        present_diseases = [CORE_DISEASES[i] for i, val in enumerate(y_list) if val == 1]
        target_output = ", ".join(present_diseases) if present_diseases else "No Finding"
        full_text = f"{input_prompt}\n\n{target_output}{tokenizer.eos_token}"
        training_texts.append(full_text)

    def tokenize_function(text_list):
        tokenized = tokenizer(
            text_list,
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    from datasets import Dataset
    hf_dataset = Dataset.from_dict({"text": training_texts})
    tokenized_dataset = hf_dataset.map(lambda x: tokenize_function(x["text"]), batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir="train/checkpoints/ke_coder",
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
    model.save_pretrained("train/checkpoints/ke_coder_final")

if __name__ == "__main__":
    main()