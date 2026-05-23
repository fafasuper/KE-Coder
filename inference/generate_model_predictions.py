import pandas as pd
import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== 路径配置 ====================
BASE = Path(__file__).parent.parent
PROMPTS_DIR = BASE / "prompts"
TEST_DATA_PATH = BASE / "data/processed_data/final_filtered_8chest_diseases.csv"
OUTPUT_DIR = BASE / "results/model_predictions"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISEASE_NAMES = [
    "Cardiomegaly",
    "Pneumothorax",
    "Pleural Effusion",
    "Edema",
    "Lung Opacity",
    "Fracture",
    "Atelectasis",
    "Enlarged Cardiomediastinum"
]

MODEL_PROMPT_MAP = {
    "vanilla": {
        "model_path": BASE / "train/checkpoints/vanilla_llama3",
        "prompt_file": "vanilla_baseline.txt"
    },
    "ke_coder": {
        "model_path": BASE / "train/checkpoints/ke_coder",
        "prompt_file": "ke-coder.txt"
    },
    "few_shot": {
        "model_path": BASE / "train/checkpoints/vanilla_llama3",
        "prompt_file": "few_shot.txt"
    },
    "disease_classification": {
        "model_path": BASE / "train/checkpoints/vanilla_llama3",
        "prompt_file": "disease_label_classification.txt"
    }
}
def load_prompt(prompt_name):
    with open(PROMPTS_DIR / "system_prompt.txt", "r", encoding="utf-8") as f:
        system = f.read().strip()

    with open(PROMPTS_DIR / prompt_name, "r", encoding="utf-8") as f:
        template = f.read().strip()

    return system, template

def build_full_prompt(template, system, text, k_def=None, k_rag=None):
    return template.format(
        system_prompt=system,
        clinical_text=text,
        knowledge_definitions=k_def or "",
        retrieved_exemplars=k_rag or ""
    )

def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model

def predict(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_all_models():
    df = pd.read_csv(TEST_DATA_PATH)
    results = []

    for model_name, cfg in MODEL_PROMPT_MAP.items():
        print(f"\n=== Running {model_name} ===")

        system_prompt, template = load_prompt(cfg["prompt_file"])
        tokenizer, model = load_model_tokenizer(cfg["model_path"])

        model_preds = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            prompt = build_full_prompt(
                template=template,
                system=system_prompt,
                text=row["discharge_summary_text"],
                k_def=row.get("knowledge_def", ""),
                k_rag=row.get("rag_context", "")
            )
            pred = predict(model, tokenizer, prompt)
            model_preds.append(pred)

        df[f"{model_name}_pred"] = model_preds

    df.to_csv(OUTPUT_DIR / "all_model_predictions.csv", index=False)
    print(f"\n✅ 所有模型预测已保存：{OUTPUT_DIR / 'all_model_predictions.csv'}")

if __name__ == "__main__":
    run_all_models()