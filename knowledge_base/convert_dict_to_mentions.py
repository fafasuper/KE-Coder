import json
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = Path("medical_knowledge_dict.json")
OUTPUT_DIR = Path("umls_data")
OUTPUT_FILE = OUTPUT_DIR / "umls_mentions.jsonl"

def convert_dict_to_mentions():
    print(f"Loading medical knowledge dictionary from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            knowledge_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        print("Please ensure you have run dictionary_builder.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_mentions_extracted = 0
    print(f"Converting nested dictionary to flat JSONL format...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for cui, data in tqdm(knowledge_dict.items(), desc="Extracting Mentions"):
            entity_name = data.get("preferred_name", "Unknown Concept")
            semantic_types = data.get("semantic_types", [])
            entity_type = semantic_types[0] if semantic_types else "Unknown"
            all_mentions = set()
            names_dict = data.get("names", {})
            if names_dict.get("preferred"):
                all_mentions.add(names_dict["preferred"])
            for key in ["synonyms", "abbreviations", "alternative_names"]:
                for name in names_dict.get(key, []):
                    if name and isinstance(name, str):
                        all_mentions.add(name.strip())

            for mention in all_mentions:
                record = {
                    "cui": cui,
                    "mention": mention,
                    "type": entity_type,
                    "entity_name": entity_name,
                    # 计算单词数量（按空格简单切分）
                    "word_count": len(mention.split())
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                total_mentions_extracted += 1

    print("\n--- Conversion Complete ---")
    print(f"Total CUIs processed: {len(knowledge_dict)}")
    print(f"Total flattened mentions generated: {total_mentions_extracted}")
    print(f"Mentions file saved successfully to: {OUTPUT_FILE}")


if __name__ == "__main__":
    convert_dict_to_mentions()