import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Set, Dict, List, Any, Tuple
from collections import Counter

UMLS_DATA_DIR = Path("umls_data")
OUTPUT_DIR = Path("main_task")
FILE_PATHS = {
    "MRCONSO": UMLS_DATA_DIR / "MRCONSO.RRF",
    "MRSTY": UMLS_DATA_DIR / "MRSTY.RRF",
    "MRREL": UMLS_DATA_DIR / "MRREL.RRF",
    "MRDEF": UMLS_DATA_DIR / "MRDEF.RRF",
}
OUTPUT_FILE = OUTPUT_DIR / "medical_knowledge_dict.json"

TARGET_SEMANTIC_TYPES = {
    "Disease or Syndrome", "Sign or Symptom", "Finding", "Pathologic Function",
    "Diagnostic Procedure", "Therapeutic or Preventive Procedure", "Pharmacologic Substance",
    "Body Part, Organ, or Organ Component", "Injury or Poisoning", "Congenital Abnormality",
    "Neoplastic Process", "Mental or Behavioral Dysfunction", "Virus", "Bacterium",
    "Laboratory Procedure", "Medical Device", "Health Care Activity"
}

CHEST_DISEASE_NAME_TO_CUI = {
    "Cardiomegaly": "C0018800",
    "Pneumonia": "C0032285",
    "Edema": "C0013604",
    "Lung Opacity": "C1334969",  
    "Pleural Effusion": "C0032227",
    "Fracture": "C0016658",
    "Enlarged Cardiomediastinum": "C0241721",  
    "Pneumothorax": "C0032326"
}

CHEST_DISEASE_CUI_TO_NAME = {v: k for k, v in CHEST_DISEASE_NAME_TO_CUI.items()}

RELATIONSHIP_MAP = {
    'may_be_treated_by': {'key': 'treatments', 'direction': 'forward'},
    'may_be_prevented_by': {'key': 'treatments', 'direction': 'forward'},
    'may_treat': {'key': 'treatments', 'direction': 'inverse'},
    'may_prevent': {'key': 'treatments', 'direction': 'inverse'},
    'treated_by': {'key': 'treatments', 'direction': 'forward'},
    # 症状/体征/发现
    'has_symptom': {'key': 'symptoms', 'direction': 'forward'},
    'disease_has_associated_finding': {'key': 'symptoms', 'direction': 'forward'},
    'manifestation_of': {'key': 'symptoms', 'direction': 'inverse'},
    'finding_of': {'key': 'symptoms', 'direction': 'inverse'},
    'has_finding': {'key': 'symptoms', 'direction': 'forward'},
    # 诊断
    'may_be_diagnosed_by': {'key': 'diagnostic_tests', 'direction': 'forward'},
    # 风险因素
    'has_risk_factor': {'key': 'risk_factors', 'direction': 'forward'},
    'risk_factor_of': {'key': 'risk_factors', 'direction': 'inverse'},
    # 身体部位
    'affects': {'key': 'affected_body_part', 'direction': 'forward'},
    'location_of': {'key': 'location', 'direction': 'inverse'},
    'site_of': {'key': 'location', 'direction': 'inverse'},
    'has_location': {'key': 'location', 'direction': 'forward'},
    # 组织结构
    'has_part': {'key': 'has_part', 'direction': 'forward'},
    'part_of': {'key': 'part_of', 'direction': 'forward'},
    # 因果关系
    'causes': {'key': 'causes', 'direction': 'forward'},
    'is_caused_by': {'key': 'causes', 'direction': 'inverse'},
    'complicates': {'key': 'complications', 'direction': 'forward'},
    'is_complication_of': {'key': 'complications', 'direction': 'inverse'},
    'due_to': {'key': 'causes', 'direction': 'inverse'},
    # 分类关系
    'isa': {'key': 'is_a', 'direction': 'forward'},
    'inverse_isa': {'key': 'is_a', 'direction': 'inverse'},
    # 其他
    'associated_with': {'key': 'associated_with', 'direction': 'forward'},
}
def parse_rrf_line(line: str) -> list:
    return line.strip().split('|')
def get_cui_whitelist_and_types() -> Tuple[Set[str], Dict[str, Set[str]]]:
    valid_cui_set = set()
    cui_to_types_map = defaultdict(set)
    with open(FILE_PATHS["MRSTY"], 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Pass 1/5: Filtering CUIs by Semantic Type"):
            parts = parse_rrf_line(line)
            if len(parts) < 4: continue
            cui, sty = parts[0], parts[3]
            if sty in TARGET_SEMANTIC_TYPES:
                valid_cui_set.add(cui)
                cui_to_types_map[cui].add(sty)
    print(f"Found {len(valid_cui_set)} CUIs matching the {len(TARGET_SEMANTIC_TYPES)} target semantic types.")
    return valid_cui_set, cui_to_types_map

def process_names_and_codes(valid_cui_set: Set[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    knowledge_dict = defaultdict(lambda: {
        "names": defaultdict(set),
        "coding": defaultdict(list)
    })
    cui_to_preferred_name = {}

    with open(FILE_PATHS["MRCONSO"], 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Pass 2/5: Processing Names & Codes"):
            parts = parse_rrf_line(line)
            if len(parts) < 15: continue
            cui, lat, ts, tty, code, sab, string = parts[0], parts[1], parts[2], parts[12], parts[13], parts[11], parts[
                14]

            if cui not in valid_cui_set or lat != "ENG":
                continue
            if tty == 'PT' and ts == 'P':
                knowledge_dict[cui]["names"]["preferred"] = string
                cui_to_preferred_name[cui] = string
            elif tty == 'SY':
                knowledge_dict[cui]["names"]["synonyms"].add(string)
            elif tty == 'AB':
                knowledge_dict[cui]["names"]["abbreviations"].add(string)
            else:
                knowledge_dict[cui]["names"]["alternative_names"].add(string)
            if code and sab in {'ICD9CM', 'ICD10CM', "ICD9", "ICD10"}:
                code_type = "icd_9_cm" if sab == 'ICD9CM' else "icd_10_cm"
                knowledge_dict[cui]["coding"][code_type].append({"code": code, "description": string})

    print(f"Processed names and codes for {len(knowledge_dict)} CUIs.")
    return knowledge_dict, cui_to_preferred_name

def add_definitions(knowledge_dict: Dict[str, Any], valid_cui_set: Set[str]):
    with open(FILE_PATHS["MRDEF"], 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Pass 3/5: Adding Definitions"):
            parts = parse_rrf_line(line)
            if len(parts) < 6: continue
            cui, definition = parts[0], parts[5]
            if cui in valid_cui_set:
                if "definition" not in knowledge_dict[cui]:
                    knowledge_dict[cui]["definition"] = set()
                knowledge_dict[cui]["definition"].add(definition)
    print("Finished adding definitions.")

def add_relationships(knowledge_dict: Dict[str, Any], valid_cui_set: Set[str], cui_to_preferred_name: Dict[str, str]):
    added_relationships = defaultdict(set)
    with open(FILE_PATHS["MRREL"], 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Pass 4/5: Building Relationships"):
            parts = parse_rrf_line(line)
            if len(parts) < 11: continue
            cui1, rela, cui2 = parts[0], parts[7], parts[4]
            if cui1 not in valid_cui_set or cui2 not in valid_cui_set:
                continue
            relation_info = RELATIONSHIP_MAP.get(rela)
            if relation_info:
                key = relation_info['key']
                source_cui, target_cui = (cui1, cui2) if relation_info['direction'] == 'forward' else (cui2, cui1)
            else:
                key = rela
                source_cui, target_cui = cui1, cui2
            relationship_tuple = (source_cui, key, target_cui)
            if relationship_tuple in added_relationships[source_cui]:
                continue
            if source_cui in knowledge_dict:
                if "relationships" not in knowledge_dict[source_cui]:
                    knowledge_dict[source_cui]["relationships"] = defaultdict(list)

                target_name = cui_to_preferred_name.get(target_cui, "Unknown Concept")
                knowledge_dict[source_cui]["relationships"][key].append({"cui": target_cui, "name": target_name})

                added_relationships[source_cui].add(relationship_tuple)
    print("Finished building relationships.")

def finalize_and_save(knowledge_dict: Dict[str, Any], cui_to_types_map: Dict[str, Set[str]]):
    final_dict = {}
    semantic_type_distribution = Counter()

    for cui, data in tqdm(knowledge_dict.items(), desc="Pass 5/5: Finalizing and QC"):
        if "preferred" not in data["names"]:
            other_names = data["names"].get("alternative_names") or data["names"].get("synonyms")
            if not other_names: continue  # 如果没有任何名称，则丢弃
            data["names"]["preferred"] = list(other_names)[0]

        final_record = {
            "preferred_name": data["names"]["preferred"],
            "semantic_types": sorted(list(cui_to_types_map[cui])),
            "definition": "\n".join(sorted(list(data.get("definition", {"No definition found."})))),
            "names": {
                "preferred": data["names"]["preferred"],
                "synonyms": sorted(list(data["names"].get("synonyms", set()))),
                "abbreviations": sorted(list(data["names"].get("abbreviations", set()))),
                "alternative_names": sorted(list(data["names"].get("alternative_names", set())))
            },
            "coding": {
                "icd_9_cm": data["coding"].get("icd_9_cm", []),
                "icd_10_cm": data["coding"].get("icd_10_cm", [])
            },
            "relationships": data.get("relationships", {})
        }

        if cui in CHEST_DISEASE_CUI_TO_NAME:
            final_record["chest_disease_info"] = {
                "is_core_chest_disease": True,
                "core_disease_tag": CHEST_DISEASE_CUI_TO_NAME[cui],
                "primary_organ_system": "respiratory"
            }
        for sty in final_record["semantic_types"]:
            semantic_type_distribution[sty] += 1

        final_dict[cui] = final_record

    print("\n--- Quality Control & Statistics ---")
    print(f"Total concepts in final dictionary: {len(final_dict)}")
    print("Distribution of concepts by semantic type:")
    for sty, count in semantic_type_distribution.most_common():
        print(f"  - {sty}: {count}")

    relationship_distribution = Counter()
    for cui_data in final_dict.values():
        for rel_type, rel_list in cui_data.get("relationships", {}).items():
            relationship_distribution[rel_type] += len(rel_list)
    print("\nDistribution of relationships by type:")
    for rel_type, count in relationship_distribution.most_common(20):  # 仅打印前20个
        print(f"  - {rel_type}: {count}")

    print(f"\nWriting final knowledge_base dictionary to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, indent=2, ensure_ascii=False)
    print("Write complete.")


if __name__ == "__main__":
    for path in FILE_PATHS.values():
        if not path.exists():
            print(f"FATAL: Missing required file: {path}")
            exit(1)
    valid_cuis, cui_types = get_cui_whitelist_and_types()
    knowledge_base, cui_to_name_map = process_names_and_codes(valid_cuis)
    add_definitions(knowledge_base, valid_cuis)
    add_relationships(knowledge_base, valid_cuis, cui_to_name_map)
    finalize_and_save(knowledge_base, cui_types)

    print("\nMedical knowledge_base dictionary reconstruction complete.")
