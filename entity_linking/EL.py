import json
import re
import random
import logging
from pathlib import Path
from tqdm import tqdm
from ahocorasick import Automaton
from collections import defaultdict
import html
from typing import Dict, List, Any, Tuple

def load_entity_dict(entity_file):
    print(f"Loading entity dictionary from {entity_file}...")
    phrase_to_cui = defaultdict(list)
    entity_dict = {}

    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading entity dictionary"):
            entity = json.loads(line)
            cui = entity['cui']
            mention = entity['mention'].strip().lower()
            entity_type = entity['type']
            entity_name = entity['entity_name']
            word_count = entity['word_count']

            mention = html.unescape(mention)
            mention = re.sub(r'[|&#]', ' ', mention)
            mention = re.sub(r'\s+', ' ', mention).strip()

            if cui not in entity_dict:
                entity_dict[cui] = {
                    "type": entity_type,
                    "name": entity_name,
                    "word_count": word_count
                }

            phrase_to_cui[mention].append(cui)

    print(f"Entity dictionary loaded with {len(entity_dict)} entities and {len(phrase_to_cui)} phrases.")
    return entity_dict, phrase_to_cui

def build_automaton(phrase_to_cui):
    A = Automaton()
    sorted_phrases = sorted(phrase_to_cui.items(), key=lambda x: len(x[0]), reverse=True)
    for phrase, cuis in sorted_phrases:
        A.add_word(phrase, (phrase, cuis))
    A.make_automaton()
    return A

def is_valid_boundary(text, start, end):
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def is_common_word(word):
    common_words = set(
        ['a', 'is', 'are', 'was', 'an', 'the', 'were', 'be', 'been', 'of', 'in', 'to', 'for', 'with', 'by', 'at', 'on'])
    return word.lower() in common_words


def preprocess_text(text):
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove or replace special characters
    text = re.sub(r'[|&#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def annotate_text(text, automaton, entity_dict):
    clean_text = preprocess_text(text)
    text_lower = clean_text.lower()
    matches = []

    for end_index, (phrase, cuis) in automaton.iter(text_lower):
        start_index = end_index - len(phrase) + 1
        if is_valid_boundary(text_lower, start_index, end_index + 1) and not is_common_word(phrase):
            matches.append((start_index, end_index + 1, phrase, cuis))

    matches.sort(key=lambda x: (-len(x[2]), x[0]))

    filtered_matches = []
    covered_ranges = set()
    linked_mentions = set()
    for start, end, phrase, cuis in matches:
        if not any((start >= r[0] and end <= r[1]) for r in covered_ranges):
            if phrase not in linked_mentions:
                filtered_matches.append((start, end, phrase, cuis))
                covered_ranges.add((start, end))
                linked_mentions.add(phrase)

    annotations = []
    for start, end, phrase, cuis in filtered_matches:
        for cui in cuis:
            annotation = {
                "mention": clean_text[start:end],
                "cui": cui,
                "start": start,
                "end": end,
                "type": entity_dict[cui]['type'],
                "entity_name": entity_dict[cui]['name']
            }
            if annotation not in annotations:
                annotations.append(annotation)
    return clean_text, annotations


def process_paragraphs(input_file, output_file, automaton, entity_dict, batch_size=1000):
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    processed_count = 0
    batch = []
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as in_file, \
            open(output_file, 'w', encoding='utf-8') as out_file:
        pbar = tqdm(total=total_lines, desc="Processing paragraphs", unit="lines")
        for line in in_file:
            try:
                data = json.loads(line)
                paragraph = data.get('text', '').strip()
                if paragraph:
                    clean_text, annotations = annotate_text(paragraph, automaton, entity_dict)
                    result = {
                        "original_text": paragraph,
                        "text": clean_text,
                        "annotations": annotations
                    }
                    batch.append(result)
                processed_count += 1
                if len(batch) >= batch_size:
                    save_batch(batch, out_file)
                    batch = []
                    pbar.update(batch_size)
            except Exception as e:
                print(f"Error processing line {processed_count + 1}: {str(e)}")

        if batch:
            save_batch(batch, out_file)
            pbar.update(len(batch))

        pbar.close()
        print(f"\nProcessed {processed_count} paragraphs. Results saved to {output_file}")

def save_batch(batch, file):
    for result in batch:
        json.dump(result, file, ensure_ascii=False)
        file.write('\n')
    file.flush()

def test_random_samples(input_file, automaton, entity_dict, num_samples=10):
    print(f"Testing {num_samples} random samples...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sample_lines = random.sample(lines, num_samples)
    for i, line in enumerate(sample_lines, 1):
        try:
            data = json.loads(line)
            paragraph = data.get('text', '').strip()
            if paragraph:
                print(f"\nSample {i}:")
                print("Original text:", paragraph[:100] + "..." if len(paragraph) > 100 else paragraph)
                annotations = annotate_text(paragraph, automaton, entity_dict)
                print("Annotations:")
                for ann in annotations:
                    print(f"  - {ann['mention']} (CUI: {ann['cui']}, Type: {ann['type']})")
                print(f"Total annotations: {len(annotations)}")
            else:
                print(f"\nSample {i}: Empty paragraph")
        except Exception as e:
            print(f"\nError processing sample {i}: {str(e)}")

def main():
    entity_file = '../knowledge_base/umls_data/umls_mentions.jsonl'
    input_file = '../data_preprocessing/test_segments.jsonl'
    output_file = 'entity_disease_prediction_results.jsonl'
    batch_size = 1000

    entity_dict, phrase_to_cui = load_entity_dict(entity_file)
    automaton = build_automaton(phrase_to_cui)
    process_paragraphs(input_file, output_file, automaton, entity_dict, batch_size)

if __name__ == "__main__":
    main()