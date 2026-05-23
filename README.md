# KE-Coder
## Data-driven Comorbidity Network as Symbolic Constraints: A Knowledge-Enhanced Generative Framework for Low-Resource Clinical Coding

## Overview
This repository contains the official implementation of KE-Coder, a knowledge-enhanced generative framework for automated ICD coding. The framework integrates medical definition constraints ($$K_{def}$$), retrieval-augmented case knowledge ($$K_{rag}$$), and cross-modal structural consistency validation to improve the accuracy and reliability of clinical coding under low-resource conditions.

The model explicitly targets the multi-label classification of 8 core thoracic diseases and leverages the predicted structural comorbidity matrices to align with real-world clinical concurrent patterns.


## Project Structure
├── data/                  # Raw and intermediate processed data (Not tracked by Git)
├── dataset_build/         # Data cleaning, entity matching, and quintuple generation
├── knowledge_base/        # Medical dictionary and definition building modules
├── prompts/               # Prompt templates (System, Baseline, KE-Coder, Few-shot)
├── train/                 # Fine-tuning scripts for LLMs (Vanilla & KE-Coder)
├── inference/             # Inference engine for multi-label disease prediction
├── evaluation/            # Metrics calculation (AUC, F1, PCC, MSE) and comorbidity analysis
├── scripts/               # Bash scripts for automating the pipeline
├── results/               # Output directory for predictions and evaluation reports
├── README.md              # Project documentation
└── requirements.txt       # Python environment dependencies

### Dataset Preparation
Public Datasets Required :All datasets used in this work are publicly available but require official credentialed access and approval via PhysioNet. The underlying datasets include: MIMIC-III, MIMIC-IV, MIMIC-CXR

Due to data privacy and compliance, raw data cannot be directly provided in this repository. You must download them from their official websites.

Data Preprocessing Reference
Preprocessing for MIMIC-III and MIMIC-IV clinical notes follows the standard pipeline widely adopted in ICD coding research. We refer to the preprocessing logic from:
CAML: Mullenbach et al. (2018)
GitHub: https://github.com/jamesmullenbach/caml-mimic
Please preprocess the datasets using the CAML project first, and then place the generated files into data/raw/ and data/processed_data/.
Processed Files
After running our pipeline, the following key intermediate files will be generated:
- final_filtered_8chest_diseases.csv
- final_filtered_data_for_combid.csv
- quintuple_dataset.csv (The final 5-tuple format for LLM instruction tuning)
Environment Setup & Quick Start
0. Install Dependencies
pip install -r requirements.txt
1. Build Knowledge Base & Data Pipeline
Run the preprocessing and knowledge retrieval pipeline to format the clinical notes and inject external medical constraints ($$K_{def}$$ & $$K_{rag}$$).
# Step 1.1: Build standard medical definition dictionary
python knowledge_base/dictionary_builder.py

# Step 1.2: Run data filtering and FAISS-based retrieval augmentation
python dataset_build/preprocess_data.py
python dataset_build/filter_disease_testset.py
python dataset_build/build_knowledge_retrieval.py

# Step 1.3: Construct the final 5-Tuple training dataset
python dataset_build/build_quintuple_dataset.py
Tip: You can also wrap the above commands into a single bash script in scripts/run_data_pipeline.sh for one-click execution.
2. Model Fine-Tuning
Train the models using the generated quintuple_dataset.csv. We provide scripts for both the knowledge-enhanced framework and a pure text baseline.
# Train the baseline model
python train/train_vanilla_baseline.py

# Train the proposed KE-Coder
python train/train_ke_coder.py
3. Run Inference
Generate ICD coding predictions from clinical notes using the trained LLM.
python inference/generate_model_predictions.py
4. Full Evaluation & Comorbidity Validation
Evaluate the predictions using standard classification metrics (Micro/Macro AUC, F1, P@K) and structural comorbidity validation (Pearson Correlation Coefficient [PCC] and Mean Squared Error [MSE]).
python evaluation/run_evaluation.py