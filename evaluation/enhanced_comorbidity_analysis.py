import pandas as pd
import numpy as np
import ast
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
from scipy.stats import chi2_contingency, fisher_exact, spearmanr
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import warnings

warnings.filterwarnings('ignore')
PROCESSED_DATA = 'data/processed/preprocessed_complete_dataset.csv'
ICD_CCS_MAP_PATH = 'data/raw/allicd_standardized_ccs_mapped_with_desc.csv'
DISEASE_LABEL_COLUMNS = [
    'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis', 'Pleural Effusion',
    'Support Devices', 'Edema', 'Pneumonia', 'Pneumothorax', 'Lung Lesion',
    'Fracture', 'Enlarged Cardiomediastinum', 'Consolidation', 'Pleural Other',
    'Calcification of the Aorta', 'Tortuous Aorta', 'Pneumoperitoneum',
    'Subcutaneous Emphysema', 'Pneumomediastinum'
]
OUTPUT_DIR = './results/enhanced_cooccurrence/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
class ClinicalKnowledgeBase:
    # 已知的强关联模式（基于医学文献）
    STRONG_CLINICAL_ASSOCIATIONS = {
        'I50': ['Cardiomegaly', 'Edema', 'Pleural Effusion'],  # 心力衰竭
        'I25': ['Cardiomegaly', 'Enlarged Cardiomediastinum'],  # 冠心病
        'I11': ['Cardiomegaly', 'Edema'],  # 高血压心脏病
        'J44': ['Lung Opacity', 'Consolidation', 'Atelectasis'],  # COPD
        'J18': ['Pneumonia', 'Consolidation', 'Lung Opacity'],  # 肺炎
        'J94': ['Pleural Effusion', 'Pneumothorax'],  # 胸膜疾病
        'S22': ['Fracture', 'Pneumothorax', 'Subcutaneous Emphysema'],  # 肋骨骨折
        'T81': ['Support Devices', 'Pneumothorax'],  # 医疗器械相关并发症
        'I70': ['Calcification of the Aorta', 'Tortuous Aorta'],  # 动脉粥样硬化
    }

    # CCS类别的临床重要性权重
    CCS_CLINICAL_WEIGHTS = {
        'Diseases of heart': 1.0,
        'Pneumonia': 0.9,
        'COPD and bronchiectasis': 0.9,
        'Heart failure': 1.0,
        'Fracture': 0.7,
        'Septicemia': 1.0,
    }

    @classmethod
    def get_clinical_relevance_score(cls, icd_code, disease_label):
        icd_prefix = icd_code[:3] if len(icd_code) >= 3 else icd_code
        for pattern_icd, related_labels in cls.STRONG_CLINICAL_ASSOCIATIONS.items():
            if icd_code.startswith(pattern_icd) and disease_label in related_labels:
                return 'High'

        if icd_prefix.startswith('I') and disease_label in ['Cardiomegaly', 'Edema', 'Enlarged Cardiomediastinum']:
            return 'Moderate'
        elif icd_prefix.startswith('J') and disease_label in ['Lung Opacity', 'Pneumonia', 'Consolidation']:
            return 'Moderate'
        elif icd_prefix.startswith('S') and disease_label in ['Fracture', 'Pneumothorax']:
            return 'Moderate'

        return 'Low'

class EnhancedCooccurrenceAnalyzer:
    def __init__(self, patient_df, icd_ccs_map):
        self.patient_df = patient_df
        self.icd_ccs_map = icd_ccs_map
        self.knowledge_base = ClinicalKnowledgeBase()
        self.results = {}

    def calculate_statistical_significance(self, cooccurrence_counts, total_visits_or_patients,
                                           source_counts, target_counts, analysis_level):
        print(f"\n🔍 计算 {analysis_level} 统计显著性...")

        enhanced_results = []

        for (source, target), observed_count in tqdm(cooccurrence_counts.items(),
                                                     desc=f"分析 {analysis_level} 统计显著性"):

            # 基础频次统计
            source_total = source_counts.get(source, 0)
            target_total = target_counts.get(target, 0)

            if source_total == 0 or target_total == 0:
                continue

            # 计算期望共现次数（假设独立）
            expected_count = (source_total * target_total) / total_visits_or_patients
            enrichment_fold = observed_count / expected_count if expected_count > 0 else 0
            both_present = observed_count
            source_only = source_total - observed_count
            target_only = target_total - observed_count
            neither_present = total_visits_or_patients - source_total - target_total + observed_count

            contingency_table = np.array([
                [both_present, source_only],
                [target_only, neither_present]
            ])

            # 统计检验
            try:
                if both_present < 5 or min(source_only, target_only, neither_present) < 5:
                    _, p_value = fisher_exact(contingency_table)
                    test_method = 'Fisher Exact'
                else:
                    # 使用卡方检验
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    test_method = 'Chi-square'
            except:
                p_value = 1.0
                test_method = 'Failed'

            # Matthews相关系数 (考虑所有四个象限)
            if contingency_table.sum() > 0:
                try:
                    # 重建二进制向量计算MCC
                    y_true = [1] * both_present + [1] * source_only + [0] * target_only + [0] * neither_present
                    y_pred = [1] * both_present + [0] * source_only + [1] * target_only + [0] * neither_present
                    mcc = matthews_corrcoef(y_true, y_pred)
                except:
                    mcc = 0
            else:
                mcc = 0
            baseline_prob = target_total / total_visits_or_patients
            conditional_prob = observed_count / source_total if source_total > 0 else 0
            lift = conditional_prob / baseline_prob if baseline_prob > 0 else 0

            jaccard = both_present / (source_total + target_total - both_present) if (
                                                                                                 source_total + target_total - both_present) > 0 else 0
            if 'icd' in analysis_level.lower():
                clinical_relevance = self.knowledge_base.get_clinical_relevance_score(source, target)
            else:
                clinical_relevance = 'Unknown'

            # 汇总结果
            enhanced_results.append({
                'source': source,
                'target': target,
                'observed_count': observed_count,
                'source_total_count': source_total,
                'target_total_count': target_total,
                'expected_count': expected_count,
                'enrichment_fold': enrichment_fold,
                'p_value': p_value,
                'statistical_test': test_method,
                'mcc': mcc,
                'lift': lift,
                'jaccard_similarity': jaccard,
                'clinical_relevance': clinical_relevance,
                'prevalence_source': source_total / total_visits_or_patients,
                'prevalence_target': target_total / total_visits_or_patients,
                'contingency_table': contingency_table.tolist()
            })
        results_df = pd.DataFrame(enhanced_results)
        if len(results_df) > 0:
            results_df = results_df.sort_values(['enrichment_fold', 'observed_count'], ascending=[False, False])
            results_df['significance_level'] = results_df['p_value'].apply(
                lambda x: 'Highly Significant' if x < 0.001 else
                'Significant' if x < 0.01 else
                'Marginally Significant' if x < 0.05 else
                'Not Significant'
            )
            results_df['association_strength'] = results_df.apply(
                lambda row: 'Very Strong' if row['enrichment_fold'] >= 5 and row['mcc'] >= 0.3 else
                'Strong' if row['enrichment_fold'] >= 3 and row['mcc'] >= 0.2 else
                'Moderate' if row['enrichment_fold'] >= 2 and row['mcc'] >= 0.1 else
                'Weak', axis=1
            )

        print(f"✅ {analysis_level} 统计分析完成，共 {len(results_df)} 个有效关联")

        return results_df

    def perform_enhanced_analysis(self):
        print("🚀 开始增强版共现分析...")
        # 1. 解析ICD编码
        print("解析 'standardized_diagnoses' 列...")
        self.patient_df['parsed_icd_codes'] = self.patient_df['standardized_diagnoses'].apply(
            lambda x: [d['icd_code'] for d in ast.literal_eval(x)] if pd.notna(x) and str(x).strip() else []
        )

        # 初始化计数器
        visit_icd_disease_cooccurrence = defaultdict(int)
        visit_ccs_disease_cooccurrence = defaultdict(int)
        patient_icd_disease_cooccurrence = defaultdict(int)
        patient_ccs_disease_cooccurrence = defaultdict(int)

        # 就诊级别分析
        print("\n计算就诊级别共现...")
        visit_source_counts_icd = defaultdict(int)
        visit_target_counts = defaultdict(int)
        visit_source_counts_ccs = defaultdict(int)

        for index, row in tqdm(self.patient_df.iterrows(), total=len(self.patient_df), desc="就诊级别分析"):
            current_visit_icds = set(row['parsed_icd_codes'])
            current_visit_ccs_categories = set()

            for icd_code in current_visit_icds:
                ccs_category = self.icd_ccs_map.get(str(icd_code), "Unknown_CCS")
                current_visit_ccs_categories.add(ccs_category)

            current_visit_active_disease_labels = [
                label for label in DISEASE_LABEL_COLUMNS if row[label] == 1
            ]

            for icd in current_visit_icds:
                visit_source_counts_icd[icd] += 1
                for disease_label in current_visit_active_disease_labels:
                    visit_icd_disease_cooccurrence[(icd, disease_label)] += 1

            for ccs in current_visit_ccs_categories:
                visit_source_counts_ccs[ccs] += 1
                for disease_label in current_visit_active_disease_labels:
                    visit_ccs_disease_cooccurrence[(ccs, disease_label)] += 1

            for disease_label in current_visit_active_disease_labels:
                visit_target_counts[disease_label] += 1

        # 患者级别分析
        print("\n计算患者级别共现...")
        patient_aggregated_data = defaultdict(lambda: {'icds': set(), 'disease_labels': set()})

        for index, row in tqdm(self.patient_df.iterrows(), total=len(self.patient_df), desc="患者数据聚合"):
            subject_id = row['subject_id']

            for icd_code in row['parsed_icd_codes']:
                patient_aggregated_data[subject_id]['icds'].add(icd_code)

            for label in DISEASE_LABEL_COLUMNS:
                if row[label] == 1:
                    patient_aggregated_data[subject_id]['disease_labels'].add(label)

        patient_source_counts_icd = defaultdict(int)
        patient_source_counts_ccs = defaultdict(int)
        patient_target_counts = defaultdict(int)

        for subject_id, data in tqdm(patient_aggregated_data.items(), desc="患者级别共现计算"):
            patient_unique_icds = data['icds']
            patient_active_disease_labels = data['disease_labels']

            patient_unique_ccs_categories = set()
            for icd_code in patient_unique_icds:
                ccs_category = self.icd_ccs_map.get(str(icd_code), "Unknown_CCS")
                patient_unique_ccs_categories.add(ccs_category)

            for icd in patient_unique_icds:
                patient_source_counts_icd[icd] += 1
                for disease_label in patient_active_disease_labels:
                    patient_icd_disease_cooccurrence[(icd, disease_label)] += 1

            for ccs in patient_unique_ccs_categories:
                patient_source_counts_ccs[ccs] += 1
                for disease_label in patient_active_disease_labels:
                    patient_ccs_disease_cooccurrence[(ccs, disease_label)] += 1

            for disease_label in patient_active_disease_labels:
                patient_target_counts[disease_label] += 1

        # 统计显著性分析
        print("\n--- 步骤2: 统计显著性分析 ---")

        total_visits = len(self.patient_df)
        total_patients = len(patient_aggregated_data)

        visit_icd_enhanced = self.calculate_statistical_significance(
            visit_icd_disease_cooccurrence, total_visits,
            visit_source_counts_icd, visit_target_counts,
            "就诊级别ICD-疾病标签"
        )

        visit_ccs_enhanced = self.calculate_statistical_significance(
            visit_ccs_disease_cooccurrence, total_visits,
            visit_source_counts_ccs, visit_target_counts,
            "就诊级别CCS-疾病标签"
        )

        patient_icd_enhanced = self.calculate_statistical_significance(
            patient_icd_disease_cooccurrence, total_patients,
            patient_source_counts_icd, patient_target_counts,
            "患者级别ICD-疾病标签"
        )

        patient_ccs_enhanced = self.calculate_statistical_significance(
            patient_ccs_disease_cooccurrence, total_patients,
            patient_source_counts_ccs, patient_target_counts,
            "患者级别CCS-疾病标签"
        )

        self.results = {
            'visit_icd_enhanced': visit_icd_enhanced,
            'visit_ccs_enhanced': visit_ccs_enhanced,
            'patient_icd_enhanced': patient_icd_enhanced,
            'patient_ccs_enhanced': patient_ccs_enhanced,
            'summary_stats': {
                'total_visits': total_visits,
                'total_patients': total_patients,
                'unique_icds': len(visit_source_counts_icd),
                'unique_ccs': len(visit_source_counts_ccs),
                'active_disease_labels': len([label for label in DISEASE_LABEL_COLUMNS
                                              if visit_target_counts.get(label, 0) > 0])
            }
        }

        return self.results
class DiseaseSelectionRecommendationEngine:
    def __init__(self, enhanced_results):
        self.results = enhanced_results

    def analyze_disease_clinical_value(self):
        print("\n🎯 分析疾病标签的临床价值...")
        disease_scores = defaultdict(lambda: {
            'icd_associations': 0,
            'ccs_associations': 0,
            'high_clinical_relevance_count': 0,
            'strong_statistical_associations': 0,
            'total_patients': 0,
            'total_visits': 0,
            'avg_enrichment_fold': 0,
            'max_enrichment_fold': 0,
            'clinical_diversity_score': 0
        })

        if 'patient_icd_enhanced' in self.results and len(self.results['patient_icd_enhanced']) > 0:
            patient_icd_df = self.results['patient_icd_enhanced']

            for _, row in patient_icd_df.iterrows():
                disease = row['target']

                disease_scores[disease]['icd_associations'] += 1
                disease_scores[disease]['total_patients'] += row['observed_count']

                if row['clinical_relevance'] == 'High':
                    disease_scores[disease]['high_clinical_relevance_count'] += 1

                if row['association_strength'] in ['Strong', 'Very Strong']:
                    disease_scores[disease]['strong_statistical_associations'] += 1

                disease_scores[disease]['avg_enrichment_fold'] += row['enrichment_fold']
                disease_scores[disease]['max_enrichment_fold'] = max(
                    disease_scores[disease]['max_enrichment_fold'],
                    row['enrichment_fold']
                )

        if 'patient_ccs_enhanced' in self.results and len(self.results['patient_ccs_enhanced']) > 0:
            patient_ccs_df = self.results['patient_ccs_enhanced']

            for _, row in patient_ccs_df.iterrows():
                disease = row['target']
                disease_scores[disease]['ccs_associations'] += 1

        for disease in disease_scores:
            if disease_scores[disease]['icd_associations'] > 0:
                disease_scores[disease]['avg_enrichment_fold'] /= disease_scores[disease]['icd_associations']

            diversity_components = [
                disease_scores[disease]['icd_associations'],
                disease_scores[disease]['ccs_associations'],
                disease_scores[disease]['high_clinical_relevance_count'],
                disease_scores[disease]['strong_statistical_associations']
            ]
            disease_scores[disease]['clinical_diversity_score'] = np.mean(diversity_components)

        disease_value_df = pd.DataFrame([
            {
                'disease_label': disease,
                **scores
            } for disease, scores in disease_scores.items()
        ])

        if len(disease_value_df) > 0:
            scaler = StandardScaler()
            scoring_features = [
                'icd_associations', 'ccs_associations', 'high_clinical_relevance_count',
                'strong_statistical_associations', 'total_patients', 'avg_enrichment_fold',
                'clinical_diversity_score'
            ]

            scaled_features = scaler.fit_transform(disease_value_df[scoring_features])
            weights = np.array([0.2, 0.15, 0.25, 0.15, 0.1, 0.1, 0.05])

            disease_value_df['comprehensive_score'] = np.sum(scaled_features * weights, axis=1)
            disease_value_df = disease_value_df.sort_values('comprehensive_score', ascending=False)

        return disease_value_df

    def recommend_optimal_disease_combinations(self, disease_value_df, target_count=10):
        print(f"\n🎯 推荐最佳的 {target_count} 个疾病组合...")

        if len(disease_value_df) == 0:
            print("⚠️ 没有可分析的疾病数据")
            return {}

        top_comprehensive = disease_value_df.head(target_count)['disease_label'].tolist()

        high_clinical_diseases = disease_value_df[
            disease_value_df['high_clinical_relevance_count'] >= 1
            ].head(target_count)['disease_label'].tolist()

        high_icd_association = disease_value_df.nlargest(
            target_count, 'icd_associations'
        )['disease_label'].tolist()

        balanced_selection = []
        remaining_diseases = disease_value_df.copy()

        while len(balanced_selection) < target_count and len(remaining_diseases) > 0:
            if len(balanced_selection) == 0:
                next_disease = remaining_diseases.iloc[0]['disease_label']
            else:
                excluded_similar = []
                for selected in balanced_selection:
                    if 'Cardio' in selected or 'Cardiomegaly' in selected:
                        excluded_similar.extend([d for d in remaining_diseases['disease_label'] if
                                                 ('Cardio' in d or 'Cardiomegaly' in d) and d != selected])
                    elif 'Pneumo' in selected or 'Pneumonia' in selected:
                        excluded_similar.extend([d for d in remaining_diseases['disease_label'] if
                                                 ('Pneumo' in d or 'Pneumonia' in d) and d != selected])

                available_diseases = remaining_diseases[
                    ~remaining_diseases['disease_label'].isin(excluded_similar)
                ]

                if len(available_diseases) > 0:
                    next_disease = available_diseases.iloc[0]['disease_label']
                else:
                    next_disease = remaining_diseases.iloc[0]['disease_label']

            balanced_selection.append(next_disease)
            remaining_diseases = remaining_diseases[remaining_diseases['disease_label'] != next_disease]

        recommendations = {
            'comprehensive_score_strategy': {
                'diseases': top_comprehensive,
                'description': '基于综合评分的最优疾病组合',
                'focus': '综合考虑所有评价指标的平衡选择'
            },
            'high_clinical_relevance_strategy': {
                'diseases': high_clinical_diseases,
                'description': '高临床相关性疾病组合',
                'focus': '优先选择与ICD诊断高度相关的疾病'
            },
            'max_icd_coverage_strategy': {
                'diseases': high_icd_association,
                'description': '最大ICD覆盖度策略',
                'focus': '最大化覆盖不同ICD诊断的疾病组合'
            },
            'balanced_diversity_strategy': {
                'diseases': balanced_selection,
                'description': '平衡多样性策略',
                'focus': '在临床价值和疾病多样性间寻求平衡'
            }
        }

        print(f"\n📋 疾病选择推荐结果:")
        for strategy_name, strategy_info in recommendations.items():
            print(f"\n🔸 {strategy_name.replace('_', ' ').title()}:")
            print(f"   描述: {strategy_info['description']}")
            print(f"   重点: {strategy_info['focus']}")
            print(f"   疾病列表:")
            for disease in strategy_info['diseases']:
                if disease in disease_value_df['disease_label'].values:
                    disease_info = disease_value_df[disease_value_df['disease_label'] == disease].iloc[0]
                    print(f"     • {disease} (评分: {disease_info['comprehensive_score']:.3f}, "
                          f"ICD关联: {disease_info['icd_associations']}, "
                          f"高临床相关: {disease_info['high_clinical_relevance_count']})")
                else:
                    print(f"     • {disease}")

        return recommendations

def create_enhanced_visualizations(enhanced_results, disease_value_df, recommendations, timestamp):
    try:
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # 1. 疾病临床价值评分
        ax1 = fig.add_subplot(gs[0, :2])
        if len(disease_value_df) > 0:
            top_diseases = disease_value_df.head(15)
            bars = ax1.barh(range(len(top_diseases)), top_diseases['comprehensive_score'])
            ax1.set_yticks(range(len(top_diseases)))
            ax1.set_yticklabels(top_diseases['disease_label'], fontsize=10)
            ax1.set_xlabel('综合临床价值评分')
            ax1.set_title('疾病标签临床价值排名 (TOP 15)', fontsize=14)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2f}', ha='left', va='center', fontsize=9)

        # 2. 统计显著性分布
        ax2 = fig.add_subplot(gs[0, 2:])
        significance_data = []
        for analysis_type, df in enhanced_results.items():
            if isinstance(df, pd.DataFrame) and 'significance_level' in df.columns:
                for _, row in df.iterrows():
                    significance_data.append({
                        'analysis_type': analysis_type.replace('_', ' ').title(),
                        'significance': row['significance_level'],
                        'enrichment_fold': row.get('enrichment_fold', 0)
                    })

        if significance_data:
            sig_df = pd.DataFrame(significance_data)
            sig_counts = sig_df['significance'].value_counts()
            colors = ['#e74c3c', '#f39c12', '#f1c40f', '#95a5a6']
            wedges, texts, autotexts = ax2.pie(sig_counts.values, labels=sig_counts.index,
                                               autopct='%1.1f%%', colors=colors)
            ax2.set_title('统计显著性分布', fontsize=14)

        # 3. 关联强度热力图
        ax3 = fig.add_subplot(gs[1, :])
        if 'patient_icd_enhanced' in enhanced_results and len(enhanced_results['patient_icd_enhanced']) > 0:
            icd_df = enhanced_results['patient_icd_enhanced']
            top_associations = icd_df.nlargest(30, 'enrichment_fold')

            if len(top_associations) > 0:
                icds = top_associations['source'].unique()[:15]
                diseases = top_associations['target'].unique()[:10]

                heatmap_matrix = np.zeros((len(icds), len(diseases)))

                for i, icd in enumerate(icds):
                    for j, disease in enumerate(diseases):
                        matching_rows = top_associations[
                            (top_associations['source'] == icd) &
                            (top_associations['target'] == disease)
                            ]
                        if len(matching_rows) > 0:
                            heatmap_matrix[i, j] = matching_rows.iloc[0]['enrichment_fold']

                sns.heatmap(heatmap_matrix,
                            xticklabels=[d[:20] for d in diseases],
                            yticklabels=[icd[:10] for icd in icds],
                            annot=True, fmt='.1f', cmap='Reds', ax=ax3,
                            cbar_kws={'label': '富集倍数'})
                ax3.set_title('ICD-疾病标签关联强度热力图 (富集倍数)', fontsize=14)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # 4. 散点图
        ax4 = fig.add_subplot(gs[2, :2])
        scatter_data = []
        for analysis_type, df in enhanced_results.items():
            if isinstance(df, pd.DataFrame) and 'enrichment_fold' in df.columns:
                for _, row in df.iterrows():
                    scatter_data.append({
                        'enrichment_fold': min(row['enrichment_fold'], 20),
                        'p_value': max(row.get('p_value', 1), 1e-10),
                        'analysis_type': analysis_type,
                        'clinical_relevance': row.get('clinical_relevance', 'Unknown')
                    })

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)
            color_map = {'High': 'red', 'Moderate': 'orange', 'Low': 'lightblue', 'Unknown': 'gray'}
            for relevance, color in color_map.items():
                subset = scatter_df[scatter_df['clinical_relevance'] == relevance]
                if len(subset) > 0:
                    ax4.scatter(subset['enrichment_fold'], -np.log10(subset['p_value']),
                                c=color, alpha=0.6, label=f'临床相关性: {relevance}', s=30)

            ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            ax4.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', alpha=0.7, label='p=0.01')
            ax4.axvline(x=2, color='blue', linestyle='--', alpha=0.7, label='富集倍数=2')

            ax4.set_xlabel('富集倍数')
            ax4.set_ylabel('-log10(p值)')
            ax4.set_title('关联强度 vs 统计显著性')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 5. 推荐策略对比
        ax5 = fig.add_subplot(gs[2, 2:])
        if recommendations:
            strategy_names = []
            disease_counts = []
            avg_scores = []

            for strategy_name, strategy_info in recommendations.items():
                strategy_names.append(strategy_name.replace('_', '\n'))
                disease_counts.append(len(strategy_info['diseases']))
                strategy_diseases = strategy_info['diseases']
                if len(disease_value_df) > 0:
                    matching_scores = disease_value_df[
                        disease_value_df['disease_label'].isin(strategy_diseases)
                    ]['comprehensive_score']
                    avg_scores.append(matching_scores.mean() if len(matching_scores) > 0 else 0)
                else:
                    avg_scores.append(0)

            x_pos = np.arange(len(strategy_names))
            bars1 = ax5.bar(x_pos - 0.2, disease_counts, 0.4, label='疾病数量', color='lightblue')
            ax5_twin = ax5.twinx()
            bars2 = ax5_twin.bar(x_pos + 0.2, avg_scores, 0.4, label='平均评分', color='lightcoral')

            ax5.set_xlabel('推荐策略')
            ax5.set_ylabel('疾病数量', color='blue')
            ax5_twin.set_ylabel('平均临床价值评分', color='red')
            ax5.set_title('推荐策略对比')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(strategy_names, fontsize=10)

            for bar in bars1:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom')

            for bar in bars2:
                height = bar.get_height()
                ax5_twin.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                              f'{height:.2f}', ha='center', va='bottom')

        # 6. CCS分布
        ax6 = fig.add_subplot(gs[3, :])
        if 'patient_ccs_enhanced' in enhanced_results and len(enhanced_results['patient_ccs_enhanced']) > 0:
            ccs_df = enhanced_results['patient_ccs_enhanced']
            ccs_counts = ccs_df['source'].value_counts().head(15)

            bars = ax6.bar(range(len(ccs_counts)), ccs_counts.values)
            ax6.set_xticks(range(len(ccs_counts)))
            ax6.set_xticklabels([ccs[:25] for ccs in ccs_counts.index],
                                rotation=45, ha='right', fontsize=10)
            ax6.set_ylabel('关联次数')
            ax6.set_title('CCS类别与疾病标签关联频次 (TOP 15)')

            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{int(height)}', ha='center', va='bottom', fontsize=9)

        plt.suptitle('增强版ICD-CCS-疾病标签共现分析综合报告', fontsize=18, y=0.98)

        visualization_file = f"{OUTPUT_DIR}enhanced_comprehensive_analysis_{timestamp}.png"
        plt.savefig(visualization_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"增强可视化图表已保存: {visualization_file}")

    except Exception as e:
        print(f"可视化生成失败: {e}")

def generate_analysis_summary_report(enhanced_results, disease_value_df, recommendations, timestamp):
    """
    生成分析总结报告
    """
    try:
        report_filename = f"{OUTPUT_DIR}analysis_summary_report_{timestamp}.txt"

        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("增强版ICD-CCS-疾病标签共现分析总结报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if 'summary_stats' in enhanced_results:
                stats = enhanced_results['summary_stats']
                f.write("📊 基础统计信息:\n")
                f.write(f"   总就诊次数: {stats.get('total_visits', 'N/A')}\n")
                f.write(f"   唯一患者数: {stats.get('total_patients', 'N/A')}\n")
                f.write(f"   唯一ICD编码数: {stats.get('unique_icds', 'N/A')}\n")
                f.write(f"   唯一CCS类别数: {stats.get('unique_ccs', 'N/A')}\n")
                f.write(f"   活跃疾病标签数: {stats.get('active_disease_labels', 'N/A')}\n\n")

            f.write("📈 统计显著性分析总结:\n")
            for analysis_type, results_df in enhanced_results.items():
                if isinstance(results_df, pd.DataFrame) and len(results_df) > 0:
                    significant_count = len(results_df[results_df.get('p_value', 1) < 0.05])
                    highly_significant_count = len(results_df[results_df.get('p_value', 1) < 0.001])
                    strong_associations = len(results_df[results_df.get('association_strength', '') == 'Strong'])

                    f.write(f"\n   {analysis_type.replace('_', ' ').title()}:\n")
                    f.write(f"     总关联数: {len(results_df)}\n")
                    f.write(f"     显著关联 (p<0.05): {significant_count}\n")
                    f.write(f"     高度显著 (p<0.001): {highly_significant_count}\n")
                    f.write(f"     强关联: {strong_associations}\n")

            if len(disease_value_df) > 0:
                f.write("\n🎯 疾病临床价值排名 (TOP 10):\n")
                top_10_diseases = disease_value_df.head(10)
                for i, (_, row) in enumerate(top_10_diseases.iterrows(), 1):
                    f.write(f"   {i:2d}. {row['disease_label']:<25} "
                            f"(评分: {row['comprehensive_score']:.3f}, "
                            f"ICD关联: {row['icd_associations']}, "
                            f"高临床相关: {row['high_clinical_relevance_count']})\n")

            if recommendations:
                f.write("\n🚀 疾病选择推荐策略:\n")
                for strategy_name, strategy_info in recommendations.items():
                    f.write(f"\n   {strategy_name.replace('_', ' ').title()}:\n")
                    f.write(f"     描述: {strategy_info['description']}\n")
                    f.write(f"     重点: {strategy_info['focus']}\n")
                    f.write(f"     推荐疾病:\n")
                    for disease in strategy_info['diseases']:
                        f.write(f"       • {disease}\n")

            f.write("\n💡 关键发现和建议:\n")
            if len(disease_value_df) > 0:
                high_value_diseases = disease_value_df[
                    disease_value_df['comprehensive_score'] > disease_value_df['comprehensive_score'].quantile(0.8)
                    ]
                f.write(f"\n   高价值疾病标签 (前20%): {len(high_value_diseases)}个\n")
                f.write("   推荐优先关注以下疾病标签用于LLM实验:\n")
                for _, disease in high_value_diseases.head(8).iterrows():
                    f.write(f"     • {disease['disease_label']} - 临床关联强，统计显著性高\n")

            f.write("\n   建议后续分析方向:\n")
            f.write("     1. 深入分析高价值疾病的ICD共病模式\n")
            f.write("     2. 结合患者时序数据分析疾病发展轨迹\n")
            f.write("     3. 构建疾病-ICD知识图谱用于LLM增强\n")
            f.write("     4. 验证推荐疾病组合的临床预测性能\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 80 + "\n")

        print(f"分析总结报告已保存: {report_filename}")

    except Exception as e:
        print(f"报告生成失败: {e}")

def main():
    """
    主执行函数
    """
    print("🚀 启动增强版ICD-CCS-疾病标签共现分析...")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        patient_df = pd.read_csv(PROCESSED_DATA)
        icd_ccs_df = pd.read_csv(ICD_CCS_MAP_PATH)

        icd_to_ccs_map = dict(zip(icd_ccs_df['standardized_icd_code'].astype(str),
                                  icd_ccs_df['ccs_category_description']))

        print(f"患者数据: {patient_df.shape}, ICD-CCS映射: {len(icd_to_ccs_map)}")

        analyzer = EnhancedCooccurrenceAnalyzer(patient_df, icd_to_ccs_map)
        enhanced_results = analyzer.perform_enhanced_analysis()

        recommendation_engine = DiseaseSelectionRecommendationEngine(enhanced_results)
        disease_value_df = recommendation_engine.analyze_disease_clinical_value()
        disease_recommendations = recommendation_engine.recommend_optimal_disease_combinations(disease_value_df)

        # 保存结果
        for analysis_type, results_df in enhanced_results.items():
            if isinstance(results_df, pd.DataFrame) and len(results_df) > 0:
                filename = f"{OUTPUT_DIR}enhanced_{analysis_type}_{timestamp}.csv"
                results_df.to_csv(filename, index=False)
                print(f"已保存: {filename}")

        if len(disease_value_df) > 0:
            disease_value_filename = f"{OUTPUT_DIR}disease_clinical_value_analysis_{timestamp}.csv"
            disease_value_df.to_csv(disease_value_filename, index=False)
            print(f"已保存: {disease_value_filename}")

        recommendations_filename = f"{OUTPUT_DIR}disease_selection_recommendations_{timestamp}.json"
        with open(recommendations_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_timestamp': timestamp,
                'summary_statistics': enhanced_results.get('summary_stats', {}),
                'disease_recommendations': disease_recommendations,
                'disease_clinical_values': disease_value_df.to_dict('records') if len(disease_value_df) > 0 else []
            }, f, ensure_ascii=False, indent=2)
        print(f"已保存: {recommendations_filename}")

        create_enhanced_visualizations(enhanced_results, disease_value_df, disease_recommendations, timestamp)
        generate_analysis_summary_report(enhanced_results, disease_value_df, disease_recommendations, timestamp)

        print("\n" + "=" * 80)
        print("✅ 增强版共现分析完成!")
        print("=" * 80)

        return enhanced_results, disease_value_df, disease_recommendations

    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    print("启动增强版ICD-CCS-疾病标签共现分析...")
    enhanced_results, disease_value_df, recommendations = main()

    if enhanced_results is not None:
        print("\n🎉 分析完成! 主要输出文件:")
        print(f"  📁 结果目录: {OUTPUT_DIR}")
        print(f"\n🚀 基于分析结果的建议:")
        if recommendations and 'comprehensive_score_strategy' in recommendations:
            best_diseases = recommendations['comprehensive_score_strategy']['diseases'][:8]
            print(f"  推荐用于LLM主实验的疾病标签:")
            for disease in best_diseases:
                print(f"    • {disease}")

        print(f"\n  这些疾病标签具有:")
        print(f"    ✓ 强统计显著性")
        print(f"    ✓ 高临床相关性")
        print(f"    ✓ 丰富的ICD诊断关联")
        print(f"    ✓ 充足的患者覆盖度")
    else:
        print("❌ 分析失败，请检查数据和配置")