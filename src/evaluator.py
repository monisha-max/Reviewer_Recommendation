import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        num_relevant = sum(1 for author in recommended_k if author in relevant_set)
        
        return num_relevant / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        num_relevant = len(recommended_k & relevant_set)
        
        return num_relevant / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(recommended: List[Tuple[str, float]], relevant: List[str], k: int) -> float:
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        dcg = 0.0
        for i, (author, score) in enumerate(recommended_k):
            relevance = 1.0 if author in relevant_set else 0.0
            dcg += relevance / np.log2(i + 2)  
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(recommended: List[str], relevant: List[str]) -> float:
        relevant_set = set(relevant)
        
        for i, author in enumerate(recommended):
            if author in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def overlap_coefficient(recommended: List[str], relevant: List[str]) -> float:
        recommended_set = set(recommended)
        relevant_set = set(relevant)
        
        intersection = len(recommended_set & relevant_set)
        min_size = min(len(recommended_set), len(relevant_set))
        
        return intersection / min_size if min_size > 0 else 0.0


class StrategyEvaluator:
    def __init__(self, output_dir: str = "evaluation_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = EvaluationMetrics()
    
    def evaluate_strategy(self, 
                         strategy_name: str,
                         recommendations: List[Tuple[str, float]],
                         ground_truth: List[str],
                         k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        recommended_names = [author for author, score in recommendations]
        
        results = {
            'strategy_name': strategy_name,
            'metrics': {}
        }
        
        for k in k_values:
            results['metrics'][f'P@{k}'] = self.metrics.precision_at_k(
                recommended_names, ground_truth, k
            )
            results['metrics'][f'R@{k}'] = self.metrics.recall_at_k(
                recommended_names, ground_truth, k
            )
            results['metrics'][f'NDCG@{k}'] = self.metrics.ndcg_at_k(
                recommendations, ground_truth, k
            )
        
        results['metrics']['MRR'] = self.metrics.mean_reciprocal_rank(
            recommended_names, ground_truth
        )
        results['metrics']['Overlap'] = self.metrics.overlap_coefficient(
            recommended_names, ground_truth
        )
        
        return results
    
    def compare_strategies(self,
                          strategies_results: Dict[str, List[Tuple[str, float]]],
                          ground_truth: List[str],
                          k_values: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        all_results = []
        
        for strategy_name, recommendations in strategies_results.items():
            result = self.evaluate_strategy(
                strategy_name, recommendations, ground_truth, k_values
            )
            
            row = {'Strategy': strategy_name}
            row.update(result['metrics'])
            all_results.append(row)
        
        df = pd.DataFrame(all_results)
        
        return df
    
    def generate_report(self,
                       comparison_df: pd.DataFrame,
                       query_info: Dict,
                       save_name: str = None) -> str:
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"evaluation_report_{timestamp}"
        csv_path = os.path.join(self.output_dir, f"{save_name}.csv")
        comparison_df.to_csv(csv_path, index=False)
        self._create_visualizations(comparison_df, save_name)
        report_path = os.path.join(self.output_dir, f"{save_name}.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REVIEWER RECOMMENDATION EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Query Information:\n")
            f.write("-" * 40 + "\n")
            for key, value in query_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("Strategy Comparison:\n")
            f.write("-" * 40 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            f.write("Best Strategies by Metric:\n")
            f.write("-" * 40 + "\n")
            
            for col in comparison_df.columns:
                if col != 'Strategy':
                    best_idx = comparison_df[col].idxmax()
                    best_strategy = comparison_df.loc[best_idx, 'Strategy']
                    best_value = comparison_df.loc[best_idx, col]
                    f.write(f"{col}: {best_strategy} ({best_value:.4f})\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report_path
    
    def _create_visualizations(self, comparison_df: pd.DataFrame, save_name: str):
        sns.set_style("whitegrid")
        metric_cols = [col for col in comparison_df.columns if col != 'Strategy']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
        
        #Plot 1: Precision@K
        precision_cols = [col for col in metric_cols if col.startswith('P@')]
        if precision_cols:
            ax = axes[0, 0]
            comparison_df.set_index('Strategy')[precision_cols].plot(
                kind='bar', ax=ax, rot=45
            )
            ax.set_title('Precision@K')
            ax.set_ylabel('Precision')
            ax.legend(title='K', bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)
        
        #Plot 2: Recall@K
        recall_cols = [col for col in metric_cols if col.startswith('R@')]
        if recall_cols:
            ax = axes[0, 1]
            comparison_df.set_index('Strategy')[recall_cols].plot(
                kind='bar', ax=ax, rot=45
            )
            ax.set_title('Recall@K')
            ax.set_ylabel('Recall')
            ax.legend(title='K', bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)
        
        #Plot 3: NDCG@K
        ndcg_cols = [col for col in metric_cols if col.startswith('NDCG@')]
        if ndcg_cols:
            ax = axes[1, 0]
            comparison_df.set_index('Strategy')[ndcg_cols].plot(
                kind='bar', ax=ax, rot=45
            )
            ax.set_title('NDCG@K')
            ax.set_ylabel('NDCG')
            ax.legend(title='K', bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)
        
        #Plot 4: Overall metrics
        other_cols = ['MRR', 'Overlap']
        existing_other_cols = [col for col in other_cols if col in metric_cols]
        if existing_other_cols:
            ax = axes[1, 1]
            comparison_df.set_index('Strategy')[existing_other_cols].plot(
                kind='bar', ax=ax, rot=45
            )
            ax.set_title('Other Metrics')
            ax.set_ylabel('Score')
            ax.legend(bbox_to_anchor=(1.05, 1))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_path}")
    
    def cross_validate_strategies(self,
                                 author_profiles: Dict,
                                 similarity_calculators: Dict,
                                 n_folds: int = 5) -> pd.DataFrame:
        
        results = []
        author_names = list(author_profiles.keys())
        test_size = min(n_folds, len(author_names))
        
        for i, test_author in enumerate(author_names[:test_size]):
            logger.info(f"Fold {i+1}/{test_size}: Testing on {test_author}")
            test_profile = author_profiles[test_author]
            if test_profile.num_publications > 0:
                query_text = test_profile.publications[0]['text']
                ground_truth = [test_author]
                strategy_results = {}
                for strategy_name, calculator in similarity_calculators.items():
                    try:
                        recommendations = calculator.calculate_similarity(query_text, top_k=10)
                        strategy_results[strategy_name] = recommendations
                    except Exception as e:
                        logger.error(f"Error with {strategy_name}: {str(e)}")
                fold_results = self.compare_strategies(
                    strategy_results, ground_truth, k_values=[1, 3, 5, 10]
                )
                fold_results['Fold'] = i + 1
                fold_results['Test_Author'] = test_author
                
                results.append(fold_results)
        if results:
            all_results = pd.concat(results, ignore_index=True)
            metric_cols = [col for col in all_results.columns 
                          if col not in ['Strategy', 'Fold', 'Test_Author']]
            
            aggregated = all_results.groupby('Strategy')[metric_cols].mean()
            aggregated['Strategy'] = aggregated.index
            aggregated = aggregated.reset_index(drop=True)
            
            return aggregated
        
        return pd.DataFrame()


if __name__ == "__main__":
    print("Testing Evaluation Framework")
    print("="*80)
    recommendations_1 = [
        ('Author1', 0.95),
        ('Author2', 0.89),
        ('Author3', 0.85),
        ('Author4', 0.78),
        ('Author5', 0.72)
    ]
    
    recommendations_2 = [
        ('Author2', 0.92),
        ('Author1', 0.88),
        ('Author5', 0.82),
        ('Author3', 0.75),
        ('Author6', 0.70)
    ]
    
    ground_truth = ['Author1', 'Author2', 'Author3']
    
    strategies = {
        'TF-IDF + Cosine': recommendations_1,
        'Jaccard Similarity': recommendations_2
    }
    
    evaluator = StrategyEvaluator()
    
    comparison = evaluator.compare_strategies(strategies, ground_truth)
    
    print("\nComparison Results:")
    print(comparison.to_string(index=False))
    query_info = {
        'Paper': 'test_paper.pdf',
        'Method': 'Comparative Evaluation',
        'Date': datetime.now().strftime('%Y-%m-%d')
    }
    
    report_path = evaluator.generate_report(comparison, query_info, 'test_evaluation')
    print(f"\nReport saved to: {report_path}")

