import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from reviewer_recommender import ReviewerRecommender, ReviewerReviewerSimilarity
from pdf_parser import PDFParser
def example_1_basic_usage():
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    print("\nInitializing system...")
    recommender.initialize(force_rebuild=False)
    stats = recommender.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Total Authors: {stats['num_authors']}")
    print(f"  Total Publications: {stats['total_publications']}")
    print(f"  Average Publications per Author: {stats['avg_publications_per_author']:.2f}")
    sample_paper = "/Users/lrao/Desktop/AppliedAI/Dataset/Amit Saxena/A Review of Clustering Techniques.pdf"
    
    print(f"\nGenerating recommendations for: {os.path.basename(sample_paper)}")
    
    recommendations = recommender.recommend_reviewers(
        paper_path=sample_paper,
        top_k=5
    )
    print("\nTop 5 Recommended Reviewers by Method:")
    for method, results in recommendations.items():
        print(f"\n{method}:")
        for rank, (author, score) in enumerate(results, 1):
            print(f"  {rank}. {author:30s} Score: {score:.4f}")


def example_2_specific_methods():
    print("\n" + "="*80)
    print("EXAMPLE 2: Using Specific Methods")
    print("="*80)
    
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    
    recommender.initialize(force_rebuild=False)
    available_methods = recommender.get_available_methods()
    print(f"\nAvailable Methods: {', '.join(available_methods)}")
    selected_methods = ['TF-IDF + Cosine', 'LDA Topic Model']
    
    sample_paper = "/Users/lrao/Desktop/AppliedAI/Dataset/Geeta Rani/sensors-21-05386.pdf"
    
    print(f"\nUsing methods: {', '.join(selected_methods)}")
    
    recommendations = recommender.recommend_reviewers(
        paper_path=sample_paper,
        top_k=10,
        methods=selected_methods
    )
    for method, results in recommendations.items():
        print(f"\n{method} - Top 10:")
        for rank, (author, score) in enumerate(results, 1):
            print(f"  {rank:2d}. {author:30s} {score:.4f}")


def example_3_ensemble_recommendations():
    print("\n" + "="*80)
    print("EXAMPLE 3: Ensemble Recommendations")
    print("="*80)
    
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    
    recommender.initialize(force_rebuild=False)
    
    sample_paper = "/Users/lrao/Desktop/AppliedAI/Dataset/Minni Jain/paper1.pdf"
    weights = {
        'TF-IDF + Cosine': 2.0,    
        'Jaccard (Bigrams)': 1.0,
        'Keyword Matching': 0.5,      
        'LDA Topic Model': 1.5,
        'NMF Topic Model': 1.5
    }
    
    print(f"\nWeights:")
    for method, weight in weights.items():
        print(f"  {method}: {weight}")
    
    print(f"\nGenerating ensemble recommendations...")
    
    ensemble_results = recommender.get_ensemble_recommendations(
        paper_path=sample_paper,
        top_k=10,
        weights=weights
    )
    
    print("\nTop 10 Ensemble Recommendations:")
    for rank, (author, score) in enumerate(ensemble_results, 1):
        print(f"  {rank:2d}. {author:30s} Score: {score:.4f}")


def example_4_text_based_recommendation():
    print("\n" + "="*80)
    print("EXAMPLE 4: Text-Based Recommendation")
    print("="*80)
    
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    
    recommender.initialize(force_rebuild=False)
    paper_text = """
    This paper presents a novel approach to deep learning for computer vision tasks.
    We propose a new convolutional neural network architecture that achieves 
    state-of-the-art results on image classification benchmarks. The method uses
    attention mechanisms and residual connections to improve feature extraction.
    We demonstrate superior performance on CIFAR-10, ImageNet, and custom datasets.
    Our approach is efficient and can be deployed on resource-constrained devices.
    """
    
    print("\nInput Text:")
    print(paper_text)
    
    recommendations = recommender.recommend_reviewers_from_text(
        paper_text=paper_text,
        top_k=5
    )
    
    print("\nTop 5 Recommended Reviewers:")
    for method, results in recommendations.items():
        print(f"\n{method}:")
        for rank, (author, score) in enumerate(results, 1):
            print(f"  {rank}. {author:30s} {score:.4f}")


def example_5_author_information():
    print("\n" + "="*80)
    print("EXAMPLE 5: Author Information")
    print("="*80)
    
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    
    recommender.initialize(force_rebuild=False)
    authors_to_check = ['Amit Saxena', 'Geeta Rani', 'Minni Jain']
    
    print("\nAuthor Profiles:")
    for author_name in authors_to_check:
        info = recommender.get_author_info(author_name)
        
        if info:
            print(f"\n{author_name}:")
            print(f"  Number of Publications: {info['num_publications']}")
            print(f"  Total Text Length: {info['total_text_length']:,} characters")
            print(f"  Average Text per Paper: {info['avg_text_length']:,.0f} characters")
            print(f"  Publications:")
            for i, pub in enumerate(info['publication_files'][:5], 1):
                print(f"    {i}. {pub}")
            if len(info['publication_files']) > 5:
                print(f"    ... and {len(info['publication_files']) - 5} more")


def example_6_reviewer_reviewer_similarity():
    print("\n" + "="*80)
    print("EXAMPLE 6: Reviewer-Reviewer Similarity")
    print("="*80)
    
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    
    recommender.initialize(force_rebuild=False)
    rr_similarity = ReviewerReviewerSimilarity(recommender.author_profiles)
    target_author = 'Amit Saxena'
    
    print(f"\nFinding reviewers similar to {target_author}...")
    
    similar_reviewers = rr_similarity.find_similar_reviewers(
        author_name=target_author,
        top_k=10
    )
    
    print(f"\nTop 10 Similar Reviewers to {target_author}:")
    for rank, (author, score) in enumerate(similar_reviewers, 1):
        print(f"  {rank:2d}. {author:30s} Similarity: {score:.4f}")


def example_7_save_results():
    print("\n" + "="*80)
    print("EXAMPLE 7: Saving Results")
    print("="*80)
    
    recommender = ReviewerRecommender(
        dataset_path="/Users/lrao/Desktop/AppliedAI/Dataset",
        cache_dir="/Users/lrao/Desktop/AppliedAI/cache",
        use_advanced_models=False
    )
    
    recommender.initialize(force_rebuild=False)
    sample_paper = "/Users/lrao/Desktop/AppliedAI/Dataset/Pabitra Mitra/paper1.pdf"
    
    recommendations = recommender.recommend_reviewers(
        paper_path=sample_paper,
        top_k=10
    )
    output_dir = "/Users/lrao/Desktop/AppliedAI/results"
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, "recommendations.json")
    recommender.save_recommendations(recommendations, sample_paper, json_path)
    
    print(f"\nResults saved to: {json_path}")
    pretty_path = os.path.join(output_dir, "recommendations_pretty.json")
    
    output = {
        'paper': os.path.basename(sample_paper),
        'top_k': 10,
        'methods': {}
    }
    
    for method, results in recommendations.items():
        output['methods'][method] = [
            {'rank': i, 'author': author, 'score': float(score)}
            for i, (author, score) in enumerate(results, 1)
        ]
    
    with open(pretty_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Pretty JSON saved to: {pretty_path}")
    import pandas as pd
    
    rows = []
    for method, results in recommendations.items():
        for rank, (author, score) in enumerate(results, 1):
            rows.append({
                'Method': method,
                'Rank': rank,
                'Author': author,
                'Score': score
            })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "recommendations.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"CSV saved to: {csv_path}")
    print("\nCSV Preview:")
    print(df.head(15).to_string(index=False))


def main():   
    print("\n")
    print("="*80)
    print("REVIEWER RECOMMENDATION SYSTEM - EXAMPLE USAGE")
    print("="*80)
    print("\nThis script demonstrates various ways to use the system.")
    print("Each example is independent and can be run separately.")
    print("\n")
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Specific Methods", example_2_specific_methods),
        ("Ensemble Recommendations", example_3_ensemble_recommendations),
        ("Text-Based Recommendation", example_4_text_based_recommendation),
        ("Author Information", example_5_author_information),
        ("Reviewer-Reviewer Similarity", example_6_reviewer_reviewer_similarity),
        ("Saving Results", example_7_save_results)
    ]
    
    print("Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nSelect example to run (1-7, or 0 for all): ", end="")
    
    try:
        choice = input().strip()
        
        if choice == '0':
            for name, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"\nError in {name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            name, func = examples[int(choice) - 1]
            func()
        else:
            print("Invalid choice. Running Example 1 by default...")
            example_1_basic_usage()
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

