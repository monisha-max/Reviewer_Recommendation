import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from reviewer_recommender import ReviewerRecommender, ReviewerReviewerSimilarity
from pdf_parser import PDFParser
st.set_page_config(
    page_title="Reviewer Recommendation System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .author-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .score-high {
        background-color: #d4edda;
        color: #155724;
    }
    .score-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .score-low {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'ensemble_results' not in st.session_state:
    st.session_state.ensemble_results = None


@st.cache_resource(show_spinner=False)
def load_recommender_system():
    try:
        dataset_path = "Dataset"
        cache_dir = "cache"
        
        recommender = ReviewerRecommender(
            dataset_path=dataset_path,
            cache_dir=cache_dir,
            use_advanced_models=True,
            use_advanced_features=True 
        )
        recommender.initialize(force_rebuild=False)
        
        return recommender
            
    except Exception as e:
        logger.error(f"Failed to load system: {str(e)}")
        return None

def auto_initialize_system():
    try:
        recommender = load_recommender_system()
        
        if recommender:
            st.session_state.recommender = recommender
            st.session_state.is_initialized = True
            return True
        else:
            return False
            
    except Exception as e:
        return False

def initialize_system():
    try:
        dataset_path = "Dataset"
        cache_dir = "cache"
        
        with st.spinner("Initializing Reviewer Recommendation System... This may take a few minutes..."):
            recommender = ReviewerRecommender(
                dataset_path=dataset_path,
                cache_dir=cache_dir,
                use_advanced_models=True,
                use_advanced_features=True 
            )
            
            recommender.initialize(force_rebuild=False)
            
            st.session_state.recommender = recommender
            st.session_state.is_initialized = True
            
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False


def get_score_class(score):
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    else:
        return "score-low"


def display_final_recommendations(ensemble_results, top_k=10):
    st.markdown('<div class="sub-header">🎯 Top Reviewer Recommendations</div>', unsafe_allow_html=True)

    for rank, (author, score, reasoning) in enumerate(ensemble_results[:top_k], 1):
        if score >= 0.8:
            score_class = "score-high"
            confidence = "Excellent Match"
            icon = "🥇"
        elif score >= 0.6:
            score_class = "score-medium" 
            confidence = "Good Match"
            icon = "🥈"
        else:
            score_class = "score-low"
            confidence = "Moderate Match"
            icon = "🥉"
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 6, 2, 2])
            
            with col1:
                st.markdown(f"### {icon}")
            
            with col2:
                st.markdown(f"**{author}**")
                if st.session_state.recommender:
                    author_info = st.session_state.recommender.get_author_info(author)
                    if author_info:
                        st.caption(f"📚 {author_info['num_publications']} publications • {author_info['total_text_length']:,} characters")
            
            with col3:
                st.markdown(
                    f'<div class="score-badge {score_class}">'
                    f'Score: {score:.3f}</div>',
                    unsafe_allow_html=True
                )
            
            with col4:
                st.caption(confidence)
            with st.expander(f"🔍 Why {author} is a good match", expanded=False):
                st.markdown("**Method Performance:**")
                numeric_items = [(k, v) for k, v in reasoning.items() if isinstance(v, (int, float, float))]
                sorted_methods = sorted(numeric_items, key=lambda x: x[1], reverse=True)
                
                for method_name, method_score in sorted_methods:
                    if method_score > 0.7:
                        method_icon = "🟢"
                        method_desc = "Strong match"
                    elif method_score > 0.5:
                        method_icon = "🟡"
                        method_desc = "Good match"
                    else:
                        method_icon = "🔴"
                        method_desc = "Moderate match"
                    
                    st.write(f"{method_icon} **{method_name}**: {method_score:.3f} ({method_desc})")
                
                
                best_paper = reasoning.get('Best Paper') if isinstance(reasoning, dict) else None
                best_paper_score = reasoning.get('Best Paper Score') if isinstance(reasoning, dict) else None
                if best_paper:
                    st.markdown("**Best Matching Paper (evidence):**")
                    try:
                        score_text = f" — score: {float(best_paper_score):.3f}" if isinstance(best_paper_score, (int, float)) else ""
                    except Exception:
                        score_text = ""
                    st.write(f"• {best_paper}{score_text}")

                top_methods = [name for name, score in sorted_methods[:3] if isinstance(score, (int, float)) and score > 0.5]
                if top_methods:
                    st.markdown("**Key Strengths:**")
                    for method in top_methods:
                        if "SciBERT" in method:
                            st.write(f"• 🧬 **Scientific expertise**: Strong match in research domain")
                        elif "Sentence-BERT" in method:
                            st.write(f"• 🧠 **Semantic understanding**: Similar research concepts")
                        elif "LDA" in method or "NMF" in method:
                            st.write(f"• 📊 **Topic alignment**: Shared research themes")
                        elif "TF-IDF" in method:
                            st.write(f"• 📝 **Text similarity**: Common terminology and keywords")
                        elif "Jaccard" in method:
                            st.write(f"• 🔗 **Content overlap**: Similar research areas")
                        elif "Keyword" in method:
                            st.write(f"• 🏷️ **Keyword matching**: Shared research vocabulary")
            
            st.divider()
    
    st.markdown("### 📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyzed", f"{len(ensemble_results)}")
    
    with col2:
        high_confidence = sum(1 for _, score, _ in ensemble_results if score >= 0.8)
        st.metric("High Confidence", f"{high_confidence}")
    
    with col3:
        avg_score = sum(score for _, score, _ in ensemble_results) / len(ensemble_results)
        st.metric("Avg Score", f"{avg_score:.3f}")
    
    with col4:
        methods_used = set()
        for _, _, reasoning in ensemble_results:
            methods_used.update([m for m, s in reasoning.items() if isinstance(s, (int, float)) and s > 0])
        st.metric("Methods Used", f"{len(methods_used)}")


def display_recommendations(recommendations, top_k=10):
    methods = list(recommendations.keys())
    
    if not methods:
        st.warning("No recommendations available.")
        return
    st.markdown('<div class="sub-header">📊 Method Comparison</div>', unsafe_allow_html=True)
    comparison_data = []
    for method, results in recommendations.items():
        for rank, (author, score) in enumerate(results[:top_k], 1):
            comparison_data.append({
                'Method': method,
                'Rank': rank,
                'Author': author,
                'Score': score
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_methods = st.multiselect(
            "Select methods to compare:",
            methods,
            default=methods[:3] if len(methods) >= 3 else methods
        )
    
    with col2:
        display_top_k = st.slider("Show top K authors:", 3, 20, top_k)
    
    if selected_methods:
        df_filtered = df_comparison[
            (df_comparison['Method'].isin(selected_methods)) & 
            (df_comparison['Rank'] <= display_top_k)
        ]
        fig = px.bar(
            df_filtered,
            x='Author',
            y='Score',
            color='Method',
            barmode='group',
            title=f'Top {display_top_k} Reviewer Recommendations by Method',
            height=500
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Author",
            yaxis_title="Similarity Score",
            legend_title="Method",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="sub-header">📋 Detailed Recommendations</div>', unsafe_allow_html=True)
        
        tabs = st.tabs(selected_methods)
        
        for i, method in enumerate(selected_methods):
            with tabs[i]:
                results = recommendations[method][:display_top_k]
                
                for rank, (author, score) in enumerate(results, 1):
                    score_class = get_score_class(score)
                    
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 2])
                        
                        with col1:
                            st.markdown(f"**#{rank}**")
                        
                        with col2:
                            st.markdown(f"**{author}**")
                            
                            # Get author info
                            if st.session_state.recommender:
                                author_info = st.session_state.recommender.get_author_info(author)
                                if author_info:
                                    st.caption(f"📚 {author_info['num_publications']} publications")
                        
                        with col3:
                            st.markdown(
                                f'<div class="score-badge {score_class}">'
                                f'Score: {score:.4f}</div>',
                                unsafe_allow_html=True
                            )
                        
                        st.divider()
        st.markdown('<div class="sub-header">🎯 Consensus Recommendations</div>', unsafe_allow_html=True)
        st.info("These recommendations combine multiple methods to find the most consistently recommended reviewers.")
        author_scores = {}
        author_ranks = {}
        
        for method in selected_methods:
            for rank, (author, score) in enumerate(recommendations[method][:display_top_k], 1):
                if author not in author_scores:
                    author_scores[author] = []
                    author_ranks[author] = []
                author_scores[author].append(score)
                author_ranks[author].append(rank)
        
        consensus = []
        for author, scores in author_scores.items():
            avg_score = sum(scores) / len(scores)
            avg_rank = sum(author_ranks[author]) / len(author_ranks[author])
            num_methods = len(scores)
            consensus.append((author, avg_score, avg_rank, num_methods))
        consensus.sort(key=lambda x: x[1], reverse=True)

        for rank, (author, avg_score, avg_rank, num_methods) in enumerate(consensus[:10], 1):
            score_class = get_score_class(avg_score)
            
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
                
                with col1:
                    st.markdown(f"**#{rank}**")
                
                with col2:
                    st.markdown(f"**{author}**")
                
                with col3:
                    st.markdown(
                        f'<div class="score-badge {score_class}">'
                        f'Avg Score: {avg_score:.4f}</div>',
                        unsafe_allow_html=True
                    )
                
                with col4:
                    st.caption(f"Avg Rank: {avg_rank:.1f}")
                    st.caption(f"In {num_methods}/{len(selected_methods)} methods")
                
                st.divider()


def main():
    st.markdown('<div class="main-header">📄 Reviewer Recommendation System</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/checklist.png", width=80)
        st.title("⚙️ Settings")
        st.subheader("System Status")
        if not st.session_state.is_initialized:
            cache_exists = (
                os.path.exists("cache/author_profiles.pkl") and
                os.path.exists("cache/similarity_models.pkl")
            )
            
            if cache_exists:
                with st.spinner("Loading from cache..."):
                    if auto_initialize_system():
                        st.session_state.is_initialized = True
                        st.rerun()
            else:
                st.warning("⚠️ **System Not Ready**\n\nCache not found. Please initialize the system first.")
                
                if st.button("🚀 Initialize AI System", type="primary"):
                    if initialize_system():
                        st.success("Upload a paper to get instant recommendations.")
                        st.rerun()
        
        if st.session_state.is_initialized:
            st.success("✅ **System Ready!**\n\nCache loaded successfully. System is ready for recommendations.")
            try:
                import faiss
                st.info("**FAISS Acceleration: Enabled**")
            except ImportError:
                st.info("ℹ️ **FAISS: Not Available**\n\nUsing standard numpy-based search.")
            
            if st.button("🔄 Reinitialize System"):
                st.session_state.is_initialized = False
                st.session_state.recommender = None
                st.rerun()
            if st.session_state.recommender:
                stats = st.session_state.recommender.get_statistics()
                
                st.subheader("📊 Dataset Statistics")
                st.metric("Total Authors", stats['num_authors'])
                st.metric("Total Publications", stats['total_publications'])
                st.metric("Avg Pubs/Author", f"{stats['avg_publications_per_author']:.1f}")
        
        st.divider()
    if not st.session_state.is_initialized:
        st.info("👈 Please initialize the system using the sidebar to get started.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📚 Dataset")
            st.write("700+ publications from 70+ authors")
        
        with col2:
            st.markdown("### 🤖 AI Methods")
            st.write("7 similarity algorithms")
        
        with col3:
            st.markdown("### 🎯 Accurate")
            st.write("Ensemble consensus recommendations")
    else:
        st.markdown('<div class="sub-header">📤 Upload Research Paper</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload the research paper for which you want to find reviewers"
        )
        
        with st.expander("Or use a sample paper from dataset"):
            if st.session_state.recommender:
                authors = list(st.session_state.recommender.author_profiles.keys())
                selected_author = st.selectbox("Select an author:", [""] + authors)
                
                if selected_author:
                    profile = st.session_state.recommender.author_profiles[selected_author]
                    if profile.publications:
                        sample_papers = [pub['pdf_path'] for pub in profile.publications]
                        selected_paper = st.selectbox("Select a paper:", sample_papers)
                        
                        if st.button("Use This Paper"):
                            st.session_state.uploaded_file_path = selected_paper
                            st.success(f"Selected: {os.path.basename(selected_paper)}")
        if uploaded_file is not None:
            upload_dir = "/Users/lrao/Desktop/AppliedAI/uploaded_papers"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, uploaded_file.name)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            st.session_state.uploaded_file_path = file_path
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        if st.session_state.uploaded_file_path:
            st.markdown('<div class="sub-header">Reviewer Analysis</div>', unsafe_allow_html=True)

            top_k = 10
            with st.spinner("system is analyzing your paper and finding the best reviewers..."):
                try:
                    #using ensemble method 
                    ensemble_results = st.session_state.recommender.get_ensemble_recommendations(
                        st.session_state.uploaded_file_path,
                        top_k=top_k,
                        weights={
                            'TF-IDF + Cosine': 0.6,
                            'BM25': 0.9,
                            'Jaccard (Bigrams)': 0.4,
                            'Keyword Matching': 0.3,
                            'Entity Matching (NER)': 0.8,
                            'LDA Topic Model': 1.0,
                            'NMF Topic Model': 1.0,
                            'Sentence-BERT': 1.5,
                            'SciBERT': 1.8,
                            'Doc2Vec': 0.9,
                            'E5 Embeddings': 1.6,
                            'Structural Similarity': 0.8,
                            'Visual Similarity (images)': 0.8,
                            'Visual Similarity (text)': 0.5,
                            'Citation Impact': 0.7,
                            'RRF Fusion': 0.8
                        }
                    )
                    
                    st.session_state.ensemble_results = ensemble_results
                    st.success("analysis complete! Here are your top reviewer recommendations:")
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.session_state.ensemble_results = None
            if st.session_state.ensemble_results:
                st.markdown("---")
                display_final_recommendations(st.session_state.ensemble_results, top_k)
                st.markdown('<div class="sub-header">💾 Export Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    output = {
                        'paper': st.session_state.uploaded_file_path,
                        'paper_name': os.path.basename(st.session_state.uploaded_file_path),
                        'recommendations': [
                            {
                                'rank': i, 
                                'author': author, 
                                'score': float(score),
                                'reasoning': reasoning
                            }
                            for i, (author, score, reasoning) in enumerate(st.session_state.ensemble_results, 1)
                        ],
                        'analysis_summary': {
                            'total_analyzed': len(st.session_state.ensemble_results),
                            'high_confidence': sum(1 for _, score, _ in st.session_state.ensemble_results if score >= 0.8),
                            'avg_score': sum(score for _, score, _ in st.session_state.ensemble_results) / len(st.session_state.ensemble_results),
                            'methods_used': 7
                        }
                    }
                    
                    results_json = json.dumps(output, indent=2)
                    st.download_button(
                        label="📥 Download as JSON",
                        data=results_json,
                        file_name="ai_reviewer_recommendations.json",
                        mime="application/json"
                    )
                
                with col2:
                    rows = []
                    for rank, (author, score, reasoning) in enumerate(st.session_state.ensemble_results, 1):
                        numeric_reasoning = [(k, v) for k, v in reasoning.items() if isinstance(v, (int, float))]
                        top_method = max(numeric_reasoning, key=lambda x: x[1]) if numeric_reasoning else ('Unknown', 0.0)
                        
                        rows.append({
                            'Rank': rank,
                            'Author': author,
                            'Score': score,
                            'Confidence': 'Excellent' if score >= 0.8 else 'Good' if score >= 0.6 else 'Moderate',
                            'Top_Method': top_method[0],
                            'Top_Method_Score': top_method[1]
                        })
                    
                    df_export = pd.DataFrame(rows)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv,
                        file_name="ai_reviewer_recommendations.csv",
                        mime="text/csv"
                    )
                st.markdown("---")
                st.markdown("### 📊 Additional Analysis")
                
                tab1, tab2, tab3 = st.tabs(["🔬 Strategy Comparison", "👥 Reviewer-Reviewer Similarity", "📑 Evaluation Report"])
                
                with tab1:
                    st.markdown("#### Compare Individual Algorithm Performance")
                    st.write("See how each of the 10 algorithms performed on this paper:")
                    comparison_data = []
                    for author, score, reasoning in st.session_state.ensemble_results[:10]:
                        row = {'Author': author, 'Final Score': f"{score:.3f}"}
                        if 'individual_scores' in reasoning:
                            for method, method_score in reasoning['individual_scores'].items():
                                row[method] = f"{method_score:.3f}" if method_score > 0 else "-"
                        
                        comparison_data.append(row)
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)
                        
                        st.markdown("**📈 Interpretation:**")
                        st.write("""
                        - Each column shows how that algorithm scored each reviewer
                        - **Final Score** is the weighted ensemble of all methods
                        - `-` means the method didn't find a strong match
                        - Compare columns to see which algorithms agree/disagree
                        """)

                        st.markdown("#### 🎯 Method Agreement Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Methods Used", "10 algorithms")
                        with col2:
                            top_reasoning = st.session_state.ensemble_results[0][2]
                            if 'individual_scores' in top_reasoning:
                                active_methods = sum(1 for s in top_reasoning['individual_scores'].values() if s > 0)
                                st.metric("Methods Agreeing on #1", f"{active_methods}/10")
                
                with tab2:
                    st.markdown("#### Find Similar Reviewers")
                    st.write("Explore which reviewers have similar expertise based on their publications.")
                    from reviewer_recommender import ReviewerReviewerSimilarity
                    
                    rr_similarity = ReviewerReviewerSimilarity(
                        st.session_state.recommender.author_profiles
                    )
                    all_authors = list(st.session_state.recommender.author_profiles.keys())
                    selected_reviewer = st.selectbox(
                        "Select a reviewer to find similar experts:",
                        [""] + all_authors,
                        key="rr_select"
                    )
                    
                    if selected_reviewer:
                        with st.spinner(f"Finding reviewers similar to {selected_reviewer}..."):
                            similar_reviewers = rr_similarity.find_similar_reviewers(
                                selected_reviewer, 
                                top_k=10
                            )
                        
                        st.markdown(f"#### Top 10 Reviewers Similar to **{selected_reviewer}**")
                        
                        for rank, (author, sim_score) in enumerate(similar_reviewers, 1):
                            if author == selected_reviewer:
                                continue  
                            
                            col1, col2, col3 = st.columns([1, 5, 2])
                            
                            with col1:
                                st.markdown(f"**#{rank}**")
                            
                            with col2:
                                st.markdown(f"**{author}**")
                            
                            with col3:
                                score_class = get_score_class(sim_score)
                                st.markdown(
                                    f'<span class="{score_class}">{sim_score:.3f}</span>',
                                    unsafe_allow_html=True
                                )
                        
                        st.markdown("---")
                        st.markdown("**💡 Use Case:**")
                        st.write("""
                        - **Conflict of Interest**: If a top recommendation has a conflict, 
                          find similar reviewers as alternatives
                        - **Backup Reviewers**: Identify experts with similar expertise
                        - **Research Networks**: Discover potential collaborators or co-reviewers
                        """)

                with tab3:
                    st.markdown("#### Run Evaluation (with Ground Truth)")
                    st.write("Upload a JSON file mapping test papers to relevant reviewers to compute metrics.")
                    
                    gt_file = st.file_uploader("Upload ground truth JSON", type=["json"], key="gt_json")
                    test_papers = []
                    
                    if gt_file is not None:
                        import json as _json
                        ground_truth = _json.loads(gt_file.getvalue().decode("utf-8"))
                        st.success(f"Loaded ground truth for {len(ground_truth)} papers")
                        with st.expander("Preview ground truth"):
                            st.json(ground_truth)
                        k_values = st.multiselect("Select K values", [1, 3, 5, 10], default=[1, 3, 5])
                        
                        if st.button("Run Evaluation", type="primary"):
                            with st.spinner("Evaluating methods..."):
                                try:
                                    test_papers = list(ground_truth.keys())
                                    report_path = st.session_state.recommender.evaluate_methods(
                                        test_papers=test_papers,
                                        ground_truth=ground_truth,
                                        k_values=k_values
                                    )
                                    if report_path:
                                        st.success("Evaluation complete. Download report below.")
                                        with open(report_path, 'r') as f:
                                            report_text = f.read()
                                        st.download_button(
                                            label="📥 Download Evaluation Report",
                                            data=report_text,
                                            file_name="evaluation_report.txt",
                                            mime="text/plain"
                                        )
                                    else:
                                        st.warning("Evaluation produced no results. Check ground truth and try again.")
                                except Exception as e:
                                    st.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()

