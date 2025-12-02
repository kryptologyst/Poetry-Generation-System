"""Streamlit demo for poetry generation."""

import logging
import sys
from pathlib import Path

import streamlit as st
import torch
from transformers import GPT2Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from poetry_generation import (
    Config,
    create_poetry_generator,
    create_poetry_sampler,
    get_device,
    set_deterministic,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Poetry Generation System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.poem-container {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}
.poem-text {
    font-family: 'Georgia', serif;
    font-size: 1.1rem;
    line-height: 1.6;
    white-space: pre-line;
}
.metric-container {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 5px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name: str = "gpt2"):
    """Load the poetry generation model."""
    try:
        device = get_device()
        generator = create_poetry_generator(
            model_type="gpt2",
            model_name=model_name,
            device=device,
        )
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        sampler = create_poetry_sampler(generator, tokenizer, device)
        
        return generator, tokenizer, sampler, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📝 Poetry Generation System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gpt2", "gpt2-medium", "gpt2-large"],
        index=0,
        help="Choose the GPT-2 model variant"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        generator, tokenizer, sampler, device = load_model(model_name)
    
    if generator is None:
        st.error("Failed to load model. Please check the error messages above.")
        return
    
    st.sidebar.success(f"Model loaded: {model_name}")
    st.sidebar.info(f"Device: {device}")
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    
    prompt = st.sidebar.text_area(
        "Poetry Prompt",
        value="The moonlit sky is full of dreams, where",
        help="Enter your prompt to start the poetry generation"
    )
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of poetry samples to generate"
    )
    
    max_length = st.sidebar.slider(
        "Maximum Length",
        min_value=50,
        max_value=300,
        value=150,
        help="Maximum length of generated poetry"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness (lower = more focused, higher = more creative)"
    )
    
    top_p = st.sidebar.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Controls diversity by limiting to top-p probability mass"
    )
    
    top_k = st.sidebar.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        help="Limits sampling to top-k most likely tokens"
    )
    
    repetition_penalty = st.sidebar.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.1,
        step=0.1,
        help="Penalty for repeating tokens (1.0 = no penalty)"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42,
        help="Seed for reproducible generation"
    )
    
    # Style selection
    st.sidebar.subheader("Style Options")
    use_styles = st.sidebar.checkbox("Generate with Different Styles", value=False)
    
    if use_styles:
        selected_styles = st.sidebar.multiselect(
            "Select Styles",
            ["romantic", "modern", "classical", "haiku", "sonnet"],
            default=["romantic", "modern"]
        )
    
    # Generate button
    if st.sidebar.button("🎭 Generate Poetry", type="primary"):
        with st.spinner("Generating poetry..."):
            try:
                # Set seed for reproducibility
                set_deterministic(seed)
                
                if use_styles and hasattr(generator, 'generate_with_style'):
                    # Generate with different styles
                    style_samples = sampler.sample_with_styles(
                        prompt=prompt,
                        styles=selected_styles,
                        num_samples_per_style=1,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                    
                    # Display results by style
                    for style, samples in style_samples.items():
                        st.markdown(f"### {style.title()} Style")
                        for i, sample in enumerate(samples, 1):
                            st.markdown(f"""
                            <div class="poem-container">
                                <div class="poem-text">{sample}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Generate regular samples
                    samples = sampler.sample(
                        prompt=prompt,
                        num_samples=num_samples,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                    )
                    
                    # Display results
                    for i, sample in enumerate(samples, 1):
                        st.markdown(f"### Sample {i}")
                        st.markdown(f"""
                        <div class="poem-container">
                            <div class="poem-text">{sample}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.success(f"Generated {num_samples} poetry samples successfully!")
                
            except Exception as e:
                st.error(f"Error generating poetry: {e}")
                logger.error(f"Generation error: {e}")
    
    # Main content area
    st.markdown("## About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Features
        - **GPT-2 Based**: Uses state-of-the-art transformer models
        - **Multiple Styles**: Generate poetry in different styles
        - **Customizable**: Adjust temperature, top-p, top-k parameters
        - **Reproducible**: Set random seeds for consistent results
        - **Real-time**: Generate poetry instantly in your browser
        """)
    
    with col2:
        st.markdown("""
        ### Usage Tips
        - **Temperature**: Lower values (0.1-0.5) for more focused poetry
        - **Top-p**: Higher values (0.8-0.95) for more diverse vocabulary
        - **Top-k**: Lower values (10-30) for more constrained generation
        - **Repetition Penalty**: Higher values (1.2-1.5) to reduce repetition
        - **Prompts**: Try different starting phrases for varied results
        """)
    
    # Model information
    with st.expander("Model Information"):
        st.markdown(f"""
        **Model**: {model_name}
        **Device**: {device}
        **Tokenizer**: GPT2Tokenizer
        **Architecture**: GPT-2 Transformer
        """)
        
        if hasattr(generator.model, 'config'):
            config = generator.model.config
            st.markdown(f"""
            **Parameters**: {config.n_embd * config.n_layer * 4:,} (estimated)
            **Vocabulary Size**: {config.vocab_size:,}
            **Max Position Embeddings**: {config.n_positions:,}
            **Hidden Size**: {config.n_embd:,}
            **Number of Layers**: {config.n_layer:,}
            **Number of Attention Heads**: {config.n_head:,}
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io/) and [Transformers](https://huggingface.co/transformers/)"
    )

if __name__ == "__main__":
    main()
