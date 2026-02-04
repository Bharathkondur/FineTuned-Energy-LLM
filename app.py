"""
Streamlit GUI for Fine-tuned Qwen2.5 3B Petroleum Engineering Assistant.
Optimized for fast inference on 8GB VRAM.
"""
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# Page configuration
st.set_page_config(
    page_title="Energy AI Assistant",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a5f;
        --secondary-color: #3d7ea6;
        --accent-color: #f39c12;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1e3a5f 0%, #3d7ea6 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .header-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        color: #b8d4e8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #2d5a87 0%, #1e3a5f 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.75rem 0;
        color: white;
    }
    
    .assistant-message {
        background: #1a1f2e;
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.75rem 0;
        color: #e0e0e0;
        border-left: 4px solid #f39c12;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #f39c12 0%, #e67e22 100%);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Info card */
    .info-card {
        background: #1a1f2e;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #3d7ea6;
        text-align: center;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f39c12;
    }
    
    .stat-label {
        color: #b8d4e8;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./qwen25_energy_finetuned"


@st.cache_resource
def load_model():
    """Load the fine-tuned model with optimizations."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    
    # Load base model with optimizations
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",  # Faster on some GPUs
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_tokens, temperature):
    """Generate a response with optimized settings."""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize with max length limit
    inputs = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=1024  # Limit input length
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate with optimized settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0.1,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Clear cache after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return response.strip()


def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⛽ Energy AI Assistant</h1>
        <p class="header-subtitle">Fine-tuned Qwen2.5 3B | Petroleum Engineering Expert</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Simplified parameters
        max_tokens = st.slider("Max Response Length", 128, 512, 256, step=64,
                               help="Higher = longer answers, but slower")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.5, step=0.1,
                               help="Higher = more creative")
        
        st.markdown("---")
        
        # System prompt
        system_prompt = st.text_area(
            "🎭 AI Persona",
            value="You are an expert petroleum engineer. Provide concise, accurate answers about drilling, reservoir engineering, and production optimization.",
            height=120
        )
        
        st.markdown("---")
        
        # GPU info
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)[:25]}")
            st.progress(mem_used / mem_total, text=f"VRAM: {mem_used:.1f}/{mem_total:.1f} GB")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        try:
            model, tokenizer = load_model()
            st.success("✅ Model loaded!", icon="✅")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return
    
    # Sample questions (only show if no messages)
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Quick Start - Click a question:")
        
        samples = [
            "What is permeability anisotropy?",
            "Explain the Darcy equation briefly.",
            "What causes stuck pipe during drilling?",
            "How does a blowout preventer work?",
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(samples):
            with cols[i % 2]:
                if st.button(f"📝 {q}", key=f"s{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{msg["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>🤖 AI:</strong><br>{msg["content"]}</div>', 
                       unsafe_allow_html=True)
    
    # Generate response for last user message if needed
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("🔄 Generating response..."):
            start = time.time()
            
            full_messages = [{"role": "system", "content": system_prompt}]
            # Only use last 3 exchanges for context (faster)
            full_messages.extend(st.session_state.messages[-6:])
            
            response = generate_response(model, tokenizer, full_messages, max_tokens, temperature)
            elapsed = time.time() - start
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f'<div class="assistant-message"><strong>🤖 AI:</strong><br>{response}<br><small style="color:#666">Generated in {elapsed:.1f}s</small></div>', 
                   unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    user_input = st.text_input("💬 Ask about petroleum engineering:", 
                               placeholder="e.g., What factors affect ROP?",
                               key="input")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Send", type="primary", use_container_width=True):
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.rerun()


if __name__ == "__main__":
    main()
