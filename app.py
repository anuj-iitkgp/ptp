"""
Prompt-to-Prompt Text-Guided Image Editing Studio
Research Dashboard and Interactive Demo

Inspired by:
"Prompt-to-Prompt Image Editing with Cross Attention Control"
Hertz et al. (Google Research, 2022)

Author: Anuj Yadav
Institution: Indian Institute of Technology Kharagpur (IIT KGP)
GitHub: https://github.com/anuj-iitkgp/ptp
"""

import os
import io
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from PIL import Image
import streamlit as st

from src.models.loader import load_model
from src.prompts.tokenizer import PromptTokenizer
from src.prompts.alignment import align_tokens
from src.pipeline.editing import edit_word_swap, edit_prompt_refinement, edit_reweight
from src.pipeline.generation import generate_with_attention
from src.pipeline.inversion import DDIMInversion
from src.attention.visualization import (
    get_token_attention_map,
    create_heatmap_image,
    create_attention_overlay,
    create_token_comparison_grid,
)
from src.evaluation.comparison import generate_evaluation_report
from src.utils.device import get_device_info, get_device
from src.utils.image import create_side_by_side

# Streamlit Page Configuration
st.set_page_config(
    page_title="Prompt-to-Prompt Studio | Anuj Yadav (IIT KGP)",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 15px 0 25px 0;
        border-bottom: 2px solid #2d3748;
        margin-bottom: 25px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f7fafc;
        margin-bottom: 4px;
    }
    .author-badge {
        font-size: 0.95rem;
        color: #a0aec0;
    }
    .author-highlight {
        color: #48bb78;
        font-weight: 600;
    }
    .metric-card {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .token-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .token-unchanged { background: #2d3748; color: #e2e8f0; }
    .token-changed { background: #9b2c2c; color: #feb2b2; font-weight: bold; }
    .token-added { background: #22543d; color: #9ae6b4; font-weight: bold; }
    .stDownloadButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_pipeline(model_id: str, low_memory: bool = False):
    """Cache the loaded diffusion model to avoid repeated disk reads."""
    return load_model(model_id=model_id, low_memory_mode=low_memory)


# Sidebar Setup
with st.sidebar:
    # Logo & Attribution
    logo_path = os.path.join("assets", "iitkgp_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)

    st.markdown("### **Prompt-to-Prompt Studio**")
    st.markdown(
        "**Researcher:** [Anuj Yadav](https://github.com/anuj-iitkgp)  \n"
        "**Affiliation:** Indian Institute of Technology Kharagpur  \n"
        "**Repository:** [`anuj-iitkgp/ptp`](https://github.com/anuj-iitkgp/ptp)"
    )
    st.markdown("---")

    # Diagnostics
    st.markdown("#### ⚙️ Hardware Diagnostics")
    dev_info = get_device_info()
    st.markdown(f"**Device:** `{dev_info['device_name']}`")
    st.markdown(f"**Backend:** `{dev_info['device_type'].upper()}`")
    st.markdown(f"**Memory:** `{dev_info['vram_gb']}`")
    st.markdown(f"**PyTorch:** `{torch.__version__}`")

    st.markdown("---")
    st.markdown("#### 🎛️ Pipeline Settings")
    model_choice = st.selectbox(
        "Model Checkpoint",
        ["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4"],
        index=0,
    )
    low_memory = st.checkbox("Low Memory Mode", value=False)


# Main Header
col_header_1, col_header_2 = st.columns([1, 6])
with col_header_1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=95)
with col_header_2:
    st.markdown(
        """
        <div class="main-title">Prompt-to-Prompt Image Editing Studio</div>
        <div class="author-badge">
            Developed by <span class="author-highlight">Anuj Yadav</span> &bull; 
            Indian Institute of Technology Kharagpur &bull; 
            Cross-Attention Control in Latent Diffusion
        </div>
        """,
        unsafe_allow_html=True,
    )

# Load pipeline with spinner
with st.spinner("Initializing Diffusion Pipeline..."):
    pipe = get_cached_pipeline(model_choice, low_memory=low_memory)
tokenizer = PromptTokenizer(pipe.tokenizer, pipe.text_encoder)

# Main Navigation Tabs
tab_edit, tab_attn, tab_inv, tab_eval, tab_exp = st.tabs(
    [
        "🎨 Text-to-Image P2P Editing",
        "🔍 Cross-Attention Visualization",
        "🖼️ Real Image Inversion",
        "📊 Quantitative Evaluation",
        "🧪 Preset Research Experiments",
    ]
)

# Session State Initialization
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "edited_image" not in st.session_state:
    st.session_state.edited_image = None
if "controller" not in st.session_state:
    st.session_state.controller = None
if "alignment" not in st.session_state:
    st.session_state.alignment = None
if "source_prompt" not in st.session_state:
    st.session_state.source_prompt = "A photo of a dog sitting on a beach"
if "target_prompt" not in st.session_state:
    st.session_state.target_prompt = "A photo of a cat sitting on a beach"


# ==========================================
# TAB 1: TEXT-TO-IMAGE P2P EDITING
# ==========================================
with tab_edit:
    col_input, col_ctrl = st.columns([3, 2])

    with col_input:
        st.markdown("#### 1. Prompt Specification")
        src_p = st.text_input(
            "Original Source Prompt",
            value=st.session_state.source_prompt,
            key="src_prompt_input",
        )
        tgt_p = st.text_input(
            "Edited Target Prompt",
            value=st.session_state.target_prompt,
            key="tgt_prompt_input",
        )

        method = st.selectbox(
            "Editing Method",
            ["Word Swap", "Prompt Refinement", "Attention Re-weighting"],
            index=0,
            help="Select the mathematical attention intervention to apply.",
        )

        # Dynamic Alignment Preview
        src_meta = tokenizer.tokenize(src_p)
        tgt_meta = tokenizer.tokenize(tgt_p)
        alignment_preview = align_tokens(src_meta["tokens"], tgt_meta["tokens"])

        st.markdown("**Token Alignment Inspection:**")
        preview_html = ""
        for ch in alignment_preview.get("changed_tokens", []):
            preview_html += f"<span class='token-badge token-changed'>{ch['src']} → {ch['tgt']}</span> "
        for ad in alignment_preview.get("added_tokens", []):
            preview_html += f"<span class='token-badge token-added'>+ {ad['tgt']}</span> "
        if not alignment_preview.get("changed_tokens") and not alignment_preview.get("added_tokens"):
            preview_html = "<span style='color: #a0aec0; font-size: 0.9rem;'>No token differences detected.</span>"
        st.markdown(preview_html, unsafe_allow_html=True)

    with col_ctrl:
        st.markdown("#### 2. Diffusion & Attention Hyperparameters")
        c1, c2 = st.columns(2)
        with c1:
            seed = st.number_input("Seed", value=42, min_value=0, max_value=999999)
            steps = st.slider("Inference Steps", min_value=10, max_value=100, value=30, step=5)
        with c2:
            guidance = st.slider("Guidance Scale (CFG)", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
            self_replace = st.slider(
                "Self-Attention Injection (τ_self)",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Fraction of steps to inject original self-attention for layout preservation.",
            )

        if method in ("Word Swap", "Prompt Refinement"):
            cross_replace = st.slider(
                "Cross-Attention Replacement (τ_cross)",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Fraction of steps to replace cross-attention for mapped tokens.",
            )
        elif method == "Attention Re-weighting":
            st.markdown("**Attention Equalizer Weights:**")
            eq_word = st.text_input("Target Word to Re-weight", value="dog")
            eq_weight = st.slider(f"Multiplier for '{eq_word}'", min_value=0.1, max_value=4.0, value=2.0, step=0.1)

    # Action Buttons
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        run_edit_btn = st.button("✨ Apply Genuine Prompt-to-Prompt Edit", type="primary", use_container_width=True)
    with col_btn2:
        run_gen_btn = st.button("🖼️ Generate Original Only", use_container_width=True)

    # Execution Flow
    if run_edit_btn:
        progress_bar = st.progress(0, text="Executing Joint Attention Controlled Diffusion...")
        start_time = time.time()

        def update_progress(step, total):
            progress_bar.progress(step / total, text=f"Diffusion Step {step}/{total}...")

        try:
            if method == "Word Swap":
                img_orig, img_edit, ctrl, align = edit_word_swap(
                    pipe=pipe,
                    source_prompt=src_p,
                    target_prompt=tgt_p,
                    cross_replace_steps=cross_replace,
                    self_replace_steps=self_replace,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    callback=update_progress,
                )
            elif method == "Prompt Refinement":
                img_orig, img_edit, ctrl, align = edit_prompt_refinement(
                    pipe=pipe,
                    source_prompt=src_p,
                    target_prompt=tgt_p,
                    cross_replace_steps=cross_replace,
                    self_replace_steps=self_replace,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    callback=update_progress,
                )
            else:  # Attention Re-weighting
                img_orig, img_edit, ctrl, align = edit_reweight(
                    pipe=pipe,
                    prompt=src_p,
                    weights_dict={eq_word: eq_weight},
                    self_replace_steps=self_replace,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    seed=seed,
                    callback=update_progress,
                )

            progress_bar.empty()
            elapsed = round(time.time() - start_time, 2)
            st.success(f"Generation & Editing completed in {elapsed}s!")

            # Store in session state
            st.session_state.original_image = img_orig
            st.session_state.edited_image = img_edit
            st.session_state.controller = ctrl
            st.session_state.alignment = align
            st.session_state.source_prompt = src_p
            st.session_state.target_prompt = tgt_p

        except Exception as e:
            progress_bar.empty()
            st.error(f"Error during execution: {str(e)}")

    elif run_gen_btn:
        progress_bar = st.progress(0, text="Generating Source Image...")

        def update_progress(step, total):
            progress_bar.progress(step / total, text=f"Step {step}/{total}...")

        try:
            img_orig, store = generate_with_attention(
                pipe=pipe,
                prompt=src_p,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
                callback=update_progress,
            )
            progress_bar.empty()
            st.session_state.original_image = img_orig
            st.session_state.edited_image = None
            st.session_state.controller = store
            st.session_state.source_prompt = src_p
            st.success("Original image generated successfully!")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Generation error: {str(e)}")

    # Display Results
    if st.session_state.original_image is not None:
        st.markdown("---")
        st.markdown("### 🖼️ Results")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"**Original Image** (`{st.session_state.source_prompt}`)")
            st.image(st.session_state.original_image, use_container_width=True)

            buf1 = io.BytesIO()
            st.session_state.original_image.save(buf1, format="PNG")
            st.download_button(
                "⬇️ Download Original",
                data=buf1.getvalue(),
                file_name="original.png",
                mime="image/png",
            )

        with res_col2:
            if st.session_state.edited_image is not None:
                st.markdown(f"**Edited Image (Prompt-to-Prompt)** (`{st.session_state.target_prompt}`)")
                st.image(st.session_state.edited_image, use_container_width=True)

                buf2 = io.BytesIO()
                st.session_state.edited_image.save(buf2, format="PNG")
                st.download_button(
                    "⬇️ Download Edited",
                    data=buf2.getvalue(),
                    file_name="edited_p2p.png",
                    mime="image/png",
                )

                # Side-by-side combined download
                sbs = create_side_by_side(st.session_state.original_image, st.session_state.edited_image)
                buf_sbs = io.BytesIO()
                sbs.save(buf_sbs, format="PNG")
                st.download_button(
                    "⬇️ Download Side-by-Side Comparison",
                    data=buf_sbs.getvalue(),
                    file_name="comparison_p2p.png",
                    mime="image/png",
                )


# ==========================================
# TAB 2: ATTENTION VISUALIZATION
# ==========================================
with tab_attn:
    st.markdown("### 🔍 Cross-Attention Spatial Heatmaps")
    st.markdown(
        "Cross-attention maps quantify the spatial binding of each word token to image pixels. "
        "Prompt-to-Prompt replaces these maps during early diffusion steps to preserve scene composition."
    )

    if st.session_state.controller is None or st.session_state.original_image is None:
        st.info("Run generation or editing in the first tab to visualize attention maps.")
    else:
        ctrl = st.session_state.controller
        meta = tokenizer.tokenize(st.session_state.source_prompt)
        clean_tokens = meta["clean_tokens"][: meta["num_real_tokens"]]

        col_tok_select, col_cmap = st.columns([3, 1])
        with col_tok_select:
            selected_token_str = st.selectbox(
                "Select Prompt Token to Inspect",
                clean_tokens,
                index=min(5, len(clean_tokens) - 1),
            )
            selected_idx = clean_tokens.index(selected_token_str)
        with col_cmap:
            cmap_choice = st.selectbox("Heatmap Colormap", ["turbo", "jet", "viridis"], index=0)

        # Compute heatmaps
        img_orig = st.session_state.original_image
        w, h = img_orig.size
        map_src = get_token_attention_map(ctrl, selected_idx, batch_idx=0, image_size=(w, h))

        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            st.markdown(f"**Original Image**")
            st.image(img_orig, use_container_width=True)
        with vcol2:
            st.markdown(f"**Attention Heatmap: `{selected_token_str}`**")
            heat_img = create_heatmap_image(map_src, colormap_name=cmap_choice)
            st.image(heat_img, use_container_width=True)
        with vcol3:
            st.markdown(f"**Spatial Overlay**")
            overlay = create_attention_overlay(img_orig, map_src, colormap_name=cmap_choice)
            st.image(overlay, use_container_width=True)

        # Comparison if edited image exists
        if st.session_state.edited_image is not None and st.session_state.alignment is not None:
            st.markdown("---")
            st.markdown("#### Original vs Edited Token Comparison")
            ch_list = st.session_state.alignment.get("changed_tokens", [])
            if ch_list:
                ch = ch_list[0]
                st.markdown(f"Swapped Pair: **`{ch['src']}` (Original)** $\\to$ **`{ch['tgt']}` (Edited)**")
                grid = create_token_comparison_grid(
                    st.session_state.original_image,
                    st.session_state.edited_image,
                    ctrl,
                    ch["src_idx"],
                    ch["tgt_idx"],
                    ch["src"],
                    ch["tgt"],
                )
                st.image(grid, use_container_width=True)


# ==========================================
# TAB 3: REAL IMAGE INVERSION (ADVANCED)
# ==========================================
with tab_inv:
    st.markdown("### 🖼️ Real Image Inversion & Prompt-to-Prompt")
    st.markdown(
        "Upload a real photo, invert it to initial noise latent $z_T$ using **DDIM Inversion**, "
        "and apply cross-attention controlled prompt editing."
    )

    uploaded_file = st.file_uploader("Upload Input Image", type=["png", "jpg", "jpeg"])
    inv_col1, inv_col2 = st.columns(2)

    with inv_col1:
        inv_prompt = st.text_input("Source Description of Uploaded Image", value="A photo of a dog")
    with inv_col2:
        inv_edit_prompt = st.text_input("Edited Target Prompt", value="A photo of a cat")

    if uploaded_file is not None:
        real_img = Image.open(uploaded_file).convert("RGB")
        st.image(real_img, caption="Input Real Image", width=300)

        if st.button("🔄 Invert Real Image & Apply Prompt-to-Prompt"):
            inv_bar = st.progress(0, text="Inverting Image with Reversed DDIM...")
            ddim_inv = DDIMInversion(pipe)

            def update_inv(step, total):
                inv_bar.progress(step / total, text=f"Inversion Step {step}/{total}...")

            z_T, _ = ddim_inv.invert(
                image=real_img,
                prompt=inv_prompt,
                num_inference_steps=30,
                guidance_scale=1.0,
                callback=update_inv,
            )

            inv_bar.progress(1.0, text="Executing Attention-Controlled Editing on Inverted Latent...")
            img_orig_rec, img_edit_rec, ctrl_rec, _ = edit_word_swap(
                pipe=pipe,
                source_prompt=inv_prompt,
                target_prompt=inv_edit_prompt,
                initial_latents=z_T,
                num_inference_steps=30,
                cross_replace_steps=0.7,
                self_replace_steps=0.3,
            )
            inv_bar.empty()

            st.success("Real Image Inversion and Editing Succeeded!")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.markdown("**Reconstructed Original**")
                st.image(img_orig_rec, use_container_width=True)
            with rcol2:
                st.markdown("**Edited Real Image**")
                st.image(img_edit_rec, use_container_width=True)


# ==========================================
# TAB 4: QUANTITATIVE EVALUATION
# ==========================================
with tab_eval:
    st.markdown("### 📊 Quantitative Evaluation Metrics")
    st.markdown(
        "Prompt-to-Prompt balances two essential objectives:  \n"
        "1. **Structure Preservation**: Background, composition, and physical geometry should remain identical.  \n"
        "2. **Edit Effectiveness**: The targeted semantic object should accurately reflect the new text description."
    )

    if st.session_state.original_image is not None and st.session_state.edited_image is not None:
        report = generate_evaluation_report(
            st.session_state.original_image,
            st.session_state.edited_image,
            st.session_state.source_prompt,
            st.session_state.target_prompt,
            pipe=pipe,
        )

        sf = report["structural_fidelity"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SSIM (Structural Similarity)", f"{sf['ssim']}", help="Range [-1, 1]. Higher indicates identical background structure.")
        m2.metric("PSNR (Peak SNR)", f"{sf['psnr_db']} dB", help="Higher indicates closer pixel intensity distribution.")
        m3.metric("Edge Preservation", f"{sf['edge_similarity']}", help="Cosine similarity of Sobel spatial contour gradients.")
        m4.metric("L1 Pixel Distance", f"{sf['l1_diff']}", help="Lower indicates subtle, localized pixel edits.")

        st.markdown("#### Metric Interpretation")
        st.info(
            f"**Evaluation Summary:** {report['summary']}.  \n"
            "An SSIM of > 0.65 with high Edge Similarity confirms that the background geometry and global scene layout "
            "are strictly preserved, validating genuine cross-attention control rather than independent generation."
        )
    else:
        st.info("Perform an edit in Tab 1 to calculate quantitative metrics.")


# ==========================================
# TAB 5: PRESET RESEARCH EXPERIMENTS
# ==========================================
with tab_exp:
    st.markdown("### 🧪 Canonical Research Experiments")
    st.markdown("Select any benchmark experiment from the paper to automatically populate parameters:")

    exp_presets = {
        "Experiment 1: Word Swap (Dog → Cat)": {
            "src": "A photo of a dog sitting on a beach",
            "tgt": "A photo of a cat sitting on a beach",
            "method": "Word Swap",
            "seed": 42,
        },
        "Experiment 2: Object Replacement (Red Car → Blue Car)": {
            "src": "A red car parked beside a house",
            "tgt": "A blue car parked beside a house",
            "method": "Word Swap",
            "seed": 123,
        },
        "Experiment 3: Prompt Refinement (Add Specifications)": {
            "src": "A dog in a park",
            "tgt": "A small golden dog in a park",
            "method": "Prompt Refinement",
            "seed": 88,
        },
        "Experiment 4: Attention Re-weighting (Hat Amplification)": {
            "src": "A portrait of a woman wearing a hat",
            "tgt": "A portrait of a woman wearing a hat",
            "method": "Attention Re-weighting",
            "seed": 100,
        },
        "Experiment 5: Multi-token Swap (Retriever → Shepherd)": {
            "src": "A golden retriever running through a green field",
            "tgt": "A German shepherd running through a green field",
            "method": "Word Swap",
            "seed": 55,
        },
    }

    preset_choice = st.selectbox("Choose Experiment Preset", list(exp_presets.keys()))
    if st.button("Load Preset"):
        p = exp_presets[preset_choice]
        st.session_state.source_prompt = p["src"]
        st.session_state.target_prompt = p["tgt"]
        st.success(f"Loaded '{preset_choice}'! Switch to Tab 1 to run.")
