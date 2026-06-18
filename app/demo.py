import os
import asyncio
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from src.data_loader import load_data
from src.extraction import extract_all_facts
from src.faithfulness import summarize_and_verify

# Set page configuration with a premium dark/sleek theme layout
st.set_page_config(
    page_title="STR-Lens — AI-Powered AML Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (glassmorphism look, modern fonts, clear color-coding)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    .card-suspicion {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .card-parties {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .card-timeline {
        background-color: #f0f9ff;
        border-left: 5px solid #06b6d4;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .card-flags {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .badge-high {
        background-color: #22c55e;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-mid {
        background-color: #f59e0b;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #ef4444;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allowed_html=True)

# Helper function to plot a Plotly network graph representing transactions
def draw_transaction_network(sender, receiver, s_bank, r_bank, amount):
    edge_x = [0, 1]
    edge_y = [0, 0]
    
    node_x = [0, 1]
    node_y = [0, 0]
    node_text = [
        f"Sender: {sender}<br>Bank: {s_bank}",
        f"Receiver: {receiver}<br>Bank: {r_bank}"
    ]
    
    fig = go.Figure()
    
    # Add Edge link
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#94a3b8'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=["Sender", "Receiver"],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=[25, 25],
            color=['#1e3a8a', '#06b6d4'],
            line_width=2
        )
    ))
    
    # Annotate amount in the middle
    fig.add_annotation(
        x=0.5, y=0.05,
        text=f"NPR {amount:,.2f}",
        showarrow=False,
        font=dict(size=14, color="#1e293b", family="Inter")
    )
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Load dataset
@st.cache_data
def get_dataset():
    # If clean_data parquet exists, load it, otherwise run loader
    parquet_path = Path("data/clean_data.parquet")
    if parquet_path.exists():
        merged_df = pd.read_parquet(parquet_path)
    else:
        merged_df, _, _, _ = load_data()
    return merged_df

merged_df = get_dataset()

# Sidebar configuration
st.sidebar.image("https://img.icons8.com/color/96/search--v1.png", width=80)
st.sidebar.markdown("<h2 style='margin-top:0;'>STR-Lens</h2>", unsafe_allowed_html=True)
st.sidebar.info(
    "AI-Powered AML intelligence pipeline analyzing suspicious transactions. "
    "Designed with faithfulness verification and automated re-prompt loops."
)

api_key = st.sidebar.text_input("Groq API Key", type="password", placeholder="Simulation mode active if empty")

# Header section
st.markdown("<h1 class='main-header'>STR-Lens</h1>", unsafe_allowed_html=True)
st.markdown("<p class='tagline'>AI-Powered AML Intelligence Extraction & Verification</p>", unsafe_allowed_html=True)

# Define Tabs
tab1, tab2 = st.tabs(["🔍 Single Report Analysis", "📁 Batch Processing"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Transaction Data")
        
        # Load sample selector
        sample_options = ["Select a sample XML report...", "Report 1 - John Jensen", "Report 2 - Jeremy Martinez", "Report 3 - Ruth Clements"]
        selected_sample = st.selectbox("Preloaded Sample Reports", sample_options)
        
        default_xml = ""
        if selected_sample == "Report 1 - John Jensen":
            default_xml = open("reports/report_000001.xml", "r", encoding="utf-8").read()
        elif selected_sample == "Report 2 - Jeremy Martinez":
            default_xml = open("reports/report_000002.xml", "r", encoding="utf-8").read()
        elif selected_sample == "Report 3 - Ruth Clements":
            # Just read third report if it exists
            default_xml = open("reports/report_000003.xml", "r", encoding="utf-8").read()
            
        xml_input = st.text_area("Paste Raw XML Report Here", value=default_xml, height=300)
        
        btn_generate = st.button("Generate Summarized Intel Card", type="primary")
        
    with col2:
        st.subheader("Intelligence Extraction Card")
        
        if btn_generate and xml_input:
            # Parse record from the XML
            try:
                import xml.etree.ElementTree as ET
                from src.data_loader import extract_narrative_text
                
                root = ET.fromstring(xml_input)
                report_id = root.find("report_id").text if root.find("report_id") is not None else "Unknown ID"
                reason = root.find("reason").text if root.find("reason") is not None else ""
                
                tx_elem = root.find("transaction")
                if tx_elem is not None:
                    tx_num = tx_elem.find("transactionnumber").text if tx_elem.find("transactionnumber") is not None else ""
                    comments = tx_elem.find("comments").text if tx_elem.find("comments") is not None else ""
                    amount_local = float(tx_elem.find("amount_local").text) if tx_elem.find("amount_local") is not None else 0.0
                    parts = tx_num.split("-")
                    row_index = int(parts[-1]) if parts else -1
                else:
                    row_index = -1
                    comments = ""
                    amount_local = 0.0
                    
                raw_text = extract_narrative_text(root)
                
                # Fetch matching row from dataframe to retrieve full transactional context
                matched_row = merged_df[merged_df["row_index"] == row_index]
                if not matched_row.empty:
                    record_dict = matched_row.iloc[0].to_dict()
                else:
                    record_dict = {
                        "str_id": report_id,
                        "xml_amount_local": amount_local,
                        "xml_reason": reason,
                        "xml_comments": comments,
                        "narrative_text": raw_text
                    }
                
                # Spinner animation states
                status_placeholder = st.empty()
                with status_placeholder.container():
                    with st.spinner("Step 1/3: Extracting entities..."):
                        facts = extract_all_facts(raw_text)
                        time.sleep(0.5)
                    with st.spinner("Step 2/3: Generating summary..."):
                        # Run async summarizer & verifier
                        res = asyncio.run(summarize_and_verify(record_dict, api_key=api_key))
                        time.sleep(0.5)
                    with st.spinner("Step 3/3: Verifying faithfulness..."):
                        time.sleep(0.5)
                status_placeholder.empty()
                
                # Render metrics dashboard row
                score = res["faithfulness_score"]
                if score >= 0.85:
                    badge_html = f"<span class='badge-high'>Faithfulness: {score:.0%}</span>"
                elif score >= 0.70:
                    badge_html = f"<span class='badge-mid'>Faithfulness: {score:.0%}</span>"
                else:
                    badge_html = f"<span class='badge-low'>Faithfulness: {score:.0%}</span>"
                    
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;'>
                    <div><h4>Report ID: {res['str_id']}</h4></div>
                    <div>{badge_html}</div>
                </div>
                """, unsafe_allowed_html=True)
                
                st.markdown(f"**Word Count:** {res['word_count']} words | **Reading Time:** ~{max(1, int(res['word_count']/150))} min | **Processing Time:** {res['processing_time_ms']} ms")
                
                if res["human_review"]:
                    st.warning(f"⚠️ Manual Verification Recommended. Missing facts: {', '.join(res['missing_facts'])}")
                
                # Color-coded sections
                st.markdown(f"""
                <div class='card-suspicion'>
                    <strong>Suspicion Type</strong><br>{res['suspicion_type']}
                </div>
                <div class='card-parties'>
                    <strong>Parties</strong><br>{res['parties']}
                </div>
                <div class='card-timeline'>
                    <strong>Transaction Summary</strong><br>{res['transaction_summary']}
                </div>
                <div class='card-flags'>
                    <strong>Key Red Flags</strong><br>{res['red_flags']}
                </div>
                """, unsafe_allowed_html=True)
                
                # Toggle views
                with st.expander("Compare with Original Narrative"):
                    st.write(raw_text)
                    
                with st.expander("Visualise Transaction Network"):
                    s_acc = record_dict.get("sender_account_number", "Sender Account")
                    r_acc = record_dict.get("receiver_account_number", "Receiver Account")
                    s_bank = record_dict.get("sender_institution", "Sender Bank")
                    r_bank = record_dict.get("receiver_institution", "Receiver Bank")
                    amt = record_dict.get("xml_amount_local", amount_local)
                    fig = draw_transaction_network(s_acc, r_acc, s_bank, r_bank, amt)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error parsing XML or generating summary: {e}")
        else:
            st.info("Input transaction XML details on the left and click generate.")

with tab2:
    st.subheader("Batch Upload & Bulk Summarisation")
    st.write("Upload a CSV file containing transaction records to run bulk summarization.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} records.")
        
        # Merge batch records to the main dataset to inherit structure
        # In a real batch pipeline, data_loader.py would merge it first
        to_process = pd.merge(batch_df, merged_df, on="row_index", how="inner")
        st.write(f"Successfully aligned {len(to_process)} records for pipeline execution.")
        
        if st.button("Start Batch Processing", type="primary"):
            from src.batch_pipeline import run_batch_pipeline
            
            output_tmp = "results/batch_outputs.jsonl"
            progress_bar = st.progress(0)
            
            # Since Streamlit execution is synchronous, run batch pipeline
            results = asyncio.run(run_batch_pipeline(to_process, output_tmp, sample_limit=len(to_process)))
            
            # Load result records and present them
            results_df = pd.DataFrame(results)
            st.write("Batch Processing Complete!")
            st.dataframe(results_df[["str_id", "suspicion_type", "faithfulness_score", "word_count"]])
            
            # Download results as CSV
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="str_lens_batch_results.csv",
                mime="text/csv"
            )
