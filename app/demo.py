import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Add project root to PYTHONPATH

import asyncio
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from src.data_loader import load_data
from src.extraction import extract_all_facts
from src.faithfulness import summarize_and_verify
from src.visualization import draw_transaction_network

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
        color: #1e293b;
    }
    .card-parties {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    .card-timeline {
        background-color: #f0f9ff;
        border-left: 5px solid #06b6d4;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #1e293b;
    }
    .card-flags {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #1e293b;
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
""", unsafe_allow_html=True)  # FIX 3: was unsafe_allowed_html (wrong kwarg name for st.markdown)

# NOTE: draw_transaction_network is imported from src.visualization above.
# A local function with the same name used to live here, which silently
# shadowed that import (Python keeps the later definition), so the package's
# visualization code was never actually reachable from this script. Removed
# so the import is the one that runs.

def humanize_xml_report(root):
    """
    Builds a labeled, human-readable summary from a parsed XML <report>
    element, instead of relying on a flattened text dump. Pulls the same
    fields demo.py already parses elsewhere (report_id, reason, transaction
    sub-fields) plus a few extra ones if present (filer, subject/party
    details), and presents each with a clear label. Missing fields are
    simply omitted rather than shown as blank or "Unknown" noise.

    Returns a list of (label, value) tuples so the caller can render them
    however it likes (markdown, table, etc.) rather than baking in a layout
    here.
    """
    def text_of(elem, tag):
        node = elem.find(tag) if elem is not None else None
        return node.text.strip() if node is not None and node.text and node.text.strip() else None

    rows = []

    report_id = text_of(root, "report_id")
    if report_id:
        rows.append(("Report ID", report_id))

    reason = text_of(root, "reason")
    if reason:
        rows.append(("Reason Filed", reason))

    tx_elem = root.find("transaction")
    if tx_elem is not None:
        tx_num = text_of(tx_elem, "transactionnumber")
        if tx_num:
            rows.append(("Transaction Number", tx_num))

        tx_type = text_of(tx_elem, "type") or text_of(tx_elem, "transactiontype")
        if tx_type:
            rows.append(("Transaction Type", tx_type))

        amount = text_of(tx_elem, "amount_local")
        currency = text_of(tx_elem, "currency") or text_of(tx_elem, "currency_local")
        if amount:
            try:
                amount_fmt = f"{float(amount):,.2f}"
            except ValueError:
                amount_fmt = amount
            rows.append(("Amount", f"{amount_fmt} {currency}".strip()))

        tx_date = text_of(tx_elem, "transactiondate") or text_of(tx_elem, "date")
        if tx_date:
            rows.append(("Transaction Date", tx_date))

        comments = text_of(tx_elem, "comments")
        if comments:
            rows.append(("Comments / Narrative", comments))

    # Party/account/institution details live in the nested
    # t_from_my_client/from_account and t_to/to_account elements, not in
    # top-level tags like <sender>/<receiver> -- those don't exist in this
    # schema, which is why the previous best-effort lookup here never found
    # anything. extract_structured_facts walks the real tags.
    from src.data_loader import extract_structured_facts
    structured = extract_structured_facts(root)

    if structured.get("sender_account_name") or structured.get("sender_institution"):
        sender_bits = []
        if structured.get("sender_account_name"):
            sender_bits.append(structured["sender_account_name"])
        if structured.get("sender_institution"):
            sender_bits.append(f"({structured['sender_institution']})")
        if structured.get("sender_account_number"):
            sender_bits.append(f"acct {structured['sender_account_number']}")
        rows.append(("Sender", " ".join(sender_bits)))

    if structured.get("receiver_account_name") or structured.get("receiver_institution"):
        receiver_bits = []
        if structured.get("receiver_account_name"):
            receiver_bits.append(structured["receiver_account_name"])
        if structured.get("receiver_institution"):
            receiver_bits.append(f"({structured['receiver_institution']})")
        if structured.get("receiver_account_number"):
            receiver_bits.append(f"acct {structured['receiver_account_number']}")
        rows.append(("Receiver", " ".join(receiver_bits)))

    if structured.get("foreign_currency_code") and structured.get("foreign_amount"):
        rows.append(("Foreign Currency Leg", f"{structured['foreign_currency_code']} {structured['foreign_amount']}"))

    return rows


# Load dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]

@st.cache_data
def get_dataset():
    # Resolved relative to the project root (one level up from this file's
    # own folder), not the current working directory -- otherwise this only
    # finds the parquet file when Streamlit happens to be launched from the
    # project root, and silently falls through to the slower load_data()
    # path (or raises FileNotFoundError there too) when launched from app/.
    parquet_path = PROJECT_ROOT / "data" / "clean_data.parquet"
    if parquet_path.exists():
        merged_df = pd.read_parquet(parquet_path)
    else:
        merged_df, _, _, _ = load_data()
    return merged_df

merged_df = get_dataset()

# Sidebar configuration
st.sidebar.image("https://img.icons8.com/color/96/search--v1.png", width=80)
st.sidebar.markdown("<h2 style='margin-top:0;'>STR-Lens</h2>", unsafe_allow_html=True)
st.sidebar.info(
    "AI-Powered AML intelligence pipeline analyzing suspicious transactions. "
    "Designed with faithfulness verification and automated re-prompt loops."
)

import os
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Retrieve API key from environment; allow manual override via sidebar
default_key = os.getenv("GROQ_API_KEY", "")
api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="Simulation mode active if empty",
    value=default_key
)

# Header section
st.markdown("<h1 class='main-header'>STR-Lens</h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>AI-Powered AML Intelligence Extraction & Verification</p>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2 = st.tabs(["🔍 Single Report Analysis", "📁 Batch Processing"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Transaction Data")
        
        sample_options = ["Select a sample XML report...", "Report 1 - John Jensen", "Report 2 - Jeremy Martinez", "Report 3 - Ruth Clements"]
        selected_sample = st.selectbox("Preloaded Sample Reports", sample_options)
        
        default_xml = ""
        # Resolved relative to PROJECT_ROOT for the same reason as the
        # parquet path above -- these would otherwise only be found when
        # launched from the project root, not from app/.
        if selected_sample == "Report 1 - John Jensen":
            default_xml = (PROJECT_ROOT / "reports" / "report_000001.xml").read_text(encoding="utf-8")
        elif selected_sample == "Report 2 - Jeremy Martinez":
            default_xml = (PROJECT_ROOT / "reports" / "report_000002.xml").read_text(encoding="utf-8")
        elif selected_sample == "Report 3 - Ruth Clements":
            default_xml = (PROJECT_ROOT / "reports" / "report_000003.xml").read_text(encoding="utf-8")
            
        xml_input = st.text_area("Paste Raw XML Report Here", value=default_xml, height=300)
        
        btn_generate = st.button("Generate Summarized Intel Card", type="primary")
        
    with col2:
        st.subheader("Intelligence Extraction Card")
        
        if btn_generate and xml_input:
            try:
                import xml.etree.ElementTree as ET
                from src.data_loader import extract_narrative_text, extract_structured_facts
                
                root = ET.fromstring(xml_input)
                report_id = root.find("report_id").text if root.find("report_id") is not None else "Unknown ID"
                reason = root.find("reason").text if root.find("reason") is not None else ""
                
                tx_elem = root.find("transaction")
                if tx_elem is not None:
                    tx_num = tx_elem.find("transactionnumber").text if tx_elem.find("transactionnumber") is not None else ""
                    comments = tx_elem.find("comments").text if tx_elem.find("comments") is not None else ""
                    amount_local = float(tx_elem.find("amount_local").text) if tx_elem.find("amount_local") is not None else 0.0
                    # Real field names per the actual XML schema: there is no
                    # "transactiondate" or "type" tag (those were a guess that
                    # didn't match the schema, which is why the context strip
                    # showed "Unknown" for Date and Type). The transaction
                    # timestamp is date_transaction; the closest equivalent to
                    # a human-readable type is transmode_comment.
                    tx_date_elem = tx_elem.find("date_transaction")
                    tx_date = tx_date_elem.text if tx_date_elem is not None and tx_date_elem.text else None
                    tx_type_elem = tx_elem.find("transmode_comment")
                    tx_type = tx_type_elem.text if tx_type_elem is not None and tx_type_elem.text else None
                    currency_elem = root.find("currency_code_local")
                    xml_currency = currency_elem.text if currency_elem is not None and currency_elem.text else None
                    parts = tx_num.split("-")
                    row_index = int(parts[-1]) if parts else -1
                else:
                    row_index = -1
                    comments = ""
                    amount_local = 0.0
                    tx_date = None
                    tx_type = None
                    xml_currency = None
                    
                raw_text = extract_narrative_text(root)
                
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

                # Structured party/account/institution facts, parsed fresh
                # from the pasted XML (see extract_structured_facts). These
                # are what was missing entirely before -- record_dict only
                # ever carried report_id/reason/comments/amount_local, so
                # neither the faithfulness checklist nor the LLM prompt had
                # any way to know who the parties were, which is the direct
                # cause of "Unknown Parties" showing up in the card.
                record_dict.update(extract_structured_facts(root))

                # Always set these from the freshly-parsed XML, regardless of
                # which branch above ran. merged_df's columns come from
                # transactions.csv / ml_features.csv, which may not carry a
                # transaction date, mode/type, or currency at all -- so this
                # is the one place that's guaranteed to reflect the actual
                # uploaded report rather than silently staying unset.
                if tx_date:
                    record_dict["date_transaction"] = tx_date
                if tx_type:
                    record_dict["transmode_comment"] = tx_type
                if xml_currency:
                    record_dict["currency_code_local"] = xml_currency
                
                status_placeholder = st.empty()
                with status_placeholder.container():
                    with st.spinner("Step 1/3: Extracting entities..."):
                        facts = extract_all_facts(raw_text)
                        time.sleep(0.5)
                    with st.spinner("Step 2/3: Generating summary..."):
                        res = asyncio.run(summarize_and_verify(record_dict, api_key=api_key))
                        time.sleep(0.5)
                    with st.spinner("Step 3/3: Verifying faithfulness..."):
                        time.sleep(0.5)
                status_placeholder.empty()
                
                score = res["faithfulness_score"]
                if score >= 0.85:
                    badge_html = f"<span class='badge-high'>Faithfulness: {score:.0%}</span>"
                elif score >= 0.70:
                    badge_html = f"<span class='badge-mid'>Faithfulness: {score:.0%}</span>"
                else:
                    badge_html = f"<span class='badge-low'>Faithfulness: {score:.0%}</span>"
                    
                # Always render a basic card even if parsing fails
                st.subheader("Intelligence Extraction Card")
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;'>
                    <div><h4>Report ID: {res.get('str_id', 'N/A')}</h4></div>
                    <div>{badge_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Render detailed sections safely
                st.markdown(f"""
                <div class='card-suspicion'>
                    <strong>Suspicion Type</strong><br>{res.get('suspicion_type', 'Unknown')}
                </div>
                <div class='card-parties'>
                    <strong>Parties</strong><br>{res.get('parties', 'Unknown')}
                </div>
                <div class='card-timeline'>
                    <strong>Transaction Summary</strong><br>{res.get('transaction_summary', 'Unknown')}
                </div>
                <div class='card-flags'>
                    <strong>Key Red Flags</strong><br>{res.get('red_flags', 'None')}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Word Count:** {res['word_count']} words | **Reading Time:** ~{max(1, int(res['word_count']/150))} min | **Processing Time:** {res['processing_time_ms']} ms")
                
                if res["human_review"]:
                    st.warning(f"⚠️ Manual Verification Recommended. Missing facts: {', '.join(res['missing_facts'])}")
                
                # Duplicate raw card rendering removed – safe card above handles display
                with st.expander("Compare with Original Narrative"):
                    summary_rows = humanize_xml_report(root)
                    if summary_rows:
                        st.markdown("**Report Details**")
                        for label, value in summary_rows:
                            st.markdown(f"- **{label}:** {value}")
                    if raw_text:
                        st.markdown("**Narrative Text Used for Extraction**")
                        st.write(raw_text)
                    else:
                        st.info("No free-text narrative field was found in this report's XML.")
                    
                with st.expander("Transaction Flow Network", expanded=True):
                    s_acc = record_dict.get("sender_account_number")
                    r_acc = record_dict.get("receiver_account_number")
                    s_bank = record_dict.get("sender_institution")
                    r_bank = record_dict.get("receiver_institution")
                    amt = record_dict.get("xml_amount_local")
                    currency = record_dict.get("currency_code_local") or "NPR"
                    # Real keys confirmed against the actual XML schema (see
                    # record_dict population above): date_transaction and
                    # transmode_comment, not the guessed "transactiondate"/
                    # "type" names that were previously always missing.
                    tx_date = record_dict.get("date_transaction") or record_dict.get("transactiondate") or record_dict.get("date")
                    tx_type = record_dict.get("transmode_comment") or record_dict.get("transaction_type") or record_dict.get("type")

                    if s_acc and r_acc and amt is not None:
                        # Context strip above the chart: the diagram alone shows
                        # *where* money moved, not *when* or *why* -- these blocks
                        # give that context at a glance without digging into the
                        # expanders above. Using markdown instead of st.metric:
                        # st.metric doesn't wrap long text, so values like
                        # "NPR 535,368.64" or a full date were getting clipped to
                        # "NPR ..." inside narrow equal-width columns.
                        same_bank = bool(s_bank) and (s_bank == r_bank)

                        display_date = tx_date
                        if tx_date:
                            try:
                                from datetime import datetime
                                display_date = datetime.fromisoformat(tx_date).strftime("%d %b %Y, %H:%M")
                            except ValueError:
                                display_date = tx_date  # leave as-is if format is unexpected

                        context_items = [
                            ("Amount", f"{currency} {float(amt):,.2f}"),
                            ("Date", display_date if display_date else "Unknown"),
                            ("Type", tx_type if tx_type else "Unknown"),
                            ("Route", "Same institution" if same_bank else "Inter-bank"),
                        ]
                        context_cols = st.columns(4)
                        for col, (label, value) in zip(context_cols, context_items):
                            with col:
                                st.markdown(
                                    f"<div style='line-height:1.4;'>"
                                    f"<span style='color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.03em;'>{label}</span><br>"
                                    f"<span style='color:#e2e8f0; font-size:1.05rem; font-weight:600; word-break:break-word;'>{value}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        fig = draw_transaction_network(s_acc, r_acc, s_bank, r_bank, amt, currency=currency)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Hover over any bar or band for full account, institution, and amount details.")
                    else:
                        st.info("Transaction network visualization unavailable – missing data.")
                    
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
        
        to_process = merged_df[merged_df['row_index'].isin(batch_df['row_index'])]
        st.write(f"Successfully aligned {len(to_process)} records for pipeline execution.")
        
        if st.button("Start Batch Processing", type="primary"):
            # Simplified batch processing with Streamlit progress bar
            records = to_process.to_dict("records")
            total = len(records)
            progress_bar = st.progress(0)
            results = []
            for idx, record in enumerate(records, start=1):
                # Run summarization & verification for each record
                res = asyncio.run(summarize_and_verify(record, api_key=api_key))
                results.append(res)
                progress_bar.progress(idx / total)
            st.success(f"Batch processing complete for {total} records.")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df[["str_id", "suspicion_type", "faithfulness_score", "word_count"]])
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="str_lens_batch_results.csv",
                mime="text/csv"
            )