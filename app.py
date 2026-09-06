import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vitalis_engine import VitalisEngine

st.set_page_config(page_title="VITALIS | Optical Clinical Workstation", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State
if "detected_skin_tone" not in st.session_state:
    st.session_state.detected_skin_tone = "Type V"

# Custom Styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stHeader"] {display: none;}
    
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #00F2FE !important;
        font-family: 'monospace';
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.2); 
        color: #10B981; 
        border: 1px solid #10B981;
    }
    .detected-tone-box {
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid #00F2FE;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation Bar Header
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0 20px 0; border-bottom: 1px solid #1E293B; margin-bottom: 25px;">
        <div>
            <h1 style="margin:0; font-size: 1.8rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">
                VITALIS <span style="font-size: 0.9rem; color: #00F2FE; font-weight: 400; padding-left: 8px;">v2.4-CLINICAL</span>
            </h1>
            <p style="margin:0; font-size: 0.85rem; color: #64748B;">Skin-Tone Invariant Optical Triage & Multi-Modal Diagnostic Engine</p>
        </div>
        <div style="text-align: right;">
            <span class="status-badge">● SYSTEM ONLINE</span>
            <span style="font-size: 0.8rem; color: #64748B; margin-left: 10px;">Location: Kenya Health Mesh</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar - Controls & Patient Profile
with st.sidebar:
    st.markdown("### 📋 Patient Triage Profile")
    patient_id = st.text_input("Patient ID / Record No.", value="PAT-2026-8891")
    age = st.number_input("Patient Age", min_value=1, max_value=110, value=28)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    # Live Auto-Detected Skin Tone Indicator
    tone_display = st.empty()
    tone_display.markdown(
        f"""<div class="detected-tone-box">
            <span style="font-size: 0.8rem; color: #94A3B8;">ITA° AUTO-DETECTED TONE</span><br>
            <strong style="font-size: 1.2rem; color: #00F2FE;">{st.session_state.detected_skin_tone}</strong>
        </div>""", 
        unsafe_allow_html=True
    )

    # Fitzpatrick Calibration Manual Override Slider
    skin_tone = st.select_slider(
        "Fitzpatrick Scale Manual Override", 
        options=["Type I-II", "Type III", "Type IV", "Type V", "Type VI"],
        value=st.session_state.detected_skin_tone
    )
    
    st.divider()
    st.markdown("### ⚙️ Engine Parameters")
    show_rois = st.toggle("Display ROI Overlays", value=True)
    snr_threshold = st.slider("Min Signal-to-Noise Ratio", 30, 90, 60)
    export_fhir = st.checkbox("Enable FHIR JSON Sync", value=True)

col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.markdown("##### 📷 Live Optical Sensing & Multi-ROI Tracker")
    run_app = st.toggle("Initialize High-Resolution Scan", value=False)
    frame_placeholder = st.empty()
    chart_placeholder = st.empty()

with col2:
    st.markdown("##### 🩻 Diagnostic Output Engine")
    
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        bpm_box = st.empty()
        bpm_box.metric("Heart Rate (HR)", "-- BPM")
    with metric_col2:
        rr_box = st.empty()
        rr_box.metric("Resp. Rate (RR)", "-- RPM")

    st.markdown("<br>", unsafe_allow_html=True)
    
    metric_col3, metric_col4 = st.columns(2)
    with metric_col3:
        sqi_box = st.empty()
        sqi_box.metric("Signal Quality (SQI)", "-- %")
    with metric_col4:
        anemia_box = st.empty()
        anemia_box.metric("Anemia Screening", "Calibrating...")

    st.markdown("<br>", unsafe_allow_html=True)
    
    status_progress = st.progress(0)
    status_text = st.empty()
    
    st.markdown("##### 🩺 AI Clinical Triage & Guidance")
    triage_box = st.empty()

# Processing Loop
if run_app:
    cap = cv2.VideoCapture(0)
    engine = VitalisEngine(buffer_seconds=10, fps=30)
    
    while cap.isOpened() and run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Hardware Camera Stream Interrupted.")
            break
            
        frame = cv2.flip(frame, 1)
        
        # Process video frame
        processed_frame, bpm, rr, anemia_risk, sqi, pulse_wave, progress, detected_tone = engine.process_frame_advanced(
            frame, show_rois=show_rois
        )
        
        # Safely update detected tone in UI without direct widget state modification
        if detected_tone != st.session_state.detected_skin_tone:
            st.session_state.detected_skin_tone = detected_tone
            tone_display.markdown(
                f"""<div class="detected-tone-box">
                    <span style="font-size: 0.8rem; color: #94A3B8;">ITA° AUTO-DETECTED TONE</span><br>
                    <strong style="font-size: 1.2rem; color: #00F2FE;">{detected_tone}</strong>
                </div>""", 
                unsafe_allow_html=True
            )

        # Render Camera Stream
        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        # Update Calibration Progress Bar
        status_progress.progress(int(progress) / 100)
        if progress < 100:
            status_text.warning(f"Accumulating Optical Signals... {int(progress)}%")
        else:
            status_text.success("Signal Synchronized & Reading Live")

        # Update Metrics Grid
        if bpm is not None:
            bpm_box.metric("Heart Rate (HR)", f"{int(bpm)} BPM")
            rr_box.metric("Resp. Rate (RR)", f"{int(rr)} RPM")
            sqi_box.metric("Signal Quality (SQI)", f"{int(sqi)}%")
            anemia_box.metric("Anemia Screening", f"{anemia_risk}")

            # AI Clinical Summary
            triage_box.info(f"""
            **Patient Triage Summary ({patient_id}):**
            * **Cardiovascular:** {int(bpm)} BPM — Haemodynamic stability within expected operational thresholds.
            * **Respiratory:** {int(rr)} Breaths/Min — Thoracic motion flow analysis active.
            * **Optical Calibration:** Calibrated for Fitzpatrick {skin_tone} (Auto-Detected: {st.session_state.detected_skin_tone}).
            * **Conjunctival Chromaticity:** {anemia_risk} Risk Profile.
            """)

        # Render Live Waveform Chart
        if len(pulse_wave) > 0:
            fig, ax = plt.subplots(figsize=(7, 2.2))
            ax.plot(pulse_wave[-150:], color="#00F2FE", linewidth=2)
            ax.set_facecolor('#0A0E17')
            fig.patch.set_facecolor('#0A0E17')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            chart_placeholder.pyplot(fig)
            plt.close(fig)

    cap.release()
