"""
=============================================================================
SCIENTIFIC BIBLIOMETRIC AI ANALYZER (ENTERPRISE ULTIMATE EDITION)
=============================================================================
Sistem Perangkat Lunak Skala Penuh untuk Akuisisi, Pembersihan, Analisis, 
dan Pemetaan Sains (Science Mapping) Berbasis Data Bibliometrik.

Versi Enterprise ini dilengkapi dengan:
- Natural Language Processing (NLP)
- Machine Learning Topic Modeling (LDA)
- Semantic Information Retrieval (TF-IDF Cosine Similarity) dengan Multi-Template
- Geo-spatial Choropleth Mapping (Scopus & WIPO Patents Hybrid)
- Advanced Graph Topology (NetworkX & Pyvis)
- Generative AI Integration (Mistral, Gemini, Groq)
- Local Storage Persistence (Penyimpanan API & Konfigurasi)
=============================================================================
"""

import os
import streamlit as st
import streamlit.components.v1 as components
import PyPDF2
import pandas as pd
import requests
import re
import unicodedata
import zlib
import json
import math
import logging
import datetime
import tempfile # Modul untuk file HTML sementara yang aman bagi Cloud
import gc  # Modul Garbage Collector untuk optimalisasi RAM
from collections import Counter, defaultdict
import io
import xml.etree.ElementTree as ET # Modul GEXF Export

# Konfigurasi Logging Dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# LOCAL STORAGE (PERSISTENCE) ENGINE
# ==============================
SETTINGS_FILE = "biblio_settings.json"

def load_settings():
    """Membaca pengaturan/API key yang tersimpan di file lokal."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Gagal membaca settings: {e}")
            return {}
    return {}

def save_settings(settings_dict):
    """Menyimpan pengaturan/API key ke file lokal."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4)
    except Exception as e:
        logging.error(f"Gagal menyimpan settings: {e}")

# ==============================
# DEPENDENSI & EKSTENSI LANJUTAN
# ==============================
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    logging.warning("Modul WordCloud tidak terdeteksi.")

try:
    from matplotlib_venn import venn2
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    logging.warning("Modul matplotlib-venn tidak terdeteksi. Diagram irisan topik dinonaktifkan.")

try:
    import networkx as nx
    from networkx.algorithms import community as nx_comm
    import itertools
    import numpy as np
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("Modul NetworkX / Numpy tidak terdeteksi.")
    
try:
    import community as community_louvain # python-louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    logging.warning("Modul python-louvain tidak terdeteksi.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.colors as pc
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logging.warning("Modul Plotly tidak terdeteksi.")

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    logging.warning("Modul Pyvis tidak terdeteksi.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("Modul Scikit-Learn tidak terdeteksi. Fitur LDA & TF-IDF dinonaktifkan.")

# ==============================
# CONFIG & STYLING UTAMA
# ==============================
st.set_page_config(
    page_title="Bibliometrik AI Analyzer Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button { 
        border-radius: 6px; 
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 5px solid #4a90e2;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[role="radiogroup"] > label {
        padding: 12px 15px;
        border-radius: 8px;
        background-color: transparent;
        margin-bottom: 5px;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    div[role="radiogroup"] > label:hover {
        background-color: rgba(74, 144, 226, 0.1);
        border: 1px solid rgba(74, 144, 226, 0.2);
    }
    .streamlit-expanderHeader {
        font-weight: bold;
        font-size: 1.1em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f6;
        border-radius: 5px 5px 0 0;
        padding-left: 15px;
        padding-right: 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a90e2;
        color: white;
    }
    /* Tab tidak aktif */
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #f1f3f6;
    border-radius: 5px 5px 0 0;
    padding-left: 15px;
    padding-right: 15px;
    color: #000000 !important; /* <<< ini biar hitam */
}

/* Tab aktif */
.stTabs [aria-selected="true"] {
    background-color: #4a90e2;
    color: white !important;
}

/* Optional: hover biar lebih jelas */
.stTabs [data-baseweb="tab"]:hover {
    color: #000000;
}
    </style>
    """, unsafe_allow_html=True)

PLOTLY_DL_CONFIG = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png', 
        'filename': 'bibliometric_plot_HD',
        'height': 900,
        'width': 1200,
        'scale': 2
    }
}

# ==============================
# KAMUS DATA & KONSTANTA MASIF
# ==============================
COMMON_STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with", 
    "as", "by", "an", "this", "from", "are", "we", "at", "be", "it", 
    "tidak", "tersedia", "yang", "dan", "di", "dari", "ke", "untuk", 
    "tidak tersedia", "n/a", "or", "was", "were", "which", "study", "research",
    "using", "used", "can", "also", "has", "have", "been", "an", "their", "our",
    "results", "show", "showed", "analysis", "paper", "article", "approach", "based",
    "between", "more", "than", "two", "three", "four", "five", "may", "such", "through",
    "not", "but", "what", "where", "when", "who", "why", "how", "all", "any", "both"
}

# Mapping 2 Huruf WIPO ke 3 Huruf ISO (Agar bisa digambar di peta Plotly)
WIPO_2_LETTER_TO_ISO3 = {
    "AE": "ARE", "AG": "ATG", "AL": "ALB", "AM": "ARM", "AO": "AGO", "AT": "AUT", "AU": "AUS", 
    "AZ": "AZE", "BA": "BIH", "BB": "BRB", "BE": "BEL", "BF": "BFA", "BG": "BGR", "BH": "BHR", 
    "BJ": "BEN", "BN": "BRN", "BR": "BRA", "BW": "BWA", "BY": "BLR", "BZ": "BLZ", "CA": "CAN", 
    "CF": "CAF", "CG": "COG", "CH": "CHE", "CI": "CIV", "CL": "CHL", "CM": "CMR", "CN": "CHN", 
    "CO": "COL", "CR": "CRI", "CU": "CUB", "CV": "CPV", "CY": "CYP", "CZ": "CZE", "DE": "DEU", 
    "DJ": "DJI", "DK": "DNK", "DM": "DMA", "DO": "DOM", "DZ": "DZA", "EC": "ECU", "EE": "EST", 
    "EG": "EGY", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GA": "GAB", "GB": "GBR", "GD": "GRD", 
    "GE": "GEO", "GH": "GHA", "GM": "GMB", "GN": "GIN", "GQ": "GNQ", "GR": "GRC", "GT": "GTM", 
    "GW": "GNB", "HN": "HND", "HR": "HRV", "HU": "HUN", "ID": "IDN", "IE": "IRL", "IL": "ISR", 
    "IN": "IND", "IQ": "IRQ", "IR": "IRN", "IS": "ISL", "IT": "ITA", "JM": "JAM", "JO": "JOR", 
    "JP": "JPN", "KE": "KEN", "KG": "KGZ", "KH": "KHM", "KM": "COM", "KN": "KNA", "KP": "PRK", 
    "KR": "KOR", "KW": "KWT", "KZ": "KAZ", "LA": "LAO", "LC": "LCA", "LI": "LIE", "LK": "LKA", 
    "LR": "LBR", "LS": "LSO", "LT": "LTU", "LU": "LUX", "LV": "LVA", "LY": "LBY", "MA": "MAR", 
    "MC": "MCO", "MD": "MDA", "ME": "MNE", "MG": "MDG", "MK": "MKD", "ML": "MLI", "MN": "MNG", 
    "MR": "MRT", "MT": "MLT", "MU": "MUS", "MW": "MWI", "MX": "MEX", "MY": "MYS", "MZ": "MOZ", 
    "NA": "NAM", "NE": "NER", "NG": "NGA", "NI": "NIC", "NL": "NLD", "NO": "NOR", "NZ": "NZL", 
    "OM": "OMN", "PA": "PAN", "PE": "PER", "PG": "PNG", "PH": "PHL", "PL": "POL", "PT": "PRT", 
    "QA": "QAT", "RO": "ROU", "RS": "SRB", "RU": "RUS", "RW": "RWA", "SA": "SAU", "SC": "SYC", 
    "SD": "SDN", "SE": "SWE", "SG": "SGP", "SI": "SVN", "SK": "SVK", "SL": "SLE", "SM": "SMR", 
    "SN": "SEN", "ST": "STP", "SV": "SLV", "SY": "SYR", "SZ": "SWZ", "TD": "TCD", "TG": "TGO", 
    "TH": "THA", "TJ": "TJK", "TM": "TKM", "TN": "TUN", "TR": "TUR", "TT": "TTO", "TZ": "TZA", 
    "UA": "UKR", "UG": "UGA", "US": "USA", "UY": "URY", "UZ": "UZB", "VC": "VCT", "VN": "VNM", 
    "WS": "WSM", "ZA": "ZAF", "ZM": "ZMB", "ZW": "ZWE",
    "EP": "EPO", "WO": "WIPO", "IB": "WIPO", "EA": "EAPO", "AP": "ARIPO", "OA": "OAPI", "XN": "XNPI", "XV": "XVPI"
}

# Translasi dari ISO3 ke Nama Negara Full untuk Tooltip
ISO3_TO_NAME = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria", "AND": "Andorra", "AGO": "Angola", 
    "ARG": "Argentina", "ARM": "Armenia", "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan", 
    "BHR": "Bahrain", "BGD": "Bangladesh", "BEL": "Belgium", "BRA": "Brazil", "BGR": "Bulgaria", 
    "KHM": "Cambodia", "CMR": "Cameroon", "CAN": "Canada", "CHL": "Chile", "CHN": "China", 
    "COL": "Colombia", "CRI": "Costa Rica", "HRV": "Croatia", "CUB": "Cuba", "CYP": "Cyprus", 
    "CZE": "Czechia", "DNK": "Denmark", "ECU": "Ecuador", "EGY": "Egypt", "EST": "Estonia", 
    "ETH": "Ethiopia", "FJI": "Fiji", "FIN": "Finland", "FRA": "France", "GEO": "Georgia", 
    "DEU": "Germany", "GHA": "Ghana", "GRC": "Greece", "HUN": "Hungary", "ISL": "Iceland", 
    "IND": "India", "IDN": "Indonesia", "IRN": "Iran", "IRQ": "Iraq", "IRL": "Ireland", 
    "ISR": "Israel", "ITA": "Italy", "JAM": "Jamaica", "JPN": "Japan", "JOR": "Jordan", 
    "KAZ": "Kazakhstan", "KEN": "Kenya", "KWT": "Kuwait", "LBN": "Lebanon", "LTU": "Lithuania", 
    "LUX": "Luxembourg", "MDG": "Madagascar", "MYS": "Malaysia", "MEX": "Mexico", "MAR": "Morocco", 
    "NPL": "Nepal", "NLD": "Netherlands", "NZL": "New Zealand", "NGA": "Nigeria", "NOR": "Norway", 
    "OMN": "Oman", "PAK": "Pakistan", "PSE": "Palestine", "PER": "Peru", "PHL": "Philippines", 
    "POL": "Poland", "PRT": "Portugal", "QAT": "Qatar", "ROU": "Romania", "RUS": "Russian Federation", 
    "SAU": "Saudi Arabia", "SEN": "Senegal", "SRB": "Serbia", "SGP": "Singapore", "SVK": "Slovakia", 
    "SVN": "Slovenia", "ZAF": "South Africa", "KOR": "Republic of Korea", "ESP": "Spain", "LKA": "Sri Lanka", 
    "SDN": "Sudan", "SWE": "Sweden", "CHE": "Switzerland", "TWN": "Taiwan", "TZA": "United Republic of Tanzania", 
    "THA": "Thailand", "TUN": "Tunisia", "TUR": "Türkiye", "UGA": "Uganda", "UKR": "Ukraine", 
    "ARE": "United Arab Emirates", "GBR": "United Kingdom", "USA": "United States of America", "URY": "Uruguay", 
    "UZB": "Uzbekistan", "VEN": "Venezuela", "VNM": "Viet Nam", "ZMB": "Zambia", "ZWE": "Zimbabwe",
    "EPO": "European Patent Organisation", "WIPO": "World Intellectual Property Organization",
    "EAPO": "Eurasian Patent Organization", "ARIPO": "African Regional Intellectual Property Organization",
    "OAPI": "African Intellectual Property Organization", "ATG": "Antigua and Barbuda", 
    "BIH": "Bosnia & Herzegovina", "BRB": "Barbados", "BFA": "Burkina Faso", "BEN": "Benin", 
    "BRN": "Brunei Darussalam", "BWA": "Botswana", "BLR": "Belarus", "BLZ": "Belize",
    "CAF": "Central African Republic", "COG": "Congo", "CIV": "Côte d'Ivoire", "CPV": "Cabo Verde",
    "DJI": "Djibouti", "DMA": "Dominica", "DOM": "Dominican Republic", "GAB": "Gabon", "GRD": "Grenada",
    "GMB": "Gambia", "GIN": "Guinea", "GNQ": "Equatorial Guinea", "GTM": "Guatemala", "GNB": "Guinea-Bissau",
    "HND": "Honduras", "COM": "Comoros", "KNA": "Saint Kitts and Nevis", "PRK": "Democratic People's Republic of Korea", 
    "LAO": "Lao People's Democratic Republic", "LCA": "Saint Lucia", "LIE": "Liechtenstein", "LBR": "Liberia", 
    "LSO": "Lesotho", "LVA": "Latvia", "LBY": "Libya", "MCO": "Monaco", "MDA": "Republic of Moldova", 
    "MNE": "Montenegro", "MKD": "North Macedonia", "MLI": "Mali", "MNG": "Mongolia", "MRT": "Mauritania", 
    "MLT": "Malta", "MUS": "Mauritius", "MWI": "Malawi", "MOZ": "Mozambique", "NAM": "Namibia", 
    "NER": "Niger", "NIC": "Nicaragua", "PAN": "Panama", "PNG": "Papua New Guinea", "RWA": "Rwanda", 
    "SYC": "Seychelles", "SLE": "Sierra Leone", "SMR": "San Marino", "STP": "Sao Tome and Principe", 
    "SLV": "El Salvador", "SYR": "Syrian Arab Republic", "SWZ": "Eswatini", "TCD": "Chad", "TGO": "Togo", 
    "TJK": "Tajikistan", "TKM": "Turkmenistan", "TTO": "Trinidad and Tobago", "VCT": "Saint Vincent and the Grenadines", 
    "WSM": "Samoa", "XNPI": "Nordic Patent Institute", "XVPI": "Visegrad Patent Institute"
}

# Reverse mapping untuk deteksi teks panjang (Scopus/WoS Fallback)
COUNTRY_ISO_MAPPING = {name: iso for iso, name in ISO3_TO_NAME.items()}
COUNTRY_ISO_MAPPING.update({
    "UK": "GBR", "England": "GBR", "USA": "USA", "United States": "USA",
    "Russia": "RUS", "South Korea": "KOR", "Vietnam": "VNM", "Turkey": "TUR",
    "Tanzania": "TZA", "UAE": "ARE"
})

BIBLIOMETRIC_GLOSSARY = {
    "Bradford's Law": "Hukum Bradford mengidentifikasi Jurnal Inti (Core Journals).",
    "Lotka's Law": "Hukum Lotka mendeskripsikan frekuensi publikasi oleh penulis.",
    "Centrality (Callon)": "Mengukur kekuatan koneksi suatu klaster dengan klaster-klaster lainnya (Relevance degree).",
    "Density (Callon)": "Mengukur kekuatan hubungan internal antar kata kunci di dalam suatu klaster (Development degree)."
}

# ==============================
# FUNGSI LLM & GENERATIVE AI
# ==============================
def stream_mistral(system_prompt: str, user_prompt: str, api_key: str, model: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.2, "stream": True}
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, stream=True)
        if response.status_code != 200: yield f"❌ Error Mistral ({response.status_code}): {response.text}"; return
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    if data_str.strip() == '[DONE]': break
                    try:
                        content = json.loads(data_str)['choices'][0]['delta'].get('content', '')
                        if content: yield content 
                    except json.JSONDecodeError: pass
    except Exception as e: yield f"❌ Terjadi kesalahan internal: {e}"

def stream_gemini(system_prompt: str, user_prompt: str, api_key: str, model: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        if response.status_code != 200: yield f"❌ Error Gemini ({response.status_code}): {response.text}"; return
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        data_json = json.loads(data_str)
                        content = data_json['candidates'][0]['content']['parts'][0].get('text', '')
                        if content: yield content
                    except: pass
    except Exception as e: yield f"❌ Terjadi kesalahan internal: {e}"

def stream_groq(system_prompt: str, user_prompt: str, api_key: str, model: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.2, "stream": True}
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, stream=True)
        if response.status_code != 200: yield f"❌ Error Groq ({response.status_code}): {response.text}"; return
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    if data_str.strip() == '[DONE]': break
                    try:
                        content = json.loads(data_str)['choices'][0]['delta'].get('content', '')
                        if content: yield content 
                    except json.JSONDecodeError: pass
    except Exception as e: yield f"❌ Terjadi kesalahan internal: {e}"

@st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
def call_mistral_sync(system_prompt: str, user_prompt: str, api_key: str, model: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=None)
        if response.status_code == 200: return response.json()["choices"][0]["message"]["content"]
        else: return f"Error: {response.text}"
    except Exception as e: return f"Error: {e}"

@st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
def call_gemini_sync(system_prompt: str, user_prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=None)
        if response.status_code == 200: return response.json()['candidates'][0]['content']['parts'][0]['text']
        else: return f"Error: {response.text}"
    except Exception as e: return f"Error: {e}"

@st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
def call_groq_sync(system_prompt: str, user_prompt: str, api_key: str, model: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=None)
        if response.status_code == 200: return response.json()["choices"][0]["message"]["content"]
        else: return f"Error: {response.text}"
    except Exception as e: return f"Error: {e}"

# ==============================
# DATA WRANGLING & EXTRACTORS
# ==============================
@st.cache_data(show_spinner=False)
def load_ipc_excel_database(file_path):
    """
    Membaca database IPC dari file Excel lokal dan mengubahnya menjadi Dictionary.
    Sangat cepat dan akurat 100%.
    """
    ipc_dict = {}
    try:
        # Membaca file Excel
        df = pd.read_excel(file_path)
        
        # Deteksi otomatis nama kolom (asumsi kolom pertama adalah Kode, kolom kedua adalah Deskripsi)
        if len(df.columns) >= 2:
            kode_col = df.columns[0]
            desc_col = df.columns[1]
            
            # Konversi dataframe ke dictionary
            # Gunakan str() untuk memastikan format teks, dan isi NaN dengan string kosong
            for index, row in df.iterrows():
                kode = str(row[kode_col]).strip().upper()
                desc = str(row[desc_col]).strip()
                if kode and kode != "NAN":
                    ipc_dict[kode] = desc
            return ipc_dict, f"✅ Sukses memuat **{len(ipc_dict)}** kode IPC dari database Excel!"
        else:
            return {}, "❌ File Excel harus memiliki minimal 2 kolom (Kode dan Deskripsi)."
            
    except Exception as e:
        return {}, f"❌ Gagal memuat database Excel: {e}"

def build_ipc_dict_with_ai(folder_path, selected_files, api_key, model, provider):
    """
    Ekstraksi IPC menggunakan AI hanya pada file PDF yang dipilih pengguna.
    """
    master_ipc_dict = {}
    
    if not selected_files:
        return master_ipc_dict, "⚠️ Tidak ada file yang dipilih untuk diproses."

    sys_prompt = """Anda adalah asisten ahli klasifikasi paten WIPO (IPC).
Tugas Anda adalah mengekstrak SETIAP kode paten dan deskripsinya dari teks kotor yang diberikan.
ATURAN MUTLAK:
1. Pahami hierarki (Misal 'A01B' adalah kelas, '1/00' adalah grup).
2. Output WAJIB murni format JSON tanpa kata-kata lain.
3. Format output: {"KODE_LENGKAP": "Deskripsi Lengkap"}"""

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file_name in enumerate(selected_files):
        file_path = os.path.join(folder_path, file_name)
        status_text.markdown(f"**⏳ AI sedang membaca:** `{file_name}`")
        
        try:
            reader = PyPDF2.PdfReader(file_path)
            # Membaca 5 halaman pertama untuk akurasi tinggi
            for page in reader.pages[:5]:
                raw_text = page.extract_text()
                if not raw_text: continue
                
                user_prompt = f"Ekstrak teks ini ke JSON:\n\n{raw_text[:3000]}"
                
                # Memanggil sinkronisasi API berdasarkan provider yang aktif [cite: 2642]
                if provider == "Mistral":
                    ai_response = call_mistral_sync(sys_prompt, user_prompt, api_key, model)
                elif provider == "Google Gemini":
                    ai_response = call_gemini_sync(sys_prompt, user_prompt, api_key, model)
                elif provider == "Groq":
                    ai_response = call_groq_sync(sys_prompt, user_prompt, api_key, model)
                
                try:
                    # Parsing hasil JSON dari AI
                    match = re.search(r'\{[\s\S]*\}', ai_response)
                    json_str = match.group(0) if match else ai_response
                    master_ipc_dict.update(json.loads(json_str))
                except:
                    continue
                    
        except Exception as e:
            st.error(f"Gagal memproses {file_name}: {e}")
            
        progress_bar.progress((idx + 1) / len(selected_files))

    status_text.empty()
    progress_bar.empty()
    return master_ipc_dict, f"✅ Sukses mengekstrak {len(master_ipc_dict)} kode dari {len(selected_files)} file."

def clean_scopus_data(raw_data: dict) -> pd.DataFrame:
    cleaned_list = []
    entries = raw_data.get('search-results', {}).get('entry', [])
    for entry in entries:
        affiliations = entry.get('affiliation', [])
        aff_countries = []
        if isinstance(affiliations, list):
            for aff in affiliations:
                country = aff.get('affiliation-country', '')
                if country: aff_countries.append(country)
        elif isinstance(affiliations, dict):
            country = affiliations.get('affiliation-country', '')
            if country: aff_countries.append(country)
            
        cleaned_list.append({
            "Judul": entry.get('dc:title', 'Tidak tersedia'),
            "Abstract": entry.get('dc:description', 'Tidak tersedia'),
            "Penulis": entry.get('dc:creator', 'Tidak tersedia'),
            "Jurnal": entry.get('prism:publicationName', 'Tidak tersedia'),
            "Tahun": str(entry.get('prism:coverDate', 'N/A'))[:4],
            "DOI": entry.get('prism:doi', 'Tidak tersedia'),
            "Negara Afiliasi": "; ".join(list(set(aff_countries))) if aff_countries else 'Tidak tersedia',
            "Citasi": entry.get('citedby-count', '0')
        })
    return pd.DataFrame(cleaned_list)

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def extract_countries_from_text(text: str) -> list:
    """Ekstraksi Hibrida untuk mendeteksi 2-Letter WIPO Codes atau Teks Nama Negara (Scopus)"""
    text_str = str(text).strip()
    detected = []
    
    # 1. Cek format kode WIPO 2 huruf (biasanya dipisahkan koma, spasi, atau titik koma)
    parts = [p.strip().upper() for p in re.split(r'[,;|/\s]+', text_str) if p.strip()]
    is_wipo_code_format = all(len(p) == 2 for p in parts) if parts else False
    
    if is_wipo_code_format:
        for part in parts:
            if part in WIPO_2_LETTER_TO_ISO3:
                iso3 = WIPO_2_LETTER_TO_ISO3[part]
                country_name = ISO3_TO_NAME.get(iso3, part)
                detected.append(country_name)
                
    # 2. Jika tidak ada yang cocok sebagai 2 huruf, cek full text untuk nama negara (Scopus/WoS fallback)
    if not detected:
        text_lower = text_str.lower()
        for country in COUNTRY_ISO_MAPPING.keys():
            if re.search(rf'\b{re.escape(country.lower())}\b', text_lower):
                detected.append(country)
                
    return list(set(detected))

def get_top_words(text_series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    all_text = " ".join(text_series.dropna().astype(str)).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    filtered_words = [word for word in words if word not in COMMON_STOPWORDS]
    word_counts = Counter(filtered_words)
    return pd.DataFrame(word_counts.most_common(top_n), columns=['Kata', 'Frekuensi']).set_index('Kata')

def calculate_bradford_law(df: pd.DataFrame, journal_col: str) -> pd.DataFrame:
    if df.empty or journal_col not in df.columns: return pd.DataFrame()
    j_counts = df[journal_col].value_counts().reset_index()
    j_counts.columns = ['Jurnal', 'Freq']
    j_counts = j_counts.sort_values(by='Freq', ascending=False)
    
    total_articles = j_counts['Freq'].sum()
    j_counts['CumFreq'] = j_counts['Freq'].cumsum()
    j_counts['CumPerc'] = j_counts['CumFreq'] / total_articles
    
    def assign_zone(perc):
        if perc <= 0.333: return "Zone 1 (Core)"
        elif perc <= 0.666: return "Zone 2 (Related)"
        else: return "Zone 3 (Peripheral)"
        
    j_counts['Bradford Zone'] = j_counts['CumPerc'].apply(assign_zone)
    return j_counts

def calculate_lotkas_law(df: pd.DataFrame, author_col: str) -> pd.DataFrame:
    if df.empty or author_col not in df.columns: return pd.DataFrame()
    all_authors = df[author_col].dropna().astype(str).str.split(";").explode().str.strip()
    all_authors = all_authors[all_authors != ""]
    
    author_counts = all_authors.value_counts().values
    doc_freq = Counter(author_counts) 
    
    lotka_data = []
    total_authors = len(all_authors.unique())
    
    for docs_written, num_authors in sorted(doc_freq.items()):
        lotka_data.append({
            "Jumlah Dokumen Ditulis": str(docs_written),
            "Jumlah Penulis Aktual": num_authors,
            "Persentase Aktual (%)": round((num_authors / total_authors) * 100, 2),
            "Prediksi Lotka Teoritis (%)": round((1 / (docs_written**2)) * (doc_freq[1]/total_authors) * 100, 2) if doc_freq[1] else 0
        })
    return pd.DataFrame(lotka_data)

# ===============================================
# TRUE BIBLIOMETRIX ENGINE (EXACT MATCH FORMULAS)
# ===============================================

def preprocess_keywords(df, field="ID", delimiter=";", is_author=False):
    df_clean = df.copy()
    def process(x):
        if pd.isna(x): return tuple() 
        words = [k.strip().title() if is_author else k.strip().lower() for k in str(x).split(delimiter)]
        return tuple([w for w in words if w and w.lower() not in COMMON_STOPWORDS and w.lower() not in ["tidak tersedia", "n/a", "no title"]])
    df_clean[field] = df_clean[field].apply(process)
    return df_clean

def build_cooccurrence(df, field="ID", minfreq=5):
    word_counts = Counter()
    for kws in df[field]:
        word_counts.update(kws)

    valid_words = {w for w, c in word_counts.items() if c >= minfreq}

    G = nx.Graph()
    for kws in df[field]:
        kws = [k for k in kws if k in valid_words]
        for w in kws:
            if not G.has_node(w):
                G.add_node(w, freq=word_counts[w], size=word_counts[w]) 
        
        for w1, w2 in itertools.combinations(set(kws), 2):
            if G.has_edge(w1, w2):
                G[w1][w2]['weight'] += 1
            else:
                G.add_edge(w1, w2, weight=1)
    return G, word_counts

def normalize_network(G, method="Equivalence"):
    G_norm = nx.Graph()
    for node, data in G.nodes(data=True):
        G_norm.add_node(node, **data)

    for u, v, data in G.edges(data=True):
        c_ij = data['weight']
        c_i = G.nodes[u]['freq']
        c_j = G.nodes[v]['freq']
        
        if method == "Association":
            weight = c_ij / (c_i * c_j) if (c_i * c_j) > 0 else 0
        elif method == "Equivalence":
            weight = (c_ij ** 2) / (c_i * c_j) if (c_i * c_j) > 0 else 0
        elif method == "Salton":
            weight = c_ij / math.sqrt(c_i * c_j) if (c_i * c_j) > 0 else 0
        elif method == "Jaccard":
            weight = c_ij / (c_i + c_j - c_ij) if (c_i + c_j - c_ij) > 0 else 0
        elif method == "Inclusion":
            weight = c_ij / min(c_i, c_j) if min(c_i, c_j) > 0 else 0
        else: # Raw
            weight = c_ij

        if weight > 0:
            G_norm.add_edge(u, v, weight=weight, raw_weight=c_ij)
    return G_norm

def filter_edges(G, min_raw_weight=1, min_norm_weight=0.0):
    G_f = nx.Graph()
    for node, data in G.nodes(data=True):
        G_f.add_node(node, **data)

    for u, v, data in G.edges(data=True):
        raw_w = data.get('raw_weight', data.get('weight', 1))
        norm_w = data.get('weight', 0)
        
        if raw_w >= min_raw_weight and norm_w >= min_norm_weight:
            G_f.add_edge(u, v, **data)
    return G_f

def detect_clusters(G, method="Louvain"):
    if method == "Louvain" and HAS_LOUVAIN:
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        clusters = {}
        for node, cid in partition.items():
            clusters.setdefault(cid, []).append(node)
        comms = list(clusters.values())
    elif method == "Betweenness":
        try:
            comms = next(nx_comm.girvan_newman(G))
        except:
            comms = [list(c) for c in nx_comm.greedy_modularity_communities(G, weight='weight')]
    elif method in ["InfoMap", "Walktrap", "Spinglass", "Leiden", "Leading Eigenvector"]:
        if method == "InfoMap":
            comms = list(nx_comm.label_propagation_communities(G))
        else:
            comms = [list(c) for c in nx_comm.greedy_modularity_communities(G, weight='weight')]
    else:
        comms = [list(c) for c in nx_comm.greedy_modularity_communities(G, weight='weight')]
        
    comms = sorted([list(c) for c in comms], key=lambda c: (-len(c), sorted(c)[0]))
    return comms

def compute_callon_metrics(G, communities, word_counts):
    theme_data = []
    for i, comm in enumerate(communities):
        if len(comm) < 2: continue

        comm_sorted = sorted(comm, key=lambda x: word_counts.get(x, 0), reverse=True)
        cluster_name = comm_sorted[0].title()

        subgraph = G.subgraph(comm)
        internal_weight = subgraph.size(weight='weight')
        n = len(comm)
        density = (internal_weight * 100) / n if n > 0 else 0

        total_weight = sum(G.degree(u, weight='weight') for u in comm)
        external_weight = total_weight - (2 * internal_weight)
        centrality = external_weight * 10

        theme_data.append({
            "Nama Tema Kunci": cluster_name,
            "Centrality": round(centrality, 4),
            "Density": round(density, 4),
            "Volume Representasi": sum(word_counts.get(w, 0) for w in comm),
            "keywords": comm_sorted
        })

    return pd.DataFrame(theme_data)

# ==============================
# ALGORITMA NLP & OPENREFINE
# ==============================
def get_fingerprint(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn') 
    return " ".join(sorted(list(set(text.split()))))

def get_ngram_fingerprint(text: str, n: int = 2) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.replace(" ", "") 
    if len(text) < n: return text
    return "".join(sorted(list(set([text[i:i+n] for i in range(len(text)-n+1)]))))

def get_soundex(token: str) -> str:
    token = str(token).upper()
    token = re.sub(r'[^A-Z]', '', token)
    if not token: return ""
    soundex = token[0]
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    for char in token[1:]:
        for key, code in dictionary.items():
            if char in key and code != soundex[-1]: soundex += code
    return soundex.replace(".", "")[:4].ljust(4, "0")

def get_phonetic_fingerprint(text: str) -> str:
    return " ".join(sorted(list(set([get_soundex(t) for t in re.sub(r'[^\w\s]', '', str(text)).split()]))))

def levenshtein(s1: str, s2: str, max_dist: int) -> int:
    if abs(len(s1) - len(s2)) > max_dist: return max_dist + 1
    if len(s1) < len(s2): return levenshtein(s2, s1, max_dist)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions, deletions = previous_row[j + 1] + 1, current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        if min(previous_row) > max_dist: return max_dist + 1
    return previous_row[-1]

def ppm_distance(s1: str, s2: str) -> float:
    c1, c2 = len(zlib.compress(s1.encode('utf-8'))), len(zlib.compress(s2.encode('utf-8')))
    c12 = len(zlib.compress((s1 + ' ' + s2).encode('utf-8')))
    return (c12 - min(c1, c2)) / max(c1, c2)

# ==============================
# ALGORITMA GRAPH EXPORTER (GEXF)
# ==============================
def generate_gexf_string(G: nx.Graph) -> bytes:
    gexf = ET.Element('gexf', xmlns="http://www.gexf.net/1.2draft", version="1.2")
    graph = ET.SubElement(gexf, 'graph', mode="static", defaultedgetype="undirected")
    
    attributes = ET.SubElement(graph, 'attributes', {'class': 'node'})
    ET.SubElement(attributes, 'attribute', id="0", title="weight", type="integer")
    
    nodes_elem = ET.SubElement(graph, 'nodes')
    for node in G.nodes():
        node_elem = ET.SubElement(nodes_elem, 'node', id=node, label=node)
        attvalues = ET.SubElement(node_elem, 'attvalues')
        ET.SubElement(attvalues, 'attvalue', {'for': "0", 'value': str(G.nodes[node].get('freq', 1))})
        
    edges_elem = ET.SubElement(graph, 'edges')
    for i, (u, v) in enumerate(G.edges()):
        edge_weight = G[u][v].get('raw_weight', G[u][v].get('weight', 1))
        ET.SubElement(edges_elem, 'edge', id=str(i), source=u, target=v, weight=str(edge_weight))
        
    return ET.tostring(gexf, encoding='utf-8', xml_declaration=True)

# ==============================
# INISIALISASI SESSION STATE 
# ==============================
if 'history' not in st.session_state: st.session_state.history = []
if 'history_actions' not in st.session_state: st.session_state.history_actions = []
if 'current_step' not in st.session_state: st.session_state.current_step = -1
if 'clustering_result' not in st.session_state: st.session_state.clustering_result = None
if 'preview_action' not in st.session_state: st.session_state.preview_action = None
if 'preview_original' not in st.session_state: st.session_state.preview_original = None
if 'preview_new' not in st.session_state: st.session_state.preview_new = None
if 'chat_messages' not in st.session_state: st.session_state.chat_messages = []
if 'map_rendered' not in st.session_state: st.session_state.map_rendered = False

@st.cache_data
def get_column_mappings(columns_tuple):
    """Mendeteksi nama kolom spesifik dari berbagai sumber (Scopus, WoS, WIPO Patents)"""
    columns = list(columns_tuple)
    return {
        'year_col': next((col for col in ['Tahun', 'Year', 'year', 'PY', 'Publication Year', 'Publication Date', 'Application Date'] if col in columns), None),
        'journal_col': next((col for col in ['Jurnal', 'Source title', 'Journal', 'SO', 'IPC', 'I P C', 'Applicants'] if col in columns), None),
        'author_col': next((col for col in ['Penulis', 'Authors', 'Author', 'AU', 'Inventors'] if col in columns), None),
        'title_col': next((col for col in ['Judul', 'Title', 'title', 'Document Title', 'TI'] if col in columns), None),
        'citation_col': next((col for col in ['Citasi', 'Cited by', 'citedby-count', 'TC'] if col in columns), None),
        'abstract_col': next((col for col in ['Abstract', 'abstract', 'Description', 'AB', 'Abstrak'] if col in columns), None),
        'affiliation_col': next((col for col in ['Negara Afiliasi', 'Affiliation', 'Affiliations', 'C1', 'Country', 'Country Code'] if col in columns), None),
        'ipc_col': next((col for col in ['IPC', 'I P C'] if col in columns), None),
        # WIPO Specific Extensions (Optional, safe mapping fallback)
        'application_id_col': next((col for col in ['Application Id', 'Application ID'] if col in columns), None),
        'application_number_col': next((col for col in ['Application Number'] if col in columns), None),
        'publication_number_col': next((col for col in ['Publication Number'] if col in columns), None),
        'priority_data_col': next((col for col in ['Priority Data', 'Priorities Data'] if col in columns), None),
        'national_phase_entries_col': next((col for col in ['National Phase Entries'] if col in columns), None)
    }

def apply_transform(func, action_name, is_row_filter=False, target_col=None):
    base_df = st.session_state.history[st.session_state.current_step].copy()
    st.session_state.preview_original = base_df[target_col].copy() if target_col else None
    if is_row_filter:
        new_df = func(base_df)
        st.session_state.preview_new = new_df[target_col].copy() if target_col else None
    else:
        new_df = base_df.copy()
        new_df[target_col] = func(new_df[target_col])
        st.session_state.preview_new = new_df[target_col].copy()
    
    # Batasi history max 3 langkah untuk menghemat RAM (Garbage Collection Fix)
    MAX_HISTORY = 3
    st.session_state.history = st.session_state.history[max(0, st.session_state.current_step - MAX_HISTORY + 1):st.session_state.current_step + 1]
    st.session_state.history_actions = st.session_state.history_actions[max(0, st.session_state.current_step - MAX_HISTORY + 1):st.session_state.current_step + 1]
    st.session_state.current_step = len(st.session_state.history) - 1
    
    st.session_state.history.append(new_df)
    st.session_state.history_actions.append(action_name)
    st.session_state.current_step += 1
    st.session_state.preview_action = action_name
    gc.collect()
    st.rerun()

# ==============================
# STRUKTUR ANTARMUKA (SIDEBAR)
# ==============================
user_prefs = load_settings()

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #4a90e2; font-weight: 800; font-family: sans-serif;'>📚 Biblio Analyzer Pro</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:12px; margin-top:-15px;'>Enterprise Mapping Edition</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    menu_options = ["📥 Data Acquisition", "📖 Library & Glossary"]
    if len(st.session_state.history) > 0:
        menu_options = [
            "📥 Data Acquisition",
            "📊 Overview & Trends",
            "🧹 Data Cleaning",
            "🤖 AI Synthesis",
            "🕸️ Conceptual Structure",
            "🌍 Komparasi Global vs Indonesia",
            "💬 AI Chatbot (RAG)",
            "📖 Library & Glossary"
        ]
    
    menu_selection = st.radio("MAIN MENU NAVIGATION", menu_options, label_visibility="visible")
    
    st.markdown("---")
    with st.expander("⚙️ Settings & API Keys", expanded=False):
        SCOPUS_API_KEY = st.text_input("Scopus API Key", type="password", value=user_prefs.get("scopus_key", ""), placeholder="Masukkan key Scopus").strip()
        st.markdown("**AI Provider**")
        
        # PENGATURAN AI PROVIDER DIPERBARUI DENGAN GROQ
        providers = ["Mistral", "Google Gemini", "Groq"]
        default_prov = user_prefs.get("ai_provider", "Mistral")
        prov_idx = providers.index(default_prov) if default_prov in providers else 0
        AI_PROVIDER = st.selectbox("Pilih Penyedia:", providers, index=prov_idx, label_visibility="collapsed")
        
        if AI_PROVIDER == "Mistral":
            AI_API_KEY = st.text_input("Mistral API Key", type="password", value=user_prefs.get("mistral_key", ""), placeholder="Masukkan key Mistral").strip()
            mistral_models = ["mistral-small-latest", "open-mistral-nemo", "mistral-large-latest"]
            m_def = user_prefs.get("mistral_model", "mistral-small-latest")
            m_idx = mistral_models.index(m_def) if m_def in mistral_models else 0
            AI_MODEL = st.selectbox("Model:", mistral_models, index=m_idx)
            st.caption("✨ 'mistral-small' = Cepat & Hemat Token.")
            
        elif AI_PROVIDER == "Google Gemini":
            AI_API_KEY = st.text_input("Gemini API Key", type="password", value=user_prefs.get("gemini_key", ""), placeholder="Masukkan key Gemini").strip()
            gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
            g_def = user_prefs.get("gemini_model", "gemini-2.5-flash")
            g_idx = gemini_models.index(g_def) if g_def in gemini_models else 0
            AI_MODEL = st.selectbox("Model:", gemini_models, index=g_idx)
            st.caption("✨ 'gemini-2.5-flash' = Sangat kencang.")
            
        elif AI_PROVIDER == "Groq":
            AI_API_KEY = st.text_input("Groq API Key", type="password", value=user_prefs.get("groq_key", ""), placeholder="Masukkan key Groq").strip()
            groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]
            q_def = user_prefs.get("groq_model", "llama-3.1-8b-instant")
            q_idx = groq_models.index(q_def) if q_def in groq_models else 0
            AI_MODEL = st.selectbox("Model:", groq_models, index=q_idx)
            st.caption("✨ 'llama-3.1-8b' = Super Cepat, 'llama-3.3-70b' = Sangat Cerdas.")

        if st.button("💾 Simpan Pengaturan", use_container_width=True):
            new_settings = user_prefs.copy()
            new_settings["scopus_key"] = SCOPUS_API_KEY
            new_settings["ai_provider"] = AI_PROVIDER
            if AI_PROVIDER == "Mistral":
                new_settings["mistral_key"] = AI_API_KEY
                new_settings["mistral_model"] = AI_MODEL
            elif AI_PROVIDER == "Google Gemini":
                new_settings["gemini_key"] = AI_API_KEY
                new_settings["gemini_model"] = AI_MODEL
            elif AI_PROVIDER == "Groq":
                new_settings["groq_key"] = AI_API_KEY
                new_settings["groq_model"] = AI_MODEL
            save_settings(new_settings)
            st.toast("Pengaturan berhasil disimpan secara lokal!", icon="✅")
            
with st.sidebar:
     with st.expander("🧰 WIPO IPC Excel Database", expanded=False):
        # --- TAUTAN REFERENSI RESMI ---
        st.markdown("Akses cepat referensi paten internasional.")
        st.markdown("[🌐 **Buka Skema IPC WIPO (Tautan Eksternal)**](https://www.wipo.int/ipc/itos4ipc/ITSupport_and_download_area/20260101/pdf/scheme/full_ipc/en/index.html)")
        st.markdown("---")

        # --- BAGIAN 1: KAMUS KODE NEGARA WIPO ---
        st.markdown("**🌍 Kamus Kode Negara WIPO**")
        wipo_query = st.text_input("Ketik 2 Huruf Kode Negara:", max_chars=2, placeholder="Misal: US, EP, WO").strip().upper()
        if len(wipo_query) == 2:
            if wipo_query in WIPO_2_LETTER_TO_ISO3:
                iso_code = WIPO_2_LETTER_TO_ISO3[wipo_query]
                country_name = ISO3_TO_NAME.get(iso_code, iso_code)
                st.success(f"✅ **{country_name}**")
            else:
                st.error("❌ Kode Tidak Ditemukan")
                
        st.markdown("---")
        
        # --- BAGIAN 2: DATABASE EXCEL LOKAL ---
        st.markdown("**📖 Database Kamus IPC (Excel)**")
        st.caption("Baca database IPC langsung dari file Excel lokal untuk kecepatan dan akurasi 100%.")
        
        # Path folder tempat Anda menyimpan file Excel database
        db_folder = st.text_input("Path Folder Database Excel:", value="./database")
        
        if os.path.exists(db_folder):
            # Mencari file Excel di dalam folder
            all_excels = [f for f in os.listdir(db_folder) if f.lower().endswith(('.xlsx', '.xls'))]
            
            if all_excels:
                selected_excel = st.selectbox("Pilih File Database IPC:", options=all_excels)
                excel_path = os.path.join(db_folder, selected_excel)
                
                if st.button("🚀 Muat Database IPC", use_container_width=True):
                    with st.spinner("Memuat database ke dalam memori..."):
                        hasil_dict, pesan = load_ipc_excel_database(excel_path)
                        
                        if hasil_dict:
                            st.session_state.ipc_excel_dictionary = hasil_dict
                            st.success(pesan)
                        else:
                            st.error(pesan)
            else:
                st.warning(f"Tidak ada file Excel (.xlsx/.xls) di folder '{db_folder}'.")
        else:
            st.error(f"Folder '{db_folder}' tidak ditemukan.")

        # --- BAGIAN 3: PENCARIAN HASIL DATABASE ---
        if 'ipc_excel_dictionary' in st.session_state and st.session_state.ipc_excel_dictionary:
            st.markdown("---")
            search_ipc = st.text_input("🔍 Cari Kode IPC atau Deskripsi (Misal: A01B atau Soil):").strip().upper()
            
            if search_ipc:
                # Pencarian Fuzzy (mencari di Kode ATAU Deskripsi)
                query_clean = search_ipc.replace(" ", "").replace("/", "")
                match_results = {}
                
                for k, v in st.session_state.ipc_excel_dictionary.items():
                    k_clean = k.replace(" ", "").replace("/", "").upper()
                    v_upper = str(v).upper()
                    
                    if query_clean in k_clean or search_ipc in v_upper:
                        match_results[k] = v
                
                if match_results:
                    # Menampilkan SEMUA hasil tanpa batasan
                    for k, v in match_results.items():
                        st.info(f"**{k}**\n\n{v}")
                    st.caption(f"Menampilkan total {len(match_results)} hasil.")
                else:
                    st.error("Pencarian tidak ditemukan dalam database.")

# ==============================
# MENU UTAMA 0: GLOSSARY (EDUKASI)
# ==============================
if menu_selection == "📖 Library & Glossary":
    st.title("📖 Library & Glossary")
    st.markdown("Pusat dokumentasi dan ensiklopedia interaktif untuk memahami seluruh metodologi matematis dan bibliometrik yang digunakan di dalam perangkat lunak ini.")
    st.markdown("---")
    
    cols_dict1, cols_dict2 = st.columns(2)
    items = list(BIBLIOMETRIC_GLOSSARY.items())
    
    for i, (term, definition) in enumerate(items):
        if i % 2 == 0:
            with cols_dict1:
                st.info(f"**{term}**\n\n{definition}")
        else:
            with cols_dict2:
                st.success(f"**{term}**\n\n{definition}")

# ==============================
# MENU UTAMA 1: DATA ACQUISITION
# ==============================
elif menu_selection == "📥 Data Acquisition":
    st.title("📥 Data Acquisition")
    st.markdown("Mulai proyek bibliometrik Anda dengan mengimpor dataset langsung dari API server Scopus atau mengunggah data lokal milik Anda sendiri.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔍 Opsi 1: Tarik via Scopus API")
        st.info("Ketikkan Query untuk mencari jurnal secara *real-time* ke server Elsevier Scopus.")
        query = st.text_input("Query Scopus", placeholder="Contoh: TITLE-ABS-KEY(Artificial Intelligence)")
        max_results = st.number_input("Jumlah Dokumen Maksimal", min_value=10, max_value=500, value=50, step=10)
        
        if st.button("Cari Data via API", type="primary", use_container_width=True):
            if not SCOPUS_API_KEY: 
                st.error("⚠️ Masukkan Scopus API Key di menu Settings (Sidebar) terlebih dahulu.")
            elif not query: 
                st.warning("⚠️ Masukkan query pencarian.")
            else:
                with st.spinner("Mengunduh data dari Elsevier (Scopus)..."):
                    # Paginasi Otomatis (Fix limit 200)
                    all_entries = []
                    start_idx = 0
                    
                    while len(all_entries) < max_results:
                        fetch_count = min(200, max_results - len(all_entries))
                        url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={fetch_count}&start={start_idx}&view=COMPLETE&apiKey={SCOPUS_API_KEY}"
                        resp = requests.get(url, headers={'Accept': 'application/json'})
                        
                        if resp.status_code == 200:
                            data_chunk = resp.json().get('search-results', {}).get('entry', [])
                            if not data_chunk: break # Data sudah habis di server
                            all_entries.extend(data_chunk)
                            start_idx += fetch_count
                        elif resp.status_code in [401, 403]:
                            st.toast("⚠️ Akses institusi Scopus tidak terdeteksi. Menggunakan mode Fallback (Tanpa Abstrak)...", icon="🔄")
                            url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={max_results}&apiKey={SCOPUS_API_KEY}"
                            resp_fallback = requests.get(url, headers={'Accept': 'application/json'})
                            if resp_fallback.status_code == 200:
                                all_entries = resp_fallback.json().get('search-results', {}).get('entry', [])
                            else:
                                st.error(f"❌ Terjadi Error API {resp_fallback.status_code}: {resp_fallback.text}")
                            break
                        else:
                            st.error(f"❌ Terjadi Error API {resp.status_code}: {resp.text}")
                            break

                    if all_entries:
                        loaded_data = clean_scopus_data({'search-results': {'entry': all_entries}})
                        st.session_state.history = [loaded_data]
                        st.session_state.history_actions = ["Data Awal (Scopus API)"]
                        st.session_state.current_step = 0
                        st.session_state.preview_action = None
                        st.session_state.map_rendered = False
                        st.success(f"✅ Berhasil menarik {len(loaded_data)} dokumen! Silakan pilih menu lain di Sidebar untuk mulai menganalisis.")

    with col2:
        st.markdown("#### 📁 Opsi 2: Unggah File Lokal (Local Data)")
        st.info("Bagi pengguna database lain (Web of Science, PubMed, WIPO Patents), Anda bisa mengunggah format .csv, .json, atau Excel ke sini.")
        # 1. Tambahkan ekstensi xls dan xlsx ke dalam parameter type
        uploaded_file = st.file_uploader("Pilih Berkas", type=["csv", "json", "xls", "xlsx"])
        
        if uploaded_file:
            if st.button("Proses File Unggahan", type="primary", use_container_width=True):
                try:
                    # 2. Perbarui logika pembacaan file berdasarkan ekstensinya
                    if uploaded_file.name.endswith(".json"):
                        loaded_data = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith((".xls", ".xlsx")):
                        # 1. Intip 15 baris pertama tanpa menetapkan header
                        df_peek = pd.read_excel(uploaded_file, header=None, nrows=15)
                        header_idx = 0
                        
                        # 2. Loop untuk mencari baris mana yang merupakan Header asli
                        for i, row in df_peek.iterrows():
                            row_vals = [str(x).strip().lower() for x in row.values if pd.notna(x)]
                            # Cek apakah baris ini mengandung nama kolom standar WIPO/Scopus
                            if any(col in row_vals for col in ['application id', 'title', 'judul', 'document title', 'authors', 'penulis']):
                                header_idx = i
                                break
                                
                        # 3. Baca ulang file dengan index header yang tepat (Untuk WIPO biasanya terbaca di index 5 / Baris 6)
                        loaded_data = pd.read_excel(uploaded_file, header=header_idx)
                        st.toast(f"ℹ️ Header otomatis terdeteksi pada baris ke-{header_idx + 1}", icon="🤖")
                    else:
                        # Fallback ke CSV (juga ditambahkan logika skip-metadata jika diperlukan)
                        try:
                            loaded_data = pd.read_csv(uploaded_file, encoding='utf-8')
                        except Exception:
                            loaded_data = pd.read_csv(uploaded_file, encoding='latin1')
                        
                    loaded_data = loaded_data.fillna("Tidak tersedia")
                    st.session_state.history = [loaded_data]
                    st.session_state.history_actions = [f"Data Awal ({uploaded_file.name})"]
                    st.session_state.current_step = 0
                    st.session_state.last_file = uploaded_file.name
                    st.session_state.preview_action = None
                    st.session_state.map_rendered = False
                    st.success(f"✅ Memuat {len(loaded_data)} baris data! Silakan pilih menu di Sidebar untuk melanjutkan.")
                except Exception as e: 
                    st.error(f"❌ Gagal memproses file: Kesalahan Parsing -> {e}")

# ==============================
# MENU-MENU ANALISIS (WAJIB ADA DATA)
# ==============================
elif len(st.session_state.history) > 0:
    base_data = st.session_state.history[st.session_state.current_step].copy()
    data = base_data.copy() 

    # ====== DETEKSI NAMA KOLOM DINAMIS (CACHED, WIPO COMPATIBLE) ======
    col_mappings = get_column_mappings(tuple(data.columns))
    year_col = col_mappings['year_col']
    journal_col = col_mappings['journal_col']
    author_col = col_mappings['author_col']
    title_col = col_mappings['title_col']
    citation_col = col_mappings['citation_col']
    abstract_col = col_mappings['abstract_col']
    affiliation_col = col_mappings['affiliation_col']

    # Robust Year Extraction for Dates (e.g., 26.06.2023 -> 2023)
    if year_col:
        # Mengekstrak 4 angka berurutan yang merepresentasikan tahun (19xx atau 20xx)
        data['Year_Numeric'] = pd.to_numeric(data[year_col].astype(str).str.extract(r'((?:19|20)\d{2})')[0], errors='coerce')

    # ---------------------------------------------------------
    # MENU 2: OVERVIEW & TRENDS
    # ---------------------------------------------------------
    if menu_selection == "📊 Overview & Trends":
        st.title("📊 Dataset Overview")
        st.markdown("Profil deskriptif data eksploratori yang menyajikan informasi statistik kunci.")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📄 Total Dokumen", len(data))
        m2.metric("📋 Total Metadata (Kolom)", len(data.columns))
        
        if year_col and 'Year_Numeric' in data.columns:
            min_y = int(data['Year_Numeric'].min()) if not pd.isna(data['Year_Numeric'].min()) else "N/A"
            max_y = int(data['Year_Numeric'].max()) if not pd.isna(data['Year_Numeric'].max()) else "N/A"
            m3.metric("📅 Rentang Tahun Publikasi", f"{min_y} - {max_y}")
        else:
            m3.metric("📅 Rentang Tahun Publikasi", "Tidak Terdeteksi")
            
        if citation_col:
            total_cites = pd.to_numeric(data[citation_col], errors='coerce').sum()
            m4.metric("📈 Total Sitasi Kumulatif", f"{int(total_cites):,}")
        else:
            m4.metric("📈 Total Sitasi Kumulatif", "N/A")
            
        st.markdown("---")
        
        # TABEL EKPLORASI
        col_view, col_dl = st.columns([4, 1])
        with col_view:
            st.markdown("#### 🗃️ Eksplorasi Data Mentah (Raw Table)")
        with col_dl:
            st.download_button(label="📥 Download Excel/CSV", data=convert_df_to_csv(data), file_name="dataset_overview.csv", mime="text/csv", use_container_width=True)
            
        all_cols = data.columns.tolist()
        default_cols = [c for c in ['Title', 'Judul', 'Authors', 'Penulis', 'Year', 'Tahun', 'Source title', 'Jurnal', 'Cited by', 'Citasi'] if c in all_cols]
        if not default_cols: default_cols = all_cols[:5]
        
        selected_cols = st.multiselect("Filter Kolom yang Ingin Anda Tinjau:", options=all_cols, default=default_cols)
        if selected_cols: 
            st.dataframe(data[selected_cols], use_container_width=True, height=250)

        st.markdown("---")
        st.markdown("#### 📈 Analisis Deskriptif Bibliometrik Dasar")
        
        tab_stats1, tab_stats2, tab_stats3, tab_stats4, tab_stats5, tab_stats6 = st.tabs(["📊 Produksi & Sitasi", "🏢 Hukum Kinerja (Lotka & Bradford)", "🗺️ Peta Produksi Negara", "🧠 Topic Modeling (LDA)", "☁️ Word Cloud", "📈 Tren IPC/Kata Kunci"])
        
        # SUBTAB: PRODUKSI
        with tab_stats1:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if year_col and 'Year_Numeric' in data.columns:
                    st.markdown("**1. Produksi Dokumen Per Tahun**")
                    yearly_counts = data['Year_Numeric'].value_counts().sort_index()
                    fig_year = px.area(x=yearly_counts.index, y=yearly_counts.values, labels={'x': 'Tahun Publikasi', 'y': 'Jumlah Dokumen'})
                    fig_year.update_traces(line_color='#4a90e2', fillcolor='rgba(74, 144, 226, 0.3)')
                    fig_year.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, xaxis=dict(dtick=1))
                    st.plotly_chart(fig_year, use_container_width=True)
                
                if citation_col and title_col:
                    st.markdown("**2. Top 5 Dokumen Paling Banyak Dikutip (High-Impact Papers)**")
                    data[citation_col] = pd.to_numeric(data[citation_col], errors='coerce').fillna(0).astype(int)
                    top_cited = data.sort_values(by=citation_col, ascending=False).head(5)
                    st.dataframe(top_cited[[title_col, citation_col]].reset_index(drop=True), use_container_width=True)

            with col_c2:
                if author_col:
                    st.markdown("**3. Entitas Kreator Paling Produktif (Berdasarkan Frekuensi Kemunculan)**")
                    all_authors = data[author_col].dropna().astype(str).str.split(";").explode().str.strip()
                    top_authors = all_authors[all_authors != ""].value_counts().head(10)
                    fig_auth = px.bar(y=top_authors.index, x=top_authors.values, orientation='h', text=top_authors.values, labels={'y':'Kreator/Penulis', 'x':'Jumlah Dokumen'})
                    fig_auth.update_traces(marker_color='#e74c3c', textposition='outside')
                    max_x_auth = top_authors.values.max() if len(top_authors) > 0 else 10
                    fig_auth.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=10, b=0), height=350, xaxis=dict(range=[0, max_x_auth * 1.2]))
                    st.plotly_chart(fig_auth, use_container_width=True)
                    
                if journal_col:
                    st.markdown("**4. Sumber / Wadah Klasifikasi Teratas**")
                    top_j = data[journal_col].value_counts().head(10)
                    short_labels = [str(label)[:30] + '...' if len(str(label)) > 30 else str(label) for label in top_j.index]
                    fig_j = px.bar(y=short_labels, x=top_j.values, orientation='h', text=top_j.values, labels={'y':'Kategori/Sumber', 'x':'Jumlah Dokumen'})
                    fig_j.update_traces(marker_color='#2ecc71', textposition='outside')
                    max_x_jour = top_j.values.max() if len(top_j) > 0 else 10
                    fig_j.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=10, b=0), height=350, xaxis=dict(range=[0, max_x_jour * 1.2]))
                    st.plotly_chart(fig_j, use_container_width=True)
                    
        # SUBTAB: BRADFORD & LOTKA
        with tab_stats2:
            bl1, bl2 = st.columns(2)
            with bl1:
                st.markdown("**Hukum Penyebaran Bradford (Bradford's Law)**")
                st.caption("Mengidentifikasi Jurnal Inti (Core Journals).")
                if journal_col:
                    bradford_df = calculate_bradford_law(data, journal_col)
                    if not bradford_df.empty:
                        zone_counts = bradford_df['Bradford Zone'].value_counts().sort_index()
                        st.dataframe(zone_counts, use_container_width=True)
                        core_journals = bradford_df[bradford_df['Bradford Zone'] == "Zone 1 (Core)"]
                        st.success(f"Terdapat **{len(core_journals)} Jurnal Inti**.")
                    else: st.warning("Data jurnal tidak memadai.")
            
            with bl2:
                st.markdown("**Hukum Lotka (Penulis & Produktivitas)**")
                st.caption("Distribusi inverse-square dari produksi saintifik.")
                if author_col:
                    lotka_df = calculate_lotkas_law(data, author_col)
                    if not lotka_df.empty:
                        st.dataframe(lotka_df, use_container_width=True)
                        try:
                            fig_lotka, ax_lotka = plt.subplots(figsize=(6, 3))
                            ax_lotka.plot(pd.to_numeric(lotka_df["Jumlah Dokumen Ditulis"]), lotka_df["Persentase Aktual (%)"], marker='o', color='#e74c3c', label='Aktual')
                            ax_lotka.plot(pd.to_numeric(lotka_df["Jumlah Dokumen Ditulis"]), lotka_df["Prediksi Lotka Teoritis (%)"], linestyle='--', color='gray', label='Lotka Law')
                            ax_lotka.set_xlabel('Dokumen')
                            ax_lotka.set_ylabel('% Penulis')
                            ax_lotka.legend()
                            st.pyplot(fig_lotka)
                        except Exception: pass

        # SUBTAB: GEO MAPPING
        with tab_stats3:
            st.markdown("**Peta Choropleth Produksi Saintifik Global**")
            if not HAS_PLOTLY:
                st.warning("Membutuhkan modul Plotly.")
            else:
                st.caption("Mengekstrak metadata negara dari kolom penulis/afiliasi untuk memetakan negara mana yang paling banyak memproduksi literatur pada set data Anda.")
                
                geo_col_target = affiliation_col if affiliation_col else author_col
                
                if geo_col_target:
                    show_map = st.checkbox("🗺️ Analisis & Tampilkan Peta Dunia", value=False)
                    if show_map:
                        with st.spinner("Mengekstrak data geospasial menggunakan Regex..."):
                            all_countries_extracted = []
                            for text in data[geo_col_target].dropna():
                                extracted = extract_countries_from_text(text)
                                all_countries_extracted.extend(extracted)
                                
                            if not all_countries_extracted:
                                st.info("Tidak ada nama negara spesifik yang dapat dikenali dari teks metadata.")
                            else:
                                country_freq = Counter(all_countries_extracted)
                                df_geo = pd.DataFrame(list(country_freq.items()), columns=['Country', 'Count'])
                                df_geo['ISO'] = df_geo['Country'].map(COUNTRY_ISO_MAPPING)
                                
                                fig_map = px.choropleth(df_geo, locations="ISO", color="Count", hover_name="Country", color_continuous_scale=px.colors.sequential.Plasma, projection="natural earth", title="Distribusi Produksi Jurnal Internasional")
                                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                                st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_DL_CONFIG)

        # SUBTAB: TOPIC MODELING (LDA)
        with tab_stats4:
            st.markdown("**Latent Dirichlet Allocation (Machine Learning Topic Modeling)**")
            if not HAS_SKLEARN:
                st.error("💡 Modul Machine Learning `scikit-learn` belum terinstal. Silakan ketik `pip install scikit-learn` di terminal.")
            else:
                st.caption("Algoritma probabilistik ini mencari topik-topik tersembunyi (*latent*) murni dari analisis statistik kemunculan kata pada abstrak tanpa pengawasan (Unsupervised).")
                col_lda1, col_lda2 = st.columns([1, 3])
                with col_lda1:
                    lda_topics = st.slider("Jumlah Topik Laten (k):", 2, 10, 4)
                    lda_words = st.slider("Kata per Topik:", 5, 20, 10)
                    lda_col = st.selectbox("Kolom Sasaran:", [abstract_col, title_col] if abstract_col else [title_col])
                with col_lda2:
                    execute_lda = st.checkbox("🧠 Analisis Ekstraksi Topik (LDA)", value=False)
                    if execute_lda and lda_col:
                        with st.spinner("Melatih Model Scikit-Learn LDA..."):
                            docs_lda = data[lda_col].dropna().astype(str)
                            custom_stop = list(COMMON_STOPWORDS) + ['based', 'approach', 'model', 'method', 'data', 'using']
                            
                            min_df_val = 2 if len(docs_lda) > 5 else 1
                            
                            try:
                                tf_vectorizer = CountVectorizer(max_df=0.95, min_df=min_df_val, stop_words=custom_stop, max_features=1500)
                                tf = tf_vectorizer.fit_transform(docs_lda)
                                
                                lda_model = LatentDirichletAllocation(n_components=lda_topics, random_state=42, max_iter=10)
                                lda_model.fit(tf)
                                
                                feature_names = tf_vectorizer.get_feature_names_out()
                                
                                st.markdown("##### 🧬 Hasil Identifikasi Topik Ekstraksi:")
                                cols_topics = st.columns(2)
                                for topic_idx, topic in enumerate(lda_model.components_):
                                    top_features_ind = topic.argsort()[:-lda_words - 1:-1]
                                    top_features = [feature_names[i] for i in top_features_ind]
                                    
                                    with cols_topics[topic_idx % 2]:
                                        st.info(f"**Topik {topic_idx + 1}**\n\n" + ", ".join([f"`{w}`" for w in top_features]))
                            except ValueError as e:
                                st.error(f"⚠️ Algoritma LDA gagal: Vocabulary kosong (terlalu banyak stopwords yang terfilter atau jumlah data terlalu sedikit). Detail: {str(e)}")
                            
                            gc.collect()
        
        # SUBTAB: WORD CLOUD
        with tab_stats5:
            st.markdown("**Pemetaan Visual Frekuensi Kata Dasar**")
            st.caption("Eksplorasi cepat kata-kata dominan dari keseluruhan dataset naratif.")
            if not HAS_WORDCLOUD:
                st.warning("⚠️ Modul WordCloud tidak terinstal. Ketik `pip install wordcloud`.")
            else:
                wc_col = st.selectbox("Pilih Kolom Sumber Kata:", [abstract_col, title_col] if abstract_col else [title_col], key="wc_col")
                if wc_col and st.button("☁️ Render Word Cloud", type="primary"):
                    with st.spinner("Mengekstrak frekuensi leksikal..."):
                        text = " ".join(data[wc_col].dropna().astype(str))
                        wordcloud = WordCloud(width=1000, height=500, background_color='white', stopwords=COMMON_STOPWORDS, colormap='viridis').generate(text)
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)

        # SUBTAB: TREN IPC / KATA KUNCI
        with tab_stats6:
            st.markdown("**Pemetaan Tren Dinamika Topik / Teknologi**")
            st.caption("Visualisasi pergerakan tren dari kata kunci spesifik (Scopus) atau kode IPC (WIPO) dari tahun ke tahun.")
            
            if not year_col or 'Year_Numeric' not in data.columns:
                st.warning("⚠️ Metadata Tahun Publikasi tidak ditemukan atau tidak dapat diekstrak.")
            else:
                text_cols_trend = data.select_dtypes(include=['object']).columns.tolist()
                if not text_cols_trend:
                    st.warning("Tidak ada kolom teks untuk dianalisis.")
                else:
                    # Coba temukan kolom default yang mengandung kata kunci atau IPC
                    kw_candidates = [c for c in text_cols_trend if any(kw in c.lower() for kw in ['keyword', 'ipc', 'index', 'class'])]
                    default_idx_trend = text_cols_trend.index(kw_candidates[0]) if kw_candidates else 0
                    
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        trend_col = st.selectbox("Pilih Kolom Entitas (Misal: Author Keywords / IPC):", text_cols_trend, index=default_idx_trend)
                    with col_t2:
                        trend_delim = st.selectbox("Delimiter (Pemisah Entitas):", [";", ",", "|", "Spasi ( )", "Tidak Ada"], index=0)
                    
                    if st.button("🚀 Render Garis Tren Temporal", type="primary"):
                        actual_delim = {";": ";", ",": ",", "|": "|", "Spasi ( )": " "}.get(trend_delim)
                        
                        with st.spinner("Menghitung dinamika temporal..."):
                            temp_df = data[['Year_Numeric', trend_col]].copy().dropna()
                            
                            # Pisahkan entitas dengan delimiter
                            if actual_delim:
                                temp_df['Entitas'] = temp_df[trend_col].astype(str).str.split(actual_delim)
                            else:
                                temp_df['Entitas'] = temp_df[trend_col].astype(str).apply(lambda x: [x])
                            
                            temp_df = temp_df.explode('Entitas')
                            temp_df['Entitas'] = temp_df['Entitas'].astype(str).str.strip().str.upper() # Upper case agar konsisten (baik IPC/Keyword)
                            
                           # Pembersihan Entitas
                            stop_upper = {w.upper() for w in COMMON_STOPWORDS}
                            valid_mask = (temp_df['Entitas'] != "") & (~temp_df['Entitas'].isin(stop_upper)) & (temp_df['Entitas'].str.len() > 1) & (~temp_df['Entitas'].isin(["TIDAK TERSEDIA", "N/A", "NO TITLE", "NAN"]))
                            temp_df = temp_df[valid_mask]
                            
                            # 1. Hitung frekuensi total untuk setiap entitas
                            entity_totals = temp_df['Entitas'].value_counts()
                            
                            # 2. Agregasi FULL per tahun dan entitas (Untuk Data Tabel)
                            full_trend_grouped = temp_df.groupby(['Year_Numeric', 'Entitas']).size().reset_index(name='Frekuensi')
                            
                            if not full_trend_grouped.empty:
                                # --- A. RENDER GRAFIK CHART (Hanya Top 10) ---
                                top_entities = entity_totals.head(10).index.tolist()
                                chart_data = full_trend_grouped[full_trend_grouped['Entitas'].isin(top_entities)]
                                
                                fig_trend = px.line(chart_data, x='Year_Numeric', y='Frekuensi', color='Entitas', markers=True,
                                                    title=f"Tren Top 10 '{trend_col}' Sepanjang Waktu",
                                                    labels={'Year_Numeric': 'Tahun', 'Frekuensi': 'Jumlah Kemunculan'})
                                fig_trend.update_layout(xaxis=dict(dtick=1), margin=dict(l=0, r=0, t=40, b=0))
                                st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_DL_CONFIG)
                                
                                # --- B. RENDER TABEL DISTRIBUSI (Seluruh Data) ---
                                st.markdown("---")
                                st.markdown(f"#### 🗃️ Tabel Distribusi Frekuensi Keseluruhan '{trend_col}'")
                                st.caption(f"Menampilkan jumlah kemunculan seluruh **{len(entity_totals)}** variasi kata kunci/IPC per tahun.")
                                
                                # Pivot table agar Tahun menjadi kolom dan Entitas menjadi baris
                                pivot_df = full_trend_grouped.pivot(index='Entitas', columns='Year_Numeric', values='Frekuensi').fillna(0).astype(int)
                                
                                # Tambahkan kolom Total Keseluruhan agar mudah disortir
                                pivot_df['Total Keseluruhan'] = pivot_df.sum(axis=1)
                                
                                # Urutkan dari yang kemunculannya paling banyak
                                pivot_df = pivot_df.sort_values(by='Total Keseluruhan', ascending=False)
                                
                                # Tampilkan Tabel Interaktif
                                st.dataframe(pivot_df, use_container_width=True, height=350)
                                
                                # Tambahkan tombol Download khusus untuk tabel ini
                                st.download_button(
                                    label="📥 Unduh Seluruh Data Tabel (.csv)",
                                    data=pivot_df.reset_index().to_csv(index=False).encode('utf-8'),
                                    file_name=f"distribusi_{trend_col}_lengkap.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.info("Data tidak mencukupi untuk visualisasi tren.")

    # ---------------------------------------------------------
    # MENU 3: DATA CLEANING (OPENREFINE)
    # ---------------------------------------------------------
    elif menu_selection == "🧹 Data Cleaning":
        st.title("🧹 Data Cleaning (Wrangler)")
        st.markdown("Studio interaktif untuk mendeteksi duplikat, merapikan sel majemuk (*Split*), dan menyatukan inkonsistensi teks (*Clustering*).")
        
        col_u, col_r, col_stat = st.columns([1, 1, 4])
        with col_u:
            if st.button("↩️ Undo Action", disabled=(st.session_state.current_step == 0), use_container_width=True):
                st.session_state.current_step -= 1
                st.session_state.preview_action = None
                st.rerun()
        with col_r:
            if st.button("↪️ Redo Action", disabled=(st.session_state.current_step == len(st.session_state.history) - 1), use_container_width=True):
                st.session_state.current_step += 1
                st.session_state.preview_action = None
                st.rerun()
        with col_stat:
            st.info(f"📍 **Status Riwayat:** {st.session_state.history_actions[st.session_state.current_step]} (Langkah {st.session_state.current_step + 1} dari {len(st.session_state.history)})")
            if len(st.session_state.history_actions) > 1:
                log_data = json.dumps(st.session_state.history_actions, indent=2)
                st.download_button("💾 Ekspor Macro/Resep (JSON)", data=log_data, file_name="resep_pembersihan.json", mime="application/json")
        
        st.markdown("---")

        st.markdown("#### 🗑️ 1. Panel Deduplikasi Dokumen")
        st.caption("Pilih kolom acuan (seperti Judul atau DOI) untuk mendeteksi dokumen yang identik.")
        col_dup1, col_dup2, col_dup3 = st.columns([2, 1, 1])
        
        dup_default_idx = data.columns.tolist().index(title_col) if title_col in data.columns else 0
        
        with col_dup1:
            dup_col = st.selectbox("Pilih Kolom Acuan Validasi (Unik):", data.columns, index=dup_default_idx, key="dup_col")
        with col_dup2:
            st.markdown("<br>", unsafe_allow_html=True)
            dup_series = data[dup_col].astype(str).str.lower().str.strip()
            duplicate_mask = dup_series.duplicated(keep='first') & (dup_series != "tidak tersedia") & (dup_series != "n/a")
            duplicate_count = duplicate_mask.sum()
            if duplicate_count > 0:
                st.error(f"⚠️ Terdeteksi {duplicate_count} Baris Duplikat!")
            else:
                st.success("✅ Bersih (0 Duplikat Exact)")
        with col_dup3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sapu Duplikat", type="primary", use_container_width=True, disabled=(duplicate_count == 0)):
                def remove_duplicates_func(df):
                    temp_series = df[dup_col].astype(str).str.lower().str.strip()
                    mask = temp_series.duplicated(keep='first') & (temp_series != "tidak tersedia") & (temp_series != "n/a")
                    return df[~mask]
                apply_transform(remove_duplicates_func, f"Drop {duplicate_count} Exact Duplicates on '{dup_col}'", is_row_filter=True, target_col=dup_col)

        st.markdown("---")

        text_cols = base_data.select_dtypes(include=['object']).columns.tolist()
        exclude_cols = [c for c in [title_col, abstract_col, 'DOI'] if c]
        cleanable_cols = [c for c in text_cols if c not in exclude_cols]
        
        st.markdown("#### 🧹 2. Manipulasi & Transformasi Sel Spesifik")
        if not cleanable_cols:
            st.warning("⚠️ Tidak ada kolom tipe String (Teks) yang valid untuk dibersihkan.")
        else:
            col_c1, col_c2 = st.columns([1, 2])
            
            with col_c1:
                target_clean_col = st.selectbox("Pilih Kolom Target Operasi:", cleanable_cols)
                
                with st.expander("📊 Distribusi Kemunculan Nilai (Text Facet)", expanded=False):
                    facet_df = base_data[target_clean_col].value_counts().reset_index()
                    facet_df.columns = ['Nilai String Unik', 'Frekuensi']
                    st.dataframe(facet_df, height=200, use_container_width=True)

                st.markdown("---")
                
                # REORDERED DATA CLEANING UI
                st.markdown("**✂️ A. Pecah Sel Multi-Nilai (Split Multi-valued Cells)**")
                st.caption("Penting dilakukan untuk kolom 'Penulis' atau 'Author Keywords' agar setiap entitas dipisah menjadi barisnya sendiri.")
                col_sd1, col_sd2 = st.columns([3, 1])
                with col_sd1:
                    split_delim = st.selectbox("Pemisah (Delimiter):", ["Titik Koma (;)", "Koma (,)", "Pemisah Garis Lurus (|)", "Spasi ( )"], label_visibility="collapsed")
                with col_sd2:
                    if st.button("Pecah", use_container_width=True):
                        actual_delim = { "Titik Koma (;)": ";", "Koma (,)": ",", "Pemisah Garis Lurus (|)": "|", "Spasi ( )": " " }.get(split_delim)
                        def split_cells(df):
                            df_out = df.copy()
                            df_out[target_clean_col] = df_out[target_clean_col].astype(str).str.split(actual_delim)
                            df_out = df_out.explode(target_clean_col)
                            df_out[target_clean_col] = df_out[target_clean_col].str.strip()
                            df_out = df_out[df_out[target_clean_col] != ""]
                            return df_out
                        apply_transform(split_cells, f"Split Cells '{target_clean_col}' by '{actual_delim}'", is_row_filter=True, target_col=target_clean_col)

                st.markdown("---")
                st.markdown("**✂️ B. Trim Extra Whitespace**")
                if st.button("Hilangkan Spasi Ganda yang Berlebihan", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x), f"Trim Spasi '{target_clean_col}'", target_col=target_clean_col)

                st.markdown("---")
                st.markdown("**✨ C. Transformasi Teks Instan**")
                col_case1, col_case2, col_case3 = st.columns(3)
                if col_case1.button("UPPERCASE", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).str.upper(), f"To UPPERCASE '{target_clean_col}'", target_col=target_clean_col)
                if col_case2.button("lowercase", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).str.lower(), f"To lowercase '{target_clean_col}'", target_col=target_clean_col)
                if col_case3.button("Title Case", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).str.title(), f"To Title Case '{target_clean_col}'", target_col=target_clean_col)

                st.markdown("---")
                st.markdown("**🗑️ D. Hapus Baris Kosong & Find Replace**")
                if st.button("Drop Baris Mengandung Data Kosong (NaN/Tidak Tersedia)", use_container_width=True):
                    def remove_blank_rows(df):
                        mask = df[target_clean_col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "tidak tersedia"])
                        return df[~mask]
                    apply_transform(remove_blank_rows, f"Hapus Baris Kosong '{target_clean_col}'", is_row_filter=True, target_col=target_clean_col)
                
                col_f, col_r = st.columns(2)
                with col_f:
                    find_txt = st.text_input("Teks dicari:", placeholder="Old")
                with col_r:
                    replace_txt = st.text_input("Pengganti:", placeholder="New")
                if st.button("Terapkan Ganti Nilai (Replace)", use_container_width=True):
                    if find_txt:
                        apply_transform(lambda col: col.astype(str).str.replace(find_txt, replace_txt, regex=False), f"Replace '{find_txt}' -> '{replace_txt}' di '{target_clean_col}'", target_col=target_clean_col)
                    else:
                        st.warning("Kotak 'Teks dicari' tidak boleh kosong.")

                st.markdown("---")
                st.markdown("**🧠 E. Clustering / Penggabungan Varian Kata**")
                st.caption("Deteksi typo, singkatan, atau pluralitas dan gabungkan menjadi satu istilah standar.")
                method = st.selectbox("Algoritma Deteksi:", ["Fingerprint", "N-Gram (2-gram)", "Phonetic (Soundex)", "Levenshtein Distance", "PPM (Compression)"])
                delimiter_opt = st.selectbox("Multi-value Delimiter Target:", ["Tidak Ada", "Titik Koma (;)", "Koma (,)", "Pemisah Garis Lurus (|)"])
                used_delim = {"Titik Koma (;)": ";", "Koma (,)": ",", "Pemisah Garis Lurus (|)": "|"}.get(delimiter_opt, None)
                
                lev_threshold, ppm_threshold = 2, 0.3
                if method == "Levenshtein Distance": 
                    lev_threshold = st.slider("Toleransi Levenshtein (Jarak Edit Huruf):", 1, 5, 2)
                elif method == "PPM (Compression)": 
                    ppm_threshold = st.slider("Toleransi NCD/PPM:", 0.1, 0.9, 0.3, 0.05)
                
                use_ai_suggestion = st.checkbox("🤖 Integrasi AI (Semantik Validation)", help="AI akan membaca isi klaster dan memvalidasi apakah variasi benar-benar bersinonim.")
                
                if st.button("🔍 Mulai Pindai Klaster", type="primary", use_container_width=True):
                    if use_ai_suggestion and not AI_API_KEY:
                        st.error(f"⚠️ {AI_PROVIDER} API Key diperlukan. Periksa menu Settings di Sidebar.")
                    else:
                        with st.spinner(f"Mesin sedang mengekstrak variasi pada kolom '{target_clean_col}'..."):
                            kw_series = base_data[target_clean_col].dropna().astype(str).str.split(used_delim).explode().str.strip() if used_delim else base_data[target_clean_col].dropna().astype(str).str.strip()
                            kw_series = kw_series[kw_series != ""]
                            kw_freq = kw_series.value_counts().to_dict()
                            unique_kws = list(kw_freq.keys())
                            
                            # Cek Overload
                            if len(unique_kws) > 1000 and method in ["Levenshtein Distance", "PPM (Compression)"]:
                                st.warning(f"⚠️ Terdeteksi {len(unique_kws)} entitas unik. Algoritma {method} mungkin memakan waktu lebih lama karena kompleksitas O(N²).")

                            valid_clusters = {}
                            if method in ["Fingerprint", "N-Gram (2-gram)", "Phonetic (Soundex)"]:
                                clusters = {}
                                for kw in unique_kws:
                                    if method == "Fingerprint": key = get_fingerprint(kw)
                                    elif method == "Phonetic (Soundex)": key = get_phonetic_fingerprint(kw)
                                    else: key = get_ngram_fingerprint(kw, 2)
                                    if key not in clusters: clusters[key] = []
                                    clusters[key].append(kw)
                                valid_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
                            else: 
                                visited = set()
                                cluster_id = 0
                                progress_bar = st.progress(0)
                                total_kws = len(unique_kws)
                                for i, kw1 in enumerate(unique_kws):
                                    if i % max(1, total_kws // 20) == 0: progress_bar.progress(min(i / total_kws, 1.0))
                                    if kw1 in visited: continue
                                    current_cluster = [kw1]
                                    visited.add(kw1)
                                    for j in range(i + 1, total_kws):
                                        kw2 = unique_kws[j]
                                        if kw2 in visited: continue
                                        match = False
                                        if method == "Levenshtein Distance":
                                            if abs(len(kw1) - len(kw2)) <= lev_threshold and levenshtein(kw1.lower(), kw2.lower(), lev_threshold) <= lev_threshold: 
                                                match = True
                                        elif method == "PPM (Compression)" and ppm_distance(kw1.lower(), kw2.lower()) <= ppm_threshold: match = True
                                        
                                        if match:
                                            current_cluster.append(kw2)
                                            visited.add(kw2)
                                    if len(current_cluster) > 1:
                                        valid_clusters[f"Cluster_{cluster_id}"] = current_cluster
                                        cluster_id += 1
                                progress_bar.empty()
                            
                            if not valid_clusters:
                                st.session_state.clustering_result = None
                                st.success("🎉 Dataset sudah bersih! Tidak ada klaster duplikat ditemukan.")
                            else:
                                ai_suggestions = {}
                                if use_ai_suggestion:
                                    total_clusters = len(valid_clusters)
                                    st.info(f"⏳ Mengevaluasi semantik {total_clusters} klaster dengan LLM...")
                                    with st.spinner(f"🤖 Mesin AI ({AI_MODEL}) sedang bekerja memvalidasi kata..."):
                                        prompt_clusters = {f"Cluster_{idx}": items for idx, (key, items) in enumerate(valid_clusters.items())}
                                        sys_prompt_ai = """Anda adalah pakar data bibliometrik akademis. Evaluasi klaster kata berikut. Tentukan apakah bermakna sama. Output HARUS murni JSON valid: {"Cluster_X": {"gabung": true, "standar": "Nama Baku", "alasan": "Alasan"}}"""
                                        usr_prompt_ai = f"Data Klaster Input:\n{json.dumps(prompt_clusters, indent=2)}"
                                        
                                        # PERUBAHAN GROQ CALL
                                        if AI_PROVIDER == "Mistral": ai_response = call_mistral_sync(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                        elif AI_PROVIDER == "Google Gemini": ai_response = call_gemini_sync(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                        elif AI_PROVIDER == "Groq": ai_response = call_groq_sync(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                            
                                        try:
                                            match = re.search(r'\{[\s\S]*\}', ai_response)
                                            ai_suggestions = json.loads(match.group(0)) if match else json.loads(ai_response)
                                        except Exception as e:
                                            st.warning("⚠️ Gagal memparsing JSON dari respon AI. Jatuh kembali ke mode Manual / Bawaan.")

                                cluster_data = []
                                for idx, (key, items) in enumerate(valid_clusters.items()):
                                    c_id = f"Cluster_{idx}"
                                    standard = max(items, key=lambda x: kw_freq[x]) 
                                    gabung, alasan = True, "-"
                                    
                                    if ai_suggestions and c_id in ai_suggestions:
                                        sugg = ai_suggestions[c_id]
                                        gabung = sugg.get("gabung", True)
                                        standard = sugg.get("standar", standard) or standard
                                        alasan = sugg.get("alasan", "-")

                                    for item in items:
                                        item_gabung = True if item == standard else gabung
                                        cluster_data.append({
                                            "ID Klaster": f"Klaster {idx+1}",
                                            "Gabung?": item_gabung,
                                            "Variasi Asli": item,
                                            "Kata Baku Baru (Edit Sini)": standard,
                                            "Freq": kw_freq[item],
                                            "Saran/Alasan AI": alasan
                                        })
                                    
                                st.session_state.clustering_result = pd.DataFrame(cluster_data)
                                st.session_state.target_clean_col = target_clean_col
                                st.session_state.used_delim = used_delim

            with col_c2:
                st.download_button(label="📥 Unduh Keadaan Dataset Saat Ini (.csv)", data=convert_df_to_csv(base_data), file_name="scopus_dataset_cleaned_progress.csv", mime="text/csv", use_container_width=True)
                st.markdown("---")
                
                if st.session_state.get('preview_action'):
                    st.success(f"✅ **Tindakan Transformasi Selesai:** {st.session_state.preview_action}")
                    st.markdown("##### 🔍 Layar Pratinjau (Sebelum vs Sesudah)")
                    orig_col = st.session_state.preview_original
                    new_col = st.session_state.preview_new
                    if orig_col is not None and new_col is not None:
                        if len(orig_col) != len(new_col):
                            if len(new_col) > len(orig_col):
                                st.info(f"Split Baris: menjadi **{len(new_col)}** baris.")
                                st.dataframe(pd.DataFrame({"Hasil Pecahan": new_col.head(10)}), use_container_width=True)
                            else:
                                deleted_idx = orig_col.index.difference(new_col.index)
                                st.info(f"Dihapus **{len(deleted_idx)}** baris (Misal: Baris Kosong/Duplikat).")
                        else:
                            try:
                                changed_mask = orig_col != new_col
                                changed_idx = changed_mask[changed_mask].index
                                if len(changed_idx) > 0:
                                    st.info(f"Modifikasi pada **{len(changed_idx)}** baris.")
                                    st.dataframe(pd.DataFrame({"Teks Asli": orig_col.loc[changed_idx[:5]], "Teks Terdampak": new_col.loc[changed_idx[:5]]}), use_container_width=True)
                                else:
                                    st.info("Tidak ada string yang terubah (sudah sesuai target).")
                            except ValueError: pass
                    
                    if st.button("✖️ Tutup Panel Pratinjau"):
                        st.session_state.preview_action = None
                        st.rerun()
                st.markdown("---")
                
                if st.session_state.clustering_result is not None:
                    st.write(f"⚠️ Ditemukan **{len(st.session_state.clustering_result['ID Klaster'].unique())}** grup variasi kata.")
                    edited_df = st.data_editor(
                        st.session_state.clustering_result,
                        column_config={
                            "ID Klaster": st.column_config.TextColumn("Grup Tautan", disabled=True),
                            "Gabung?": st.column_config.CheckboxColumn("Merge (Gabung)?", default=True),
                            "Variasi Asli": st.column_config.TextColumn("Variasi Terdeteksi", disabled=True),
                            "Kata Baku Baru (Edit Sini)": st.column_config.TextColumn("Nilai Akhir (Sel Edit)"),
                            "Freq": st.column_config.NumberColumn("Jml Muncul", disabled=True),
                            "Saran/Alasan AI": st.column_config.TextColumn("Sintesis Penjelasan AI", disabled=True)
                        },
                        hide_index=True, use_container_width=True, height=600
                    )
                    
                    if st.button("Proses Eksekusi Penggabungan (Merge Selected Clusters)", type="primary"):
                        mapping = {}
                        for index, row in edited_df.iterrows():
                            if row["Gabung?"]: mapping[row["Variasi Asli"].strip()] = row["Kata Baku Baru (Edit Sini)"]
                        
                        if mapping:
                            def cluster_apply(col):
                                def apply_mapping(val):
                                    if pd.isna(val): return val
                                    if st.session_state.used_delim:
                                        kws = [k.strip() for k in str(val).split(st.session_state.used_delim)]
                                        new_kws = [mapping.get(k, k) for k in kws if k]
                                        return f"{st.session_state.used_delim} ".join(sorted(list(set(new_kws))))
                                    else:
                                        val_str = str(val).strip()
                                        return mapping.get(val_str, val_str)
                                return col.apply(apply_mapping)
                            
                            st.session_state.clustering_result = None
                            apply_transform(cluster_apply, f"Merge Variasi Klaster pada '{st.session_state.target_clean_col}'", target_col=st.session_state.target_clean_col)
                        else:
                            st.warning("Kotak Merge kosong semua. Centang minimal satu baris variasi.")


    # ---------------------------------------------------------
    # MENU 4: AI SYNTHESIS (Batching System)
    # ---------------------------------------------------------
    elif menu_selection == "🤖 AI Synthesis":
        st.title("🤖 Generative AI Literature Synthesis")
        st.markdown("Mesin analitik kualitatif. Memanfaatkan Large Language Models (LLM) untuk membaca seluruh metadata abstrak dan mengekstrak narasi kesimpulan secara massal (Batching).")
        
        ipc_col = col_mappings.get('ipc_col')
        
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            target_col = st.selectbox("Pilih Kolom Narasi Target (Misal: Abstract / Title):", data.columns)
            analysis_task = st.selectbox("Jenis Tugas Sintesis NLP:", [
                "Buatkan Ringkasan Eksekutif (Meta-Review)", 
                "Identifikasi Tren & Kesenjangan Riset (Research Gaps)", 
                "Klasifikasikan Metodologi Utama yang Digunakan", 
                "Ekstrak Kata Kunci Utama & Sekunder Tersembunyi", 
                "Tanya Bebas (Custom Master Prompt)"
            ])
            custom_prompt = st.text_area("Parameter Instruksi (Prompt) Khusus:", placeholder="Contoh: Fokuskan pencarian pada limitasi algoritma jaringan saraf tiruan...") if analysis_task == "Tanya Bebas (Custom Prompt)" else ""
        
        with col_cfg2:
            num_samples = st.slider("Alokasi Maksimum Dokumen yang Dibaca (N):", 1, min(200, len(data)), min(30, len(data)), help="Semakin besar jumlahnya, waktu tunggu AI akan semakin lama. Proses dibatch per 15 dokumen.")
            sort_options = ["Urutan Alami Dokumen (Ascending)"]
            if year_col: sort_options.extend(["Terbaru (Descending Year)", "Terlama (Ascending Year)"])
            if citation_col: sort_options.append("High-Impact (Sitasi Terbanyak)")
            sort_order = st.selectbox("Prioritas Penyaringan Corpus:", sort_options)
            
            ipc_filter = ""
            if ipc_col:
                ipc_filter = st.text_input("Syarat Tambahan: Filter Kode IPC (Opsional)", placeholder="Misal: A61K. Kosongkan = Tidak Ada")

        if st.button("🚀 Eksekusi Mesin AI Batching", type="primary"):
            if not AI_API_KEY: 
                st.error(f"⚠️ Kredensial {AI_PROVIDER} API Key hilang. Harap lengkapi di menu Expand Settings Sidebar.")
            elif analysis_task == "Tanya Bebas (Custom Prompt)" and not custom_prompt.strip(): 
                st.warning("⚠️ Untuk Custom Prompt, area ketik instruksi mutlak diperlukan.")
            else:
                ai_data = data.copy()
                
                # FILTER IPC
                if ipc_col and ipc_filter.strip():
                    ai_data = ai_data[ai_data[ipc_col].astype(str).str.contains(ipc_filter.strip(), case=False, na=False)]
                    if ai_data.empty:
                        st.error(f"⚠️ Tidak ditemukan dokumen dengan kode IPC yang mengandung '{ipc_filter}'.")
                        
                if not ai_data.empty:
                    if sort_order == "Terbaru (Descending Year)" and year_col: 
                        if 'Year_Numeric' in ai_data.columns:
                            ai_data = ai_data.sort_values(by='Year_Numeric', ascending=False)
                    elif sort_order == "Terlama (Ascending Year)" and year_col: 
                        if 'Year_Numeric' in ai_data.columns:
                            ai_data = ai_data.sort_values(by='Year_Numeric', ascending=True)
                    elif sort_order == "High-Impact (Sitasi Terbanyak)" and citation_col:
                        ai_data[citation_col] = pd.to_numeric(ai_data[citation_col], errors='coerce').fillna(0)
                        ai_data = ai_data.sort_values(by=citation_col, ascending=False)
                    
                    docs_to_process = []
                    sampled_titles = [] 
                    
                    for i in range(min(num_samples, len(ai_data))):
                        doc_text = str(ai_data[target_col].iloc[i])
                        # Potong panjang abstrak max 1500 char untuk hemat token API
                        doc_text_safe = doc_text[:1500] + "..." if len(doc_text) > 1500 else doc_text

                        if title_col:
                            doc_title = str(ai_data[title_col].iloc[i])
                            sampled_titles.append(f"**[{i+1}]** {doc_title}")
                            docs_to_process.append(f"Dokumen {i+1} [TI: {doc_title}]:\n{doc_text_safe}\n\n")
                        else:
                            sampled_titles.append(f"**[{i+1}]** (Metadata Judul tidak direferensikan)")
                            docs_to_process.append(f"Dokumen {i+1}:\n{doc_text_safe}\n\n")

                    task_to_send = custom_prompt if analysis_task == "Tanya Bebas (Custom Prompt)" else analysis_task
                    system_prompt = f"Anda adalah Profesor dan Analis Riset Akademik Senior. Lakukan insturksi berikut terhadap batch dokumen ini: {task_to_send}\nATURAN KETAT: 1. Selalu referensikan dokumen dengan 'Judul Asli' miliknya. 2. JANGAN mengubah/menerjemahkan frasa 'Judul Asli'. 3. Keluaran harus menggunakan kaidah Bahasa Indonesia yang tinggi, tertata, dan saintifik."

                    st.markdown("---")
                    st.markdown(f"### 📝 Output Laporan Dinamis ({AI_PROVIDER})")
                    with st.expander(f"📄 Indeks Corpus Terpilih (Total {len(sampled_titles)} Dokumen)", expanded=False):
                        for title in sampled_titles: st.markdown(title)
                    
                    BATCH_SIZE = 15 
                    full_report_text = ""
                    
                    for i in range(0, len(docs_to_process), BATCH_SIZE):
                        batch = docs_to_process[i:i+BATCH_SIZE]
                        formatted_texts = "".join(batch)
                        
                        chunk_no = i//BATCH_SIZE + 1
                        total_chunks = (len(docs_to_process)-1)//BATCH_SIZE + 1
                        user_prompt = f"Konstan. Berikut adalah Data Batch ({chunk_no} from {total_chunks}). Evaluasi bagian corpus ini:\n\n{formatted_texts}"
                        
                        st.markdown(f"##### ⏳ Streaming Inference (Batch {chunk_no}/{total_chunks}) [Dokumen ke-{i+1} s/d {i+len(batch)}]...")
                        with st.container(border=True):
                            # PERUBAHAN GROQ CALL
                            if AI_PROVIDER == "Mistral":
                                result_text = st.write_stream(stream_mistral(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                            elif AI_PROVIDER == "Google Gemini":
                                result_text = st.write_stream(stream_gemini(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                            elif AI_PROVIDER == "Groq":
                                result_text = st.write_stream(stream_groq(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                            
                            if "❌ Error" not in result_text:
                                full_report_text += f"\n\n=======================================\nSINTESIS BATCH {chunk_no} (DOKUMEN {i+1}-{i+len(batch)})\n=======================================\n" + result_text

                    if full_report_text:
                        st.success("✅ Seluruh antrian Batch berhasil dieksekusi secara berurutan.")
                        st.download_button("📥 Ekspor Laporan Skrip (.txt)", data=full_report_text, file_name="laporan_ai_sintesis_full.txt", mime="text/plain", type="primary")

                    gc.collect()

    # ---------------------------------------------------------
    # MENU 5: CONCEPTUAL STRUCTURE (SCIENCE MAPPING - BIBLIOSHINY)
    # ---------------------------------------------------------
    elif menu_selection == "🕸️ Conceptual Structure":
        st.title("🕸️ Conceptual Structure (Science Mapping)")
        st.markdown("Merupakan inti dari perangkat lunak bibliometrik. Mentransformasikan data teks majemuk menjadi **Pemetaan Topologi Kuantitatif** dan mengklasifikasikan aliran riset ke dalam berbagai kuadran.")

        if not HAS_NETWORKX or not HAS_PLOTLY:
            st.error("💡 Modul matematika topologi (NetworkX, Numpy, & Plotly) belum terinstal/terhubung.")
            st.code("pip install networkx numpy plotly", language="bash")
        else:
            text_cols_b = base_data.select_dtypes(include=['object']).columns.tolist()
            exclude_cols_b = [c for c in [title_col, abstract_col, 'DOI'] if c]
            cleanable_cols_b = [c for c in text_cols_b if c not in exclude_cols_b]

            if not cleanable_cols_b:
                st.warning("⚠️ Formulasi Graphing membutuhkan minimal satu entitas kolom string (Contoh: Author Keywords).")
            else:
                # ==================================
                # UI KONFIGURASI GLOBAL (DIBAGI 2)
                # ==================================
                st.markdown("**⚙️ Global Mapping Parameters**")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    analysis_type = st.radio("Model Jaringan Analisis:", ["Kookurensi Kata (Co-word)", "Kolaborasi Penulis (Co-authorship)"])
                    is_author_network = (analysis_type == "Kolaborasi Penulis (Co-authorship)")
                with col_g2:
                    if is_author_network:
                        default_idx = cleanable_cols_b.index(author_col) if author_col in cleanable_cols_b else 0
                        net_col = st.selectbox("Entitas Pemetaan (Nodes):", cleanable_cols_b, index=default_idx, key="net_col_author")
                    else:
                        kw_candidates = [c for c in cleanable_cols_b if 'keyword' in c.lower() or 'authkey' in c.lower() or 'index' in c.lower() or 'ipc' in c.lower()]
                        default_idx = cleanable_cols_b.index(kw_candidates[0]) if kw_candidates else 0
                        net_col = st.selectbox("Entitas Pemetaan (Nodes):", cleanable_cols_b, index=default_idx, key="net_col_coword")
                        
                    net_delim = st.selectbox("Format Pemisah Internal:", [";", ",", "|", "Tidak Ada"], index=0, key=f"delim_{is_author_network}")

                st.markdown("---")
                
                with st.expander("🛠️ Pengaturan Lanjutan (Advanced Parameters - Biblioshiny Style)", expanded=True):
                    col_set1, col_set2 = st.columns(2)
                    
                    with col_set1:
                        st.markdown("#### 🕸️ Co-occurrence Network Settings")
                        
                        st.markdown("<p style='color:#2c3e50; font-weight:bold; margin-bottom:0;'>Method Parameters</p>", unsafe_allow_html=True)
                        col_n1, col_n2 = st.columns(2)
                        net_layout = col_n1.selectbox("Network Layout", ["Automatic layout", "Circle layout", "Kamada-Kawai", "Fruchterman-Reingold"], index=0)
                        net_algo = col_n2.selectbox("Clustering Algorithm (Network)", ["Louvain", "Betweenness", "InfoMap", "Leading Eigenvector", "Leiden", "Spinglass", "Walktrap"], index=0, key="net_algo")
                        net_norm = col_n1.selectbox("Normalization Method", ["Association", "Equivalence", "Jaccard", "Salton", "Inclusion", "Raw"], index=0)
                        net_color_year = col_n2.selectbox("Node Color by Year", ["No", "Yes"], index=0)

                        st.markdown("<p style='color:#2c3e50; font-weight:bold; margin-bottom:0; margin-top:10px;'>Network Size</p>", unsafe_allow_html=True)
                        col_n3, col_n4 = st.columns(2)
                        net_n_nodes = col_n3.number_input("Number of Nodes", 10, 500, 50, step=10, key="net_n_nodes")
                        net_repulsion = col_n4.number_input("Repulsion Force (Network)", 0.05, 5.0, 0.5, step=0.05, key="net_repulsion")

                        st.markdown("<p style='color:#f39c12; font-weight:bold; margin-bottom:0; margin-top:10px;'>Filtering Options</p>", unsafe_allow_html=True)
                        col_n5, col_n6 = st.columns(2)
                        net_remove_isolates = col_n5.selectbox("Remove Isolated Nodes", ["Yes", "No"], index=0)
                        net_min_edges = col_n6.number_input("Minimum Number of Edges", 1, 20, 2, key="net_min_edges")

                    with col_set2:
                        st.markdown("#### 📍 Parameters (Thematic Map)")
                        
                        st.markdown("<p style='color:#2c3e50; font-weight:bold; margin-bottom:0;'>Data Parameters</p>", unsafe_allow_html=True)
                        col_t1, col_t2 = st.columns(2)
                        theme_n_words = col_t1.number_input("Number of Words", 10, 1000, 250, step=10)
                        theme_min_freq = col_t2.number_input("Min Cluster Frequency (per thousand docs)", 1, 50, 5)

                        st.markdown("<p style='color:#27ae60; font-weight:bold; margin-bottom:0; margin-top:10px;'>Display Parameters</p>", unsafe_allow_html=True)
                        col_t3, col_t4 = st.columns(2)
                        theme_num_labels = col_t3.number_input("Number of Labels", 1, 10, 3)
                        theme_label_size = col_t4.number_input("Label Size", 0.1, 2.0, 0.3, step=0.1)

                        st.markdown("<p style='color:#e91e63; font-weight:bold; margin-bottom:0; margin-top:10px;'>Network Parameters</p>", unsafe_allow_html=True)
                        col_t5, col_t6 = st.columns(2)
                        theme_repulsion = col_t5.number_input("Community Repulsion", 0.05, 5.0, 0.5, step=0.05, key="thm_rep")
                        theme_algo = col_t6.selectbox("Clustering Algorithm (Thematic)", ["Louvain", "Betweenness", "InfoMap", "Leading Eigenvector", "Leiden", "Spinglass", "Walktrap"], index=0, key="thm_algo")

                btn_map = st.button("🚀 Eksekusi Render Pemetaan (Build Maps)", type="primary", use_container_width=True)
                
                # REAKTIVITAS KODE: MENGGUNAKAN VARIABEL SLIDER SECARA LANGSUNG
                if btn_map:
                    st.session_state.map_rendered = True

                if st.session_state.get('map_rendered', False):
                    actual_net_delim = {";": ";", ",": ",", "|": "|"}.get(net_delim)
                    
                    # ---------------------------------------------------------
                    # 6-STEP PREPROCESSING ENGINE FOR SCIENCE MAPPING
                    # ---------------------------------------------------------
                    with st.spinner("Mengeksekusi matriks jarak dan struktur graf secara otomatis..."):
                        
                        # PIPELINE 1: NETWORK MAP (Co-occurrence)
                        df_mapped_net = preprocess_keywords(data, field=net_col, delimiter=actual_net_delim, is_author=is_author_network)
                        
                        G_raw_net, wc_net = build_cooccurrence(df_mapped_net, field=net_col, minfreq=1)
                        top_words_net = [w for w, c in Counter(wc_net).most_common(net_n_nodes)]
                        G_raw_net = G_raw_net.subgraph(top_words_net).copy()

                        G_norm_net = normalize_network(G_raw_net, method=net_norm)
                        G_final_net = filter_edges(G_norm_net, min_raw_weight=net_min_edges, min_norm_weight=0.0)

                        if net_remove_isolates == "Yes":
                            G_final_net.remove_nodes_from(list(nx.isolates(G_final_net)))

                        if len(G_final_net.nodes()) > 0:
                            communities_net = detect_clusters(G_final_net, method=net_algo)
                            
                            # Layout Mapping
                            if net_layout == "Circle layout":
                                pos_net = nx.circular_layout(G_final_net)
                            elif net_layout == "Kamada-Kawai":
                                pos_net = nx.kamada_kawai_layout(G_final_net)
                            else:
                                pos_net = nx.spring_layout(G_final_net, k=net_repulsion, iterations=150, seed=42)
                        else:
                            communities_net = []
                            pos_net = {}

                        # PIPELINE 2: THEMATIC MAP
                        total_documents = len(df_mapped_net) # Reuse mapped DF
                        actual_min_freq_theme = max(1, math.ceil((theme_min_freq / 1000) * total_documents))

                        G_raw_theme, wc_theme = build_cooccurrence(df_mapped_net, field=net_col, minfreq=actual_min_freq_theme)
                        
                        top_words_theme = [w for w, c in Counter({w: c for w, c in wc_theme.items() if c >= actual_min_freq_theme}).most_common(theme_n_words)]
                        G_raw_theme = G_raw_theme.subgraph(top_words_theme).copy()
                        G_raw_theme.remove_nodes_from(list(nx.isolates(G_raw_theme)))

                        G_norm_theme = normalize_network(G_raw_theme, method="Association") 
                        G_final_theme = filter_edges(G_norm_theme, min_raw_weight=1, min_norm_weight=0.0)

                        if len(G_final_theme.nodes()) > 0:
                            communities_theme = detect_clusters(G_final_theme, method=theme_algo)
                            df_theme = compute_callon_metrics(G_final_theme, communities_theme, wc_theme)
                        else:
                            communities_theme = []
                            df_theme = pd.DataFrame()

                    network_title = "🕸️ Co-authorship Network" if is_author_network else "🕸️ Co-occurrence Network"
                    tab_net, tab_map, tab_evol, tab_three, tab_table = st.tabs([
                        network_title, "📍 Thematic Map", "⏳ Thematic Evolution", 
                        "🔀 Three-Fields Plot", "📊 Network Analytics"
                    ])
                    
                    # ===============================================
                    # 1. NETWORK MAP RENDERER (INTERAKTIF DENGAN PYVIS)
                    # ===============================================
                    with tab_net:
                        if len(G_final_net.nodes()) == 0:
                            st.warning("Jaringan kosong. Silakan kurangi 'Minimum Number of Edges' atau tambah 'Number of Nodes' pada Pengaturan Network.")
                        else:
                            st.caption("Visualisasi interaktif graf berbasis fisika dinamis (Scroll/Drag untuk interaksi).")
                            
                            col_dl1, col_dl2 = st.columns([4, 1])
                            with col_dl2:
                                gexf_str = generate_gexf_string(G_final_net)
                                st.download_button(label="📥 Export to Gephi (.gexf)", data=gexf_str, file_name="network_graph.gexf", mime="application/xml")

                            if HAS_PYVIS:
                                # Rendering interaktif dengan Pyvis
                                net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
                                
                                color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
                                
                                word_avg_years = {}
                                if net_color_year == "Yes" and year_col and 'Year_Numeric' in data.columns:
                                    word_years = defaultdict(list)
                                    for idx_df, row in df_mapped_net.iterrows():
                                        try:
                                            # Robust year lookup
                                            yr_val = data.iloc[idx_df]['Year_Numeric']
                                            if not pd.isna(yr_val):
                                                yr = int(yr_val)
                                                for w in row[net_col]:
                                                    if w in G_final_net.nodes():
                                                        word_years[w].append(yr)
                                        except: pass
                                    word_avg_years = {w: np.mean(yrs) for w, yrs in word_years.items() if yrs}
                                    
                                min_y, max_y = 0, 0
                                if net_color_year == "Yes" and word_avg_years:
                                    min_y = min(word_avg_years.values()) if word_avg_years else 0
                                    max_y = max(word_avg_years.values()) if word_avg_years else 0
                                    cmap = plt.get_cmap('viridis')

                                node_to_comm_net = {}
                                for idx, comm in enumerate(communities_net):
                                    for node in comm: node_to_comm_net[node] = idx

                                for node in G_final_net.nodes():
                                    freq = G_final_net.nodes[node]['freq']
                                    degree_count = G_final_net.degree(node) # Menghitung total keterhubungan edges node ini
                                    n_size = min(max(freq * 1.5, 15), 65)
                                    
                                    # Styling Text Label dengan Outline Putih agar sangat jelas terbaca
                                    font_size = min(max(10 + freq * 0.5, 12), 35)
                                    font_config = {
                                        'size': font_size,
                                        'face': 'Arial',
                                        'strokeWidth': 3,
                                        'strokeColor': 'rgba(255,255,255,0.9)',
                                        'color': '#111111'
                                    }
                                    
                                    if net_color_year == "Yes" and node in word_avg_years:
                                        avg_y = word_avg_years[node]
                                        norm_val = 0.5 if max_y == min_y else (avg_y - min_y) / (max_y - min_y)
                                        hex_color = mcolors.to_hex(cmap(norm_val))
                                        n_title = f"<b>{node}</b><br>Frekuensi: {freq}<br>Keterhubungan (Degree): {degree_count} Node<br>Rata-rata Tahun Publikasi: {round(avg_y, 1)}"
                                    else:
                                        c_idx = node_to_comm_net.get(node, 0)
                                        hex_color = color_palette[c_idx % len(color_palette)]
                                        n_title = f"<b>{node}</b><br>Frekuensi: {freq}<br>Keterhubungan (Degree): {degree_count} Node<br>Klaster Komunitas: {c_idx+1}"
                                        
                                    net.add_node(node, label=node, title=n_title, color=hex_color, size=n_size, font=font_config)
                                
                                for u, v, d in G_final_net.edges(data=True):
                                    edge_weight = d.get('weight', 1)
                                    net.add_edge(u, v, value=edge_weight, title=f"Kekuatan Relasi: {round(edge_weight, 3)}")
                                    
                                net.repulsion(node_distance=150, spring_length=200)
                                
                                try:
                                    # MENGGUNAKAN TEMPFILE UNTUK MENCEGAH RACE CONDITION MULTI-USER
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                                        net.save_graph(tmp_file.name)
                                        with open(tmp_file.name, 'r', encoding='utf-8') as HtmlFile:
                                            source_code = HtmlFile.read()
                                        components.html(source_code, height=670, scrolling=True)
                                except Exception as e:
                                    st.error(f"Gagal merender interaktivitas graf Pyvis: {e}")
                            else:
                                st.warning("Modul Pyvis tidak terinstal. Tampilan grafik Plotly kaku dimatikan. Silakan install Pyvis via terminal untuk fitur ini.")

                    # ===============================================
                    # 2. THEMATIC MAP
                    # ===============================================
                    with tab_map:
                        if df_theme.empty:
                            st.warning("Algoritma tidak dapat mendeduksi tema apa pun. Rasio kata terhadap frekuensi minimum terlalu rendah. Coba kurangi 'Min Cluster Freq' atau tambah 'Number of Words'.")
                        else:
                            color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
                            theme_cluster_words = {} 
                            for comm in communities_theme:
                                comm_sorted = sorted(comm, key=lambda x: wc_theme.get(x, 0), reverse=True)
                                cluster_name = comm_sorted[0].title() 
                                theme_cluster_words[cluster_name] = {w: wc_theme.get(w, 0) for w in comm_sorted}

                            df_theme['Label Bubble'] = df_theme['keywords'].apply(lambda kws: "<br>".join(kws[:theme_num_labels]))
                            df_theme['Anggota Parsial'] = df_theme['keywords'].apply(lambda kws: ", ".join(kws))
                            
                            # Mengubah mean() menjadi median() agar kuadran lebih resisten terhadap outlier ekstrem
                            mid_c = df_theme['Centrality'].median() if not df_theme['Centrality'].empty else 0
                            mid_d = df_theme['Density'].median() if not df_theme['Density'].empty else 0
                            
                            fig_theme = px.scatter(df_theme, x="Centrality", y="Density", size="Volume Representasi", color="Nama Tema Kunci", hover_name="Nama Tema Kunci", hover_data={"Centrality": True, "Density": True, "Volume Representasi": True, "Label Bubble": False, "Nama Tema Kunci": False}, text="Label Bubble", size_max=85, height=700, color_discrete_sequence=color_palette)
                            
                            fig_theme.update_traces(
                                textposition='middle center', 
                                textfont=dict(color='#2c3e50', size=theme_label_size * 40, family="Arial"), 
                                marker=dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.65)
                            )
                            
                            fig_theme.add_hline(y=mid_d, line_dash="dash", line_color="#333", opacity=0.7)
                            fig_theme.add_vline(x=mid_c, line_dash="dash", line_color="#333", opacity=0.7)
                            
                            fig_theme.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", text="Niche Themes<br><i style='font-size:10px'>(Perkembangan Spesifik Ekstrem)</i>", showarrow=False, font=dict(size=14, color="gray"), xanchor="left", yanchor="top", align="left")
                            fig_theme.add_annotation(x=0.99, y=0.99, xref="paper", yref="paper", text="Motor Themes<br><i style='font-size:10px'>(Utama, Mapan, Tumbuh)</i>", showarrow=False, font=dict(size=14, color="gray"), xanchor="right", yanchor="top", align="right")
                            fig_theme.add_annotation(x=0.01, y=0.01, xref="paper", yref="paper", text="Emerging/Declining Themes<br><i style='font-size:10px'>(Baru Muncul / Mati)</i>", showarrow=False, font=dict(size=14, color="gray"), xanchor="left", yanchor="bottom", align="left")
                            fig_theme.add_annotation(x=0.99, y=0.01, xref="paper", yref="paper", text="Basic & Transversal Themes<br><i style='font-size:10px'>(Dasar Keilmuan Menyeluruh)</i>", showarrow=False, font=dict(size=14, color="gray"), xanchor="right", yanchor="bottom", align="right")

                            fig_theme.update_layout(plot_bgcolor='white', xaxis_title="Relevance degree (Kekuatan Hubungan Eksternal / Centrality)", yaxis_title="Development degree (Kepadatan Hubungan Internal / Density)", showlegend=False, xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False), margin=dict(l=20, r=20, t=20, b=20))
                            
                            st.plotly_chart(fig_theme, use_container_width=True, config=PLOTLY_DL_CONFIG)
                            
                            st.markdown("---")
                            st.markdown("#### Detail Analisis Komunitas (Drill-down Klaster)")
                            st.caption("Rincian top distribusi kata yang membentuk masing-masing sentralitas tema di atas.")
                            
                            num_clusters = len(df_theme)
                            cols_per_row = 3
                            rows = math.ceil(num_clusters / cols_per_row)
                            
                            for r in range(rows):
                                c_cols = st.columns(cols_per_row)
                                for c in range(cols_per_row):
                                    idx = r * cols_per_row + c
                                    if idx < num_clusters:
                                        row_data = df_theme.iloc[idx]
                                        c_name = row_data['Nama Tema Kunci']
                                        
                                        top_5_kws = row_data['keywords'][:5]
                                        kws_freq = [{"Kata": k, "Frekuensi": wc_theme.get(k, 0)} for k in top_5_kws]
                                        df_c = pd.DataFrame(kws_freq).sort_values('Frekuensi', ascending=True)
                                        
                                        fig_bar = px.bar(df_c, x='Frekuensi', y='Kata', orientation='h', title=f"Tema: {c_name}")
                                        fig_bar.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250, yaxis_title=None)
                                        fig_bar.update_traces(marker_color=color_palette[idx % len(color_palette)])
                                        c_cols[c].plotly_chart(fig_bar, use_container_width=True)

                    # ===============================================
                    # 3. THEMATIC EVOLUTION (SANKEY DATA FLOW)
                    # ===============================================
                    with tab_evol:
                        st.markdown("#### ⏳ Peta Evolusi Aliran Tema (Thematic Evolution Sankey)")
                        st.caption("Diagram komparasi asimetris untuk memvisualisasikan asal usul peleburan konsep antardua periode spektrum waktu publikasi.")
                        
                        if not year_col or 'Year_Numeric' not in data.columns:
                            st.warning("Metadata Temporal (Tahun Publikasi) tidak ditemukan. Modul evolusi dinonaktifkan.")
                        else:
                            temp_df = data.copy()
                            temp_df = temp_df.dropna(subset=['Year_Numeric'])
                            
                            if not temp_df.empty:
                                min_y = int(temp_df['Year_Numeric'].min())
                                max_y = int(temp_df['Year_Numeric'].max())
                                
                                if min_y < max_y:
                                    cut_year = st.slider("Tentukan Garis Limitasi Periode (Cutting Year):", min_y, max_y, (min_y + max_y) // 2)
                                    
                                    def extract_themes_for_period(df_p):
                                        df_p_mapped = preprocess_keywords(df_p, field=net_col, delimiter=actual_net_delim, is_author=is_author_network)
                                        G_r, wc = build_cooccurrence(df_p_mapped, field=net_col, minfreq=actual_min_freq_theme)
                                        if len(G_r.nodes()) == 0: return []
                                        
                                        tw = [w for w, c in Counter({w: c for w, c in wc.items() if c >= actual_min_freq_theme}).most_common(theme_n_words)]
                                        G_r = G_r.subgraph(tw).copy()
                                        
                                        G_n = normalize_network(G_r, method="Association")
                                        G_f = filter_edges(G_n, min_raw_weight=1, min_norm_weight=0.0)
                                        if len(G_f.nodes()) == 0: return []
                                        
                                        comms = detect_clusters(G_f, method=theme_algo)
                                        
                                        themes_p = []
                                        for comm in comms:
                                            if len(comm) < 2: continue
                                            c_sorted = sorted(comm, key=lambda x: wc.get(x, 0), reverse=True)
                                            themes_p.append({"name": c_sorted[0].title(), "words": set(comm)})
                                        return themes_p

                                    with st.spinner("Mengekstrak tema periode longitudinal..."):
                                        df_p1 = temp_df[temp_df['Year_Numeric'] <= cut_year]
                                        df_p2 = temp_df[temp_df['Year_Numeric'] > cut_year]
                                        
                                        themes_1 = extract_themes_for_period(df_p1)
                                        themes_2 = extract_themes_for_period(df_p2)
                                    
                                    if not themes_1 or not themes_2:
                                        st.warning("Data isolasi sub-periode tidak mencukupi untuk menjalankan algoritma klaster Thematic Evolution.")
                                    else:
                                        labels = [f"{t['name']} (P1)" for t in themes_1] + [f"{t['name']} (P2)" for t in themes_2]
                                        source_idx, target_idx, vals = [], [], []
                                        offset = len(themes_1)
                                        
                                        for i, t1 in enumerate(themes_1):
                                            for j, t2 in enumerate(themes_2):
                                                intersect = t1['words'].intersection(t2['words'])
                                                if len(intersect) > 0:
                                                    source_idx.append(i)
                                                    target_idx.append(offset + j)
                                                    vals.append(len(intersect))
                                        
                                        if len(source_idx) > 0:
                                            fig_sankey = go.Figure(data=[go.Sankey(
                                                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="#4a90e2"),
                                                link=dict(source=source_idx, target=target_idx, value=vals, color="rgba(74, 144, 226, 0.4)")
                                            )])
                                            fig_sankey.update_layout(title_text=f"Evolusi Garis Waktu: Tahun Publikasi ≤{cut_year} ➔ Lebih Besar dari {cut_year}", font_size=12, height=600)
                                            st.plotly_chart(fig_sankey, use_container_width=True, config=PLOTLY_DL_CONFIG)
                                        else:
                                            st.info("Berdasarkan rentang pemotongan waktu ini, tidak ada kata kunci perantara yang menjembatani evolusi antara kedua blok.")
                                else:
                                    st.info("Deviasi Tahun (Standar Deviasi temporal) bernilai konstan (satu titik tahun).")

                    # ===============================================
                    # 4. THREE-FIELDS PLOT (SANKEY)
                    # ===============================================
                    with tab_three:
                        st.markdown("#### 🔀 Three-Fields Plot Dinamis")
                        st.caption("Visualisasi komprehensif 3 entitas untuk melihat relasi persilangan (Misal: Negara ➔ Penulis ➔ Topik).")
                        
                        # DINAMISASI THREE-FIELDS PLOT
                        col_tf1, col_tf2, col_tf3 = st.columns(3)
                        with col_tf1: 
                            field_left = st.selectbox("Field Kiri (Kategori 1):", data.columns, index=data.columns.tolist().index(author_col) if author_col in data.columns else 0)
                        with col_tf2: 
                            field_mid = st.selectbox("Field Tengah (Kategori 2):", data.columns, index=data.columns.tolist().index(net_col) if net_col in data.columns else 1 % len(data.columns))
                        with col_tf3: 
                            field_right = st.selectbox("Field Kanan (Kategori 3):", data.columns, index=data.columns.tolist().index(journal_col) if journal_col in data.columns else 2 % len(data.columns))

                        try:
                            N_TOP = 15 
                            tf_df = data.copy()
                            
                            # Fungsi untuk mengekstrak list items dengan auto-delimiter detection
                            def get_items(series, force_delim=None):
                                if force_delim:
                                    return series.astype(str).str.split(force_delim)
                                else:
                                    sample = series.dropna().astype(str).head(50).str.cat(sep='')
                                    delim = ';' if ';' in sample else (',' if ',' in sample else ('|' if '|' in sample else None))
                                    if delim:
                                        return series.astype(str).str.split(delim)
                                    return series.astype(str).apply(lambda x: [x])

                            list_left = get_items(tf_df[field_left])
                            list_mid = get_items(tf_df[field_mid])
                            list_right = get_items(tf_df[field_right])
                            
                            all_l = [i.strip() for sublist in list_left.dropna() for i in sublist if i.strip() and i.strip().lower() not in COMMON_STOPWORDS and len(i.strip()) > 1]
                            top_l = [x[0] for x in Counter(all_l).most_common(N_TOP)]
                            
                            all_m = [i.strip() for sublist in list_mid.dropna() for i in sublist if i.strip() and i.strip().lower() not in COMMON_STOPWORDS and len(i.strip()) > 1]
                            top_m = [x[0] for x in Counter(all_m).most_common(N_TOP)]
                            
                            all_r = [i.strip() for sublist in list_right.dropna() for i in sublist if i.strip() and i.strip().lower() not in COMMON_STOPWORDS and len(i.strip()) > 1]
                            top_r = [x[0] for x in Counter(all_r).most_common(N_TOP)]
                            
                            labels = top_l + top_m + top_r
                            
                            link_counts_lm = defaultdict(int)
                            link_counts_mr = defaultdict(int)
                            
                            for i in range(len(tf_df)):
                                vals_l = [x.strip() for x in list_left.iloc[i] if x.strip() in top_l] if isinstance(list_left.iloc[i], list) else []
                                vals_m = [x.strip() for x in list_mid.iloc[i] if x.strip() in top_m] if isinstance(list_mid.iloc[i], list) else []
                                vals_r = [x.strip() for x in list_right.iloc[i] if x.strip() in top_r] if isinstance(list_right.iloc[i], list) else []
                                
                                for l in vals_l:
                                    for m in vals_m:
                                        link_counts_lm[(l, m)] += 1
                                
                                for m in vals_m:
                                    for r in vals_r:
                                        link_counts_mr[(m, r)] += 1
                            
                            source_idx = []
                            target_idx = []
                            vals = []
                            
                            for (l, m), v in link_counts_lm.items():
                                source_idx.append(labels.index(l))
                                target_idx.append(labels.index(m))
                                vals.append(v)
                                
                            for (m, r), v in link_counts_mr.items():
                                source_idx.append(labels.index(m))
                                target_idx.append(labels.index(r))
                                vals.append(v)
                                
                            if len(source_idx) > 0:
                                fig_three = go.Figure(data=[go.Sankey(
                                    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="#2ecc71"),
                                    link=dict(source=source_idx, target=target_idx, value=vals, color="rgba(46, 204, 113, 0.4)")
                                )])
                                
                                fig_three.update_layout(title_text=f"{field_left} ➔ {field_mid} ➔ {field_right}", font_size=11, height=750)
                                st.plotly_chart(fig_three, use_container_width=True, config=PLOTLY_DL_CONFIG)
                            else:
                                st.warning("Tidak ditemukan irisan data yang cukup untuk memplot diagram Sankey antar ketiga kolom ini.")
                            
                        except Exception as e:
                            st.error(f"Gagal memproses matriks 3 dimensi: {e}")

                    # ===============================================
                    # 5. TABLES AND NODE CENTRALITY METRICS
                    # ===============================================
                    with tab_table:
                        st.markdown("#### Tabel Agregat Modularity")
                        st.caption("Tabulasi ringkasan klaster hasil kalkulasi Peta Tematik.")
                        if not df_theme.empty:
                            df_metrics = df_theme.drop(columns=["Label Bubble", "keywords"])
                            st.dataframe(df_metrics, use_container_width=True)
                        else:
                            st.info("Matriks nol. Operasi komputasi kosong.")
                            
                        st.markdown("---")
                        st.markdown("#### Matriks Lanjutan (Edge List & Adjacency)")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            try:
                                adj_df = nx.to_pandas_adjacency(G_final_net)
                                st.download_button(
                                    label="📥 Ekspor Adjacency Matrix (.csv)",
                                    data=convert_df_to_csv(adj_df.reset_index()),
                                    file_name="adjacency_matrix.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception: pass
                        with col_dl2:
                            try:
                                edge_df = nx.to_pandas_edgelist(G_final_net)
                                st.download_button(
                                    label="📥 Ekspor Edge List (.csv)",
                                    data=convert_df_to_csv(edge_df),
                                    file_name="edge_list.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception: pass

                        st.markdown("---")
                        st.markdown("#### Tabel Kedudukan Simpul Tunggal (Node Statistics)")
                        st.caption("Peringkat pergerakan, kendali, dan pengaruh masing-masing entitas individu dalam keseluruhan ekosistem (Sangat penting bagi ko-kolaborasi).")
                        try:
                            # Centrality Computations (Network)
                            deg_cent = nx.degree_centrality(G_final_net)
                            with st.spinner("Memuat metrik sentralitas individu..."):
                                bet_cent = nx.betweenness_centrality(G_final_net)
                                clo_cent = nx.closeness_centrality(G_final_net)
                                pr = nx.pagerank(G_final_net, weight='weight')
                                
                                node_metrics = []
                                for node in G_final_net.nodes():
                                    node_metrics.append({
                                        "Entitas (Node)": node,
                                        "Frekuensi Absolut": G_final_net.nodes[node].get('freq', 0),
                                        "Degree (Relasi Langsung)": round(deg_cent[node], 4),
                                        "Betweenness (Kontrol)": round(bet_cent[node], 4),
                                        "Closeness (Akses)": round(clo_cent[node], 4),
                                        "PageRank (Pengaruh Global)": round(pr[node], 4)
                                    })
                                df_nodes = pd.DataFrame(node_metrics).sort_values("PageRank (Pengaruh Global)", ascending=False)
                                st.dataframe(df_nodes, use_container_width=True, height=350)
                                
                                st.download_button(
                                    label="📥 Ekspor Laporan Node (.csv)",
                                    data=convert_df_to_csv(df_nodes),
                                    file_name="individual_node_metrics.csv",
                                    mime="text/csv",
                                )
                        except Exception as e:
                            st.warning("Sub-jaringan tidak kompatibel untuk pemeringkatan matriks terpusat.")
                        
                        gc.collect()

    # ---------------------------------------------------------
    # MENU 6: KOMPARASI GLOBAL VS INDONESIA
    # ---------------------------------------------------------
    elif menu_selection == "🌍 Komparasi Global vs Indonesia":
        st.title("🌍 Komparasi Riset: Global vs Indonesia")
        st.markdown("Fitur analitik khusus untuk menyandingkan performa, dampak sitasi, dan lanskap penelitian antara skala Global dengan Indonesia menggunakan data bibliometrik Anda.")
        
        geo_col_target = affiliation_col if affiliation_col else author_col
        if not geo_col_target:
            st.error("⚠️ Kolom afiliasi (Negara) atau Penulis tidak terdeteksi secara otomatis. Tidak dapat memisahkan asal publikasi.")
        else:
            with st.spinner("Memisahkan korpus data Global dan Indonesia..."):
                # Deteksi Afiliasi Indonesia (Regex sederhana)
                mask_indo = data[geo_col_target].astype(str).str.contains(r'\bIndonesia\b|\bIDN\b|\bID\b', case=False, na=False)
                df_indo = data[mask_indo].copy()
                df_global = data[~mask_indo].copy() # Data non-Indonesia untuk baseline
                
            if df_indo.empty:
                st.warning("⚠️ Tidak ditemukan publikasi berafiliasi 'Indonesia' di dalam dataset ini. Pemetaan komparatif tidak dapat dijalankan.")
            else:
                st.success(f"✅ Klasifikasi Selesai: **{len(df_indo)}** Dokumen Riset Indonesia vs **{len(df_global)}** Dokumen Riset Global.")
                
                tab_metrics, tab_collab, tab_topic, tab_journal, tab_ai_comp = st.tabs([
                    "📈 Pertumbuhan & Dampak", "🤝 Pemetaan Kolaborasi", "🎯 Analisis Topik (Venn)", "🏢 Kualitas Wadah Publikasi", "🧠 AI Strategic Advisor"
                ])
                
                with tab_metrics:
                    st.markdown("#### 1. Perbandingan Volume & Kualitas Dampak (Sitasi)")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    col_m1.metric("🌐 Total Volume (Global)", f"{len(df_global)} Dokumen")
                    col_m2.metric("🇮🇩 Total Volume (Indonesia)", f"{len(df_indo)} Dokumen")
                    
                    avg_cite_indo = 0
                    avg_cite_global = 0
                    if citation_col:
                        df_indo[citation_col] = pd.to_numeric(df_indo[citation_col], errors='coerce').fillna(0)
                        df_global[citation_col] = pd.to_numeric(df_global[citation_col], errors='coerce').fillna(0)
                        
                        avg_cite_indo = df_indo[citation_col].mean()
                        avg_cite_global = df_global[citation_col].mean()
                        
                        col_m3.metric("📈 Rata-Rata Sitasi / Kualitas", 
                                      f"ID: {round(avg_cite_indo, 1)} | GLB: {round(avg_cite_global, 1)}", 
                                      delta=round(avg_cite_indo - avg_cite_global, 1), 
                                      help="Rata-rata jumlah sitasi per dokumen (Field-Weighted Proxy)")
                    
                    st.markdown("---")
                    st.markdown("#### 2. Analisis Tren Pertumbuhan (Base-100 Index Proxy CAGR)")
                    st.caption("Membandingkan laju percepatan pertumbuhan jumlah publikasi, menstandarkan tahun awal masing-masing dengan nilai Indeks 100.")
                    
                    if 'Year_Numeric' in data.columns and not data['Year_Numeric'].isna().all():
                        g_year = df_global['Year_Numeric'].value_counts().sort_index()
                        i_year = df_indo['Year_Numeric'].value_counts().sort_index()
                        
                        if not g_year.empty and not i_year.empty:
                            g_base = g_year.iloc[0]
                            i_base = i_year.iloc[0]
                            
                            df_g_idx = pd.DataFrame({'Tahun': g_year.index, 'Indeks Pertumbuhan': (g_year.values / g_base) * 100, 'Wilayah': 'Global'})
                            df_i_idx = pd.DataFrame({'Tahun': i_year.index, 'Indeks Pertumbuhan': (i_year.values / i_base) * 100, 'Wilayah': 'Indonesia'})
                            
                            df_growth = pd.concat([df_g_idx, df_i_idx])
                            
                            fig_growth = px.line(df_growth, x='Tahun', y='Indeks Pertumbuhan', color='Wilayah', markers=True,
                                                 color_discrete_map={'Global': '#3498db', 'Indonesia': '#e74c3c'})
                            fig_growth.update_layout(xaxis=dict(dtick=1))
                            fig_growth.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Base Index (100)")
                            st.plotly_chart(fig_growth, use_container_width=True)
                    else:
                        st.warning("Data temporal (Tahun Publikasi) tidak mencukupi.")
                    
                   # --- TAB 2: KOLABORASI ---
                with tab_collab:
                    st.markdown("#### 🤝 Pemetaan Kolaborasi Internasional (International Co-authorship Rate)")
                    st.caption("Seberapa terintegrasi periset Indonesia dengan jaringan sains global?")
                    
                    indo_collab_status = []
                    partner_countries = []
                    
                    for text in df_indo[geo_col_target].dropna():
                        countries = extract_countries_from_text(text)
                        non_indo = [c for c in countries if c.lower() not in ['indonesia', 'idn']]
                        if non_indo:
                            indo_collab_status.append('Kolaborasi Internasional')
                            partner_countries.extend(non_indo)
                        else:
                            indo_collab_status.append('Domestik (Hanya Indonesia)')
                            
                    if indo_collab_status:
                        col_c1, col_c2 = st.columns(2)
                        
                        with col_c1:
                            df_collab_ratio = pd.DataFrame({'Status': indo_collab_status})
                            ratio_counts = df_collab_ratio['Status'].value_counts().reset_index()
                            ratio_counts.columns = ['Status Kolaborasi', 'Jumlah']
                            
                            fig_pie = px.pie(ratio_counts, names='Status Kolaborasi', values='Jumlah', hole=0.4,
                                             color='Status Kolaborasi', color_discrete_map={'Domestik (Hanya Indonesia)': '#95a5a6', 'Kolaborasi Internasional': '#2ecc71'})
                            fig_pie.update_traces(textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                        with col_c2:
                            if partner_countries:
                                df_partners = pd.DataFrame(Counter(partner_countries).most_common(5), columns=['Negara Mitra', 'Jumlah Joint-Paper'])
                                fig_bar_partner = px.bar(df_partners, y='Negara Mitra', x='Jumlah Joint-Paper', orientation='h')
                                fig_bar_partner.update_layout(yaxis={'categoryorder':'total ascending'})
                                fig_bar_partner.update_traces(marker_color='#f39c12')
                                st.markdown("**Top 5 Negara Mitra Kolaborasi Indonesia**")
                                st.plotly_chart(fig_bar_partner, use_container_width=True)
                            else:
                                st.info("Tidak ada data negara mitra kolaborasi yang terdeteksi.")

                # --- TAB 3: IRISAN TOPIK (VENN) ---
                with tab_topic:
                    st.markdown("#### 🎯 Analisis Irisan Topik (Topic Overlap & Jaccard Similarity)")
                    st.caption("Mendeteksi selaraskah fokus riset nasional dengan tren global, menggunakan Jaccard Index (Set Theory).")
                    
                    kw_candidates = [c for c in data.columns if 'keyword' in c.lower() or 'authkey' in c.lower() or 'index' in c.lower()]
                    kw_col = kw_candidates[0] if kw_candidates else title_col
                    
                    if kw_col:
                        def get_top_kws(df, col, n=10):
                            kws = df[col].astype(str).str.lower().str.split(';').explode().str.strip()
                            kws = kws[~kws.isin(COMMON_STOPWORDS) & (kws.str.len() > 3)]
                            return kws.value_counts().head(n)

                        top_100_g = set(get_top_kws(df_global, kw_col, 100).index)
                        top_100_i = set(get_top_kws(df_indo, kw_col, 100).index)
                        
                        intersection = top_100_g.intersection(top_100_i)
                        union = top_100_g.union(top_100_i)
                        jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                        
                        st.metric("🧬 Jaccard Similarity Index (Kesamaan Topik)", f"{round(jaccard * 100, 2)}%", help="100% berarti fokus riset sama persis. 0% berarti sangat terisolasi/berbeda sama sekali.")
                        
                        col_v1, col_v2 = st.columns([1, 1])
                        with col_v1:
                            if HAS_VENN:
                                fig_venn, ax_venn = plt.subplots(figsize=(6, 4))
                                v = venn2([top_100_g, top_100_i], set_labels=('Top 100 Global', 'Top 100 Indonesia'), ax=ax_venn)
                                if v.get_patch_by_id('10'): v.get_patch_by_id('10').set_color('#3498db')
                                if v.get_patch_by_id('01'): v.get_patch_by_id('01').set_color('#e74c3c')
                                if v.get_patch_by_id('11'): v.get_patch_by_id('11').set_color('#9b59b6')
                                st.pyplot(fig_venn)
                            else:
                                st.warning("Modul `matplotlib-venn` tidak terinstal. Tampilan grafik Venn dilewati.")
                                st.info(f"Topik Irisan: {len(intersection)} | Murni Global: {len(top_100_g - intersection)} | Murni Indo: {len(top_100_i - intersection)}")
                        
                        with col_v2:
                            st.markdown("**🔍 Topik Frontier (Hanya Diteliti Global):**")
                            st.caption(", ".join(list(top_100_g - intersection)[:15]) + "...")
                            st.markdown("**🌴 Topik Endemik (Hanya Diteliti Indonesia):**")
                            st.caption(", ".join(list(top_100_i - intersection)[:15]) + "...")
                            st.markdown("**🤝 Topik Selaras (Shared Overlap):**")
                            st.caption(", ".join(list(intersection)[:15]) + "...")

                # --- TAB 4: KUALITAS WADAH (JURNAL) ---
                with tab_journal:
                    st.markdown("#### 🏢 Pemetaan Kualitas Wadah Publikasi (Journal Distribution)")
                    if journal_col:
                        col_j1, col_j2 = st.columns(2)
                        with col_j1:
                            st.markdown("**🌐 Top 5 Jurnal Publikasi Global**")
                            top_j_g = df_global[journal_col].value_counts().head(5).reset_index()
                            top_j_g.columns = ['Jurnal Target', 'Total']
                            st.dataframe(top_j_g, use_container_width=True)
                        with col_j2:
                            st.markdown("**🇮🇩 Top 5 Jurnal Publikasi Indonesia**")
                            top_j_i = df_indo[journal_col].value_counts().head(5).reset_index()
                            top_j_i.columns = ['Jurnal Target', 'Total']
                            st.dataframe(top_j_i, use_container_width=True)
                    else:
                        st.warning("Metadata nama Jurnal/Sumber tidak tersedia.")

                # --- TAB 5: AI ADVISOR ---
                with tab_ai_comp:
                    st.markdown("#### 🧠 Syntesis Strategis AI (Gap Analysis & Policy Recommendation)")
                    st.info("AI akan bertindak sebagai **Strategic Policy Advisor**, mengevaluasi kesenjangan topik riset, kolaborasi, dan memberi rekomendasi pendanaan/kebijakan untuk mengejar megatrend global.")
                    
                    if st.button("🚀 Rumuskan Laporan Kebijakan AI", type="primary"):
                        if not AI_API_KEY:
                            st.error(f"⚠️ {AI_PROVIDER} API Key belum disetel. Harap atur di menu Setelan Sidebar.")
                        else:
                            with st.spinner(f"Menyusun evaluasi strategis menggunakan {AI_MODEL}..."):
                                # Ekstraksi Abstrak (Teknologi)
                                tech_g, tech_i = "N/A", "N/A"
                                if abstract_col:
                                    tech_g = ", ".join(get_top_kws(df_global, abstract_col, 20).index.tolist())
                                    tech_i = ", ".join(get_top_kws(df_indo, abstract_col, 20).index.tolist())

                                top_g_str = ", ".join(list(top_100_g)[:15]) if kw_col else "N/A"
                                top_i_str = ", ".join(list(top_100_i)[:15]) if kw_col else "N/A"
                                overlap_str = str(round(jaccard * 100, 2)) + "%" if kw_col else "N/A"
                                
                                collab_rate = round((ratio_counts[ratio_counts['Status Kolaborasi'] == 'Kolaborasi Internasional']['Jumlah'].values[0] / len(df_indo)) * 100, 2) if 'Kolaborasi Internasional' in ratio_counts['Status Kolaborasi'].values else 0
                                
                                prompt_sys = """Anda adalah 'Strategic Research Policy Advisor' & Pakar Bibliometrik level Kementerian/CSIRO. 
Buat Laporan Evaluasi Kebijakan Riset komprehensif yang membandingkan performa riset Indonesia dengan lanskap Global.

Gunakan Format Analisis ini:
1. **Analisis Produktivitas & Dampak (Impact)**: Evaluasi perbedaan rata-rata sitasi dan indikasi kualitas wadah publikasi.
2. **Kesehatan Kolaborasi Internasional**: Berdasarkan persentase kolaborasi Indonesia, apakah riset kita cukup terintegrasi dengan jaringan global?
3. **Kesenjangan Topik & Adopsi Teknologi (Gap Analysis)**: Berdasarkan data irisan topik dan terminologi ABSTRAK, sebutkan teknologi apa yang 'lagging' (hanya ada di Global) dan apa yang jadi fokus 'endemik' Indonesia.
4. **Rekomendasi Kebijakan Strategis**: Berikan 3 poin rekomendasi konkret (misal: strategi pendanaan, prioritas riset) untuk mengejar ketertinggalan megatrend global.

Gunakan Bahasa Indonesia formal akademis tinggi. Analisis HANYA berdasarkan data konkret di bawah ini."""

                                prompt_user = f"""
[DATA METRIK KOMPARATIF]
- Volume: Indonesia ({len(df_indo)} dok), Global ({len(df_global)} dok)
- Sitasi Rata-rata per Dokumen: Indonesia ({round(avg_cite_indo,2)}), Global ({round(avg_cite_global,2)})
- Tingkat Kolaborasi Internasional Peneliti Indonesia: {collab_rate}%
- Kesamaan Fokus Riset (Jaccard Topic Overlap): {overlap_str}
- Topik/Keyword Riset Indonesia: {top_i_str}
- Topik/Keyword Riset Global: {top_g_str}
- Tren Terminologi Teknologi di Abstrak Indonesia: {tech_i}
- Tren Terminologi Teknologi di Abstrak Global: {tech_g}

Buat laporan analisis strategis sekarang!"""
                                
                                response_placeholder = st.empty()
                                full_resp = ""
                                
                                if AI_PROVIDER == "Mistral":
                                    for chunk in stream_mistral(prompt_sys, prompt_user, AI_API_KEY, AI_MODEL):
                                        if "❌ Error" in chunk: full_resp = chunk; break
                                        full_resp += chunk
                                        response_placeholder.markdown(full_resp + "▌")
                                elif AI_PROVIDER == "Google Gemini":
                                    for chunk in stream_gemini(prompt_sys, prompt_user, AI_API_KEY, AI_MODEL):
                                        if "❌ Error" in chunk: full_resp = chunk; break
                                        full_resp += chunk
                                        response_placeholder.markdown(full_resp + "▌")
                                elif AI_PROVIDER == "Groq":
                                    for chunk in stream_groq(prompt_sys, prompt_user, AI_API_KEY, AI_MODEL):
                                        if "❌ Error" in chunk: full_resp = chunk; break
                                        full_resp += chunk
                                        response_placeholder.markdown(full_resp + "▌")
                                
                                response_placeholder.markdown(full_resp)
                                st.download_button("📥 Unduh Policy Brief (.txt)", full_resp, file_name="AI_Policy_Brief_Indo_vs_Global.txt")

    # ---------------------------------------------------------
    # MENU 7: AI CHATBOT (STRICT RAG & MULTI-TEMPLATES)
    # ---------------------------------------------------------
    elif menu_selection == "💬 AI Chatbot (RAG)":
        st.title("💬 Interactive AI Research Assistant (Strict RAG)")
        st.markdown("Asisten AI yang dirancang HANYA menjawab berdasarkan data. Jika pertanyaan Anda tidak ada di dalam jurnal, AI diinstruksikan untuk jujur menolak menjawab untuk mencegah halusinasi data.")
        
        target_rag_col = abstract_col if abstract_col else title_col
        
        if not target_rag_col:
            st.error("⚠️ Kolom teks naratif (Abstrak/Judul) tidak ditemukan di dataset untuk diriset.")
        elif not HAS_SKLEARN:
            st.error("💡 Modul Machine Learning `scikit-learn` mutlak diperlukan untuk pencarian Semantic RAG. Jalankan `pip install scikit-learn`.")
        else:
            with st.spinner("Menyiapkan Korpus Pencarian Global..."):
                # Buat kolom Search_Text virtual
                df_search = data.copy()
                df_search['Search_Text'] = ""
                if title_col: df_search['Search_Text'] += df_search[title_col].fillna("").astype(str) + " "
                if abstract_col: df_search['Search_Text'] += df_search[abstract_col].fillna("").astype(str)
                
                corpus_texts = df_search['Search_Text'].tolist()
                titles_list = data[title_col].fillna("Tanpa Judul").astype(str).tolist() if title_col else ["Tanpa Judul"] * len(corpus_texts)
                
                # Gunakan Year_Numeric untuk tahun yang valid
                if 'Year_Numeric' in data.columns:
                    years_list = data['Year_Numeric'].fillna("N/A").astype(str).tolist()
                else:
                    years_list = data[year_col].fillna("N/A").astype(str).tolist() if year_col else ["N/A"] * len(corpus_texts)
                
                # --- PERBAIKAN: PROFIL GLOBAL DIPERKAYA DENGAN DATA 'OVERVIEW & TRENDS' ---
                total_rows = len(data)
                col_names = ", ".join(data.columns)
                
                # Ekstrak rentang tahun
                if 'Year_Numeric' in data.columns:
                    valid_years = data['Year_Numeric'].dropna()
                    year_info = f"{int(valid_years.min())} hingga {int(valid_years.max())}" if not valid_years.empty else "Tidak diketahui"
                else:
                    year_info = "Tidak tersedia"
                
                # Ekstrak Top Penulis
                if author_col:
                    all_authors = data[author_col].dropna().astype(str).str.split(";").explode().str.strip()
                    top_authors = ", ".join(all_authors[all_authors != ""].value_counts().head(5).index.tolist())
                else:
                    top_authors = "Tidak tersedia"
                    
                # Ekstrak Top Jurnal
                if journal_col:
                    top_journals = ", ".join(data[journal_col].value_counts().head(5).index.tolist())
                else:
                    top_journals = "Tidak tersedia"
                    
                # Ekstrak Top Sitasi
                if citation_col and title_col:
                    temp_data = data.copy()
                    temp_data[citation_col] = pd.to_numeric(temp_data[citation_col], errors='coerce').fillna(0).astype(int)
                    top_cited_df = temp_data.sort_values(by=citation_col, ascending=False).head(5)
                    top_cited_papers = "\n                ".join([f"- '{row[title_col]}' ({row[citation_col]} sitasi)" for idx, row in top_cited_df.iterrows()])
                else:
                    top_cited_papers = "Tidak tersedia"

                global_data_profile = f"""
                [PROFIL GLOBAL DATASET & STATISTIK]
                - Total Dokumen/Jurnal di database ini: {total_rows} dokumen.
                - Rentang Tahun Publikasi: {year_info}
                - 5 Penulis/Kreator Paling Produktif: {top_authors}
                - 5 Kategori/Penerbit Teratas: {top_journals}
                - 5 Dokumen dengan Sitasi Tertinggi (High-Impact):
                {top_cited_papers}
                """

            with st.expander("⚙️ Pengaturan Sensitivitas Pencarian & Memori", expanded=False):
                top_k = st.slider("Jumlah Maksimal Jurnal Referensi untuk ditarik (Top K):", 3, 20, 10)
                
                col_clear, col_export = st.columns(2)
                with col_clear:
                    if st.button("🗑️ Bersihkan Riwayat Percakapan Chatbot", use_container_width=True):
                        st.session_state.chat_messages = []
                        st.rerun()
                with col_export:
                    if st.session_state.chat_messages:
                        chat_export = "\n\n".join([f"[{msg['role'].upper()}]\n{msg['content']}" for msg in st.session_state.chat_messages])
                        st.download_button("📥 Ekspor Transkrip Diskusi (.txt)", data=chat_export, file_name="riwayat_chat_ai.txt", mime="text/plain", use_container_width=True)
                    else:
                        st.button("📥 Ekspor Transkrip Diskusi (.txt)", disabled=True, use_container_width=True)

            st.markdown("---")

            # Tampilkan riwayat chat di UI
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            st.markdown("---")
            
            # --- SISTEM PILIHAN GANDA (TEMPLATE PERTANYAAN) ---
            with st.container(border=True):
                st.markdown("#### 🎯 Ajukan Pertanyaan ke AI")
                opsi_template = [
                    # Ditambahkan dari Dashboard Overview & Trends
                    "📈 Analisis tren produksi dokumen dari tahun ke tahun berdasarkan dataset ini.",
                    "✍️ Siapa saja penulis paling produktif dan apa fokus utama penelitian mereka?",
                    "🏆 Apa saja dokumen atau artikel dengan sitasi tertinggi (high-impact) dan apa kontribusi utamanya?",
                    "📰 Jurnal atau sumber penerbit apa yang paling mendominasi publikasi di bidang ini?",
                    # Template Analitik
                    "📚 Ringkaskan tren dan temuan utama dari literatur di dataset ini.",
                    "🔬 Apa saja metodologi, pendekatan, atau algoritma yang dominan digunakan?",
                    "💡 Identifikasi celah penelitian (research gap) yang paling potensial dieksplorasi di masa depan.",
                    "⏳ Bagaimana evolusi atau pergeseran fokus topik penelitian ini dari waktu ke waktu?",
                    "🧩 Teori, model, atau kerangka konseptual apa saja yang sering menjadi landasan studi?",
                    "⚠️ Apa saja batasan penelitian (limitations) yang paling sering disebutkan oleh para penulis?",
                    "💼 Apa saja implikasi praktis atau rekomendasi yang disarankan dari temuan-temuan ini?",
                    # Template Khusus WIPO
                    "⚙️ [WIPO] Apa saja teknologi atau paten utama (berdasarkan klasifikasi IPC) yang paling banyak muncul?",
                    "🏢 [WIPO] Siapa saja perusahaan atau inventor (Applicant/Inventor) paling dominan dalam dataset paten ini?",
                    "🛠️ [WIPO] Berdasarkan abstrak paten, apa masalah teknis utama yang berusaha diselesaikan oleh penemuan-penemuan ini?"
                ]
                
                pilihan_user = st.radio("Pilih panduan pertanyaan (Pilihan Ganda):", opsi_template)
                kirim_btn = st.button("🚀 Tanya AI", type="primary", use_container_width=True)

            # Input pengguna (Trigger dari tombol)
            if kirim_btn:
                user_chat_input = pilihan_user

                if not AI_API_KEY:
                    st.error(f"⚠️ Operasi diinterupsi. Masukkan {AI_PROVIDER} API Key pada prapengaturan (Settings Sidebar).")
                else:
                    # Tambahkan ke riwayat tampilan
                    st.session_state.chat_messages.append({"role": "user", "content": user_chat_input})
                    with st.chat_message("user"):
                        st.markdown(user_chat_input)

                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_chat_response = ""
                        
                        with st.spinner("🔍 Mesin TF-IDF sedang memindai seluruh dokumen..."):
                            # Hapus stop_words='english' agar tidak terlalu agresif memotong istilah penting
                            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                            tfidf_matrix = vectorizer.fit_transform(corpus_texts)
                            
                            # --- MAPPING TEMPLATE KE KATA KUNCI INGGRIS ---
                            template_keywords = {
                                opsi_template[0]: "trend year annual production publication volume increase decrease",
                                opsi_template[1]: "author researcher productive contribute leading focus study",
                                opsi_template[2]: "citation cited impact highly influential paper document contribution",
                                opsi_template[3]: "journal publisher source publication outlet conference proceeding",
                                opsi_template[4]: "trend review overview development main finding conclusion",
                                opsi_template[5]: "methodology approach algorithm method technique framework model",
                                opsi_template[6]: "future work research gap limitation challenge open issue",
                                opsi_template[7]: "evolution history past shift focus review decade trend",
                                opsi_template[8]: "theory conceptual model framework foundation literature theoretical",
                                opsi_template[9]: "limitation restrict downside future work scope boundaries constraint",
                                opsi_template[10]: "practical implication industry policy practice application real world",
                                opsi_template[11]: "ipc patent technology classification field code invention",
                                opsi_template[12]: "applicant inventor company assignee patent dominate top owner",
                                opsi_template[13]: "problem technical solve solution overcome invention patent abstract method"
                            }
                            
                            search_query = template_keywords[pilihan_user]
                            active_threshold = 0.0  # Selalu bypass threshold untuk template agar mendapat konteks sampel
                            
                            query_vec = vectorizer.transform([search_query])
                            
                            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                            top_indices = similarities.argsort()[-top_k:][::-1]
                            
                            retrieved_contexts = []
                            debug_retrieved_info = [] # Untuk transparansi di UI
                            
                            for idx in top_indices:
                                if similarities[idx] >= active_threshold:
                                    # Batasi max 2000 karakter agar tidak Token Limit Error
                                    text_lengkap = corpus_texts[idx].strip()
                                    text_aman = text_lengkap[:2000] + "...(dipotong)" if len(text_lengkap) > 2000 else text_lengkap
                                    
                                    ref_item = f"-[Judul Jurnal: {titles_list[idx]}]\n-[Tahun: {years_list[idx]}]\n-[Isi Teks]: {text_aman}\n\n"
                                    retrieved_contexts.append(ref_item)
                                    debug_retrieved_info.append(f"**{titles_list[idx]}** (Skor Kemiripan TF-IDF: {similarities[idx]:.3f})")

                            context_string = "".join(retrieved_contexts)
                            
                        # --- UI TRANSPARANSI ---
                        if debug_retrieved_info:
                            with st.expander("🔍 Lihat Referensi Jurnal yang Ditemukan Sistem untuk Menjawab Ini:", expanded=False):
                                st.write("Sistem memberikan jurnal-jurnal ini ke AI sebagai bahan bacaan:")
                                for info in debug_retrieved_info:
                                    st.markdown(f"- {info}")
                        else:
                            st.warning("⚠️ Sistem tidak menemukan kata kunci yang cocok di dalam abstrak/judul jurnal Anda.")
                            context_string = "KOSONG. TIDAK ADA JURNAL YANG RELEVAN."

                        # --- MEMORI CHAT YANG SINGKAT ---
                        chat_history_str = ""
                        if len(st.session_state.chat_messages) > 1:
                            chat_history_str = "[RIWAYAT PERCAKAPAN KITA SEBELUMNYA]\n"
                            recent_chats = st.session_state.chat_messages[-5:-1] 
                            for msg in recent_chats:
                                role_name = "User" if msg["role"] == "user" else "AI"
                                chat_history_str += f"{role_name}: {msg['content']}\n"

                        # --- PROMPT STRICT ANTI-HALUSINASI ---
                        with st.spinner("🤖 AI sedang membaca referensi & menulis jawaban..."):
                            rag_system_prompt = f"""Anda adalah 'BiblioBot', Asisten Riset Akademik yang bertugas menjawab pertanyaan user berdasarkan dataset yang diunggah.
                            
{global_data_profile}

{chat_history_str}

[CUPLIKAN JURNAL HASIL PENCARIAN UNTUK PERTANYAAN TERAKHIR USER]
{context_string}

ATURAN MUTLAK (ANTI-HALUSINASI):
1. Anda HANYA boleh menjawab menggunakan informasi yang ada di dalam [CUPLIKAN JURNAL HASIL PENCARIAN] dan [PROFIL GLOBAL DATASET].
2. Jika bagian [CUPLIKAN JURNAL] berisi kata "KOSONG" atau informasi yang ditanyakan TIDAK ADA di dalam cuplikan, ANDA WAJIB MENJAWAB: "Maaf, berdasarkan hasil pemindaian sistem ke dalam teks dataset, saya tidak menemukan informasi mengenai hal tersebut."
3. DILARANG KERAS MENGARANG JAWABAN (HALUSINASI) MENGGUNAKAN PENGETAHUAN LUAR ANDA.
4. Jika Anda menemukan jawabannya di dalam referensi, selalu kutip dengan menyebutkan Judul Jurnalnya.
5. Gunakan bahasa Indonesia yang baik, terstruktur (gunakan bullet points jika perlu), dan santai layaknya asisten profesional.
"""
                            # PERUBAHAN GROQ CALL
                            if AI_PROVIDER == "Mistral":
                                for chunk in stream_mistral(rag_system_prompt, user_chat_input, AI_API_KEY, AI_MODEL):
                                    if "❌ Error" in chunk:
                                        full_chat_response = chunk
                                        break
                                    full_chat_response += chunk
                                    response_placeholder.markdown(full_chat_response + "▌")
                            elif AI_PROVIDER == "Google Gemini":
                                for chunk in stream_gemini(rag_system_prompt, user_chat_input, AI_API_KEY, AI_MODEL):
                                    if "❌ Error" in chunk:
                                        full_chat_response = chunk
                                        break
                                    full_chat_response += chunk
                                    response_placeholder.markdown(full_chat_response + "▌")
                            elif AI_PROVIDER == "Groq":
                                for chunk in stream_groq(rag_system_prompt, user_chat_input, AI_API_KEY, AI_MODEL):
                                    if "❌ Error" in chunk:
                                        full_chat_response = chunk
                                        break
                                    full_chat_response += chunk
                                    response_placeholder.markdown(full_chat_response + "▌")
                                    
                        response_placeholder.markdown(full_chat_response)
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_chat_response})

# ==============================
# FALLBACK: BELUM ADA DATA
# ==============================
else:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.info("👈 Selamat Datang di **Biblio Analyzer Pro**. Silakan gunakan panel navigasi di bilah sebelah kiri untuk memasukkan Kredensial Akses Anda dan mengunggah/mengimpor set Data Riset (.csv/.json).")
    st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'><i>Engineered and enhanced by LLM Architecture. Built for modern enterprise-scale bibliometric studies.</i></p>", unsafe_allow_html=True)
