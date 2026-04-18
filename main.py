"""
=============================================================================
SCIENTIFIC BIBLIOMETRIC AI ANALYZER (ENTERPRISE MASTER EDITION)
=============================================================================
Sistem Perangkat Lunak Skala Penuh untuk Akuisisi, Pembersihan, Analisis, 
dan Pemetaan Sains (Science Mapping) Berbasis Data Bibliometrik.

Versi Enterprise ini dilengkapi dengan:
- Natural Language Processing (NLP)
- Machine Learning Topic Modeling (LDA)
- Semantic Information Retrieval (TF-IDF Cosine Similarity)
- Geo-spatial Choropleth Mapping
- Advanced Graph Topology (NetworkX)
- Generative AI Integration (Mistral & Google Gemini)
- Full Biblioshiny Replication (Three-Fields, Thematic, Trend Topics)
=============================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import re
import unicodedata
import zlib
import json
import math
import logging
import datetime
from collections import Counter, defaultdict
import io
import xml.etree.ElementTree as ET # Modul GEXF Export

# Konfigurasi Logging Dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# DEPENDENSI & EKSTENSI LANJUTAN
# ==============================
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    logging.warning("Modul WordCloud tidak terdeteksi.")

try:
    import networkx as nx
    from networkx.algorithms import community as nx_comm # FIX: Eksplisit import community networkx
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

COUNTRY_ISO_MAPPING = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO", 
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", 
    "Austria": "AUT", "Azerbaijan": "AZE", "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", 
    "Brazil": "BRA", "Canada": "CAN", "China": "CHN", "France": "FRA", "Germany": "DEU", 
    "India": "IND", "Indonesia": "IDN", "Italy": "ITA", "Japan": "JPN", "Malaysia": "MYS",
    "Netherlands": "NLD", "Russia": "RUS", "South Korea": "KOR", "Spain": "ESP", 
    "United Kingdom": "GBR", "UK": "GBR", "England": "GBR", "United States": "USA", "USA": "USA", 
    "Vietnam": "VNM", "Taiwan": "TWN"
}

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

@st.cache_data(ttl=3600, show_spinner=False)
def call_mistral_sync(system_prompt: str, user_prompt: str, api_key: str, model: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=None)
        if response.status_code == 200: return response.json()["choices"][0]["message"]["content"]
        else: return f"Error: {response.text}"
    except Exception as e: return f"Error: {e}"

@st.cache_data(ttl=3600, show_spinner=False)
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

# ==============================
# DATA WRANGLING & EXTRACTORS
# ==============================
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
    text_lower = str(text).lower()
    detected = []
    for country in COUNTRY_ISO_MAPPING.keys():
        if re.search(rf'\b{country.lower()}\b', text_lower):
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
        if pd.isna(x): return []
        words = [k.strip().title() if is_author else k.strip().lower() for k in str(x).split(delimiter)]
        return [w for w in words if w and w.lower() not in COMMON_STOPWORDS and w.lower() not in ["tidak tersedia", "n/a", "no title"]]
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
    # FIX: Memanfaatkan nx_comm agar tidak memicu AttributeError
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
    st.session_state.history = st.session_state.history[:st.session_state.current_step + 1]
    st.session_state.history_actions = st.session_state.history_actions[:st.session_state.current_step + 1]
    st.session_state.history.append(new_df)
    st.session_state.history_actions.append(action_name)
    st.session_state.current_step += 1
    st.session_state.preview_action = action_name
    st.rerun()

# ==============================
# STRUKTUR ANTARMUKA (SIDEBAR)
# ==============================
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
            "💬 AI Chatbot (RAG)",
            "📖 Library & Glossary"
        ]
    
    menu_selection = st.radio("MAIN MENU NAVIGATION", menu_options, label_visibility="visible")
    
    st.markdown("---")
    with st.expander("⚙️ Settings & API Keys", expanded=False):
        SCOPUS_API_KEY = st.text_input("Scopus API Key", type="password", placeholder="Masukkan key Scopus").strip()
        st.markdown("**AI Provider**")
        AI_PROVIDER = st.selectbox("Pilih Penyedia:", ["Mistral", "Google Gemini"], label_visibility="collapsed")
        
        if AI_PROVIDER == "Mistral":
            AI_API_KEY = st.text_input("Mistral API Key", type="password", placeholder="Masukkan key Mistral").strip()
            AI_MODEL = st.selectbox("Model:", ["mistral-small-latest", "open-mistral-nemo", "mistral-large-latest"])
            st.caption("✨ 'mistral-small' = Cepat & Hemat Token.")
        elif AI_PROVIDER == "Google Gemini":
            AI_API_KEY = st.text_input("Gemini API Key", type="password", placeholder="Masukkan key Gemini").strip()
            AI_MODEL = st.selectbox("Model:", ["gemini-2.5-flash", "gemini-2.5-pro"])
            st.caption("✨ 'gemini-2.5-flash' = Sangat kencang.")

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
                    url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={max_results}&view=COMPLETE&apiKey={SCOPUS_API_KEY}"
                    resp = requests.get(url, headers={'Accept': 'application/json'})
                    
                    if resp.status_code in [401, 403]:
                        st.toast("⚠️ Akses institusi Scopus tidak terdeteksi. Menggunakan mode Fallback (Tanpa Abstrak)...", icon="🔄")
                        url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={max_results}&apiKey={SCOPUS_API_KEY}"
                        resp = requests.get(url, headers={'Accept': 'application/json'})

                    if resp.status_code == 200:
                        loaded_data = clean_scopus_data(resp.json())
                        st.session_state.history = [loaded_data]
                        st.session_state.history_actions = ["Data Awal (Scopus API)"]
                        st.session_state.current_step = 0
                        st.session_state.preview_action = None
                        
                        st.session_state.map_rendered = False
                        
                        st.success(f"✅ Berhasil menarik {len(loaded_data)} dokumen! Silakan pilih menu lain di Sidebar untuk mulai menganalisis.")
                    else: 
                        st.error(f"❌ Terjadi Error API {resp.status_code}: {resp.text}")

    with col2:
        st.markdown("#### 📁 Opsi 2: Unggah File Lokal (Local Data)")
        st.info("Bagi pengguna database lain (Web of Science, PubMed, dll), Anda bisa mengunggah format .csv atau .json ke sini.")
        uploaded_file = st.file_uploader("Pilih Berkas", type=["csv", "json"])
        
        if uploaded_file:
            if st.button("Proses File Unggahan", type="primary", use_container_width=True):
                try:
                    loaded_data = pd.read_json(uploaded_file) if uploaded_file.name.endswith(".json") else pd.read_csv(uploaded_file, encoding='utf-8')
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

    # ====== DETEKSI NAMA KOLOM DINAMIS ======
    year_col = next((col for col in ['Tahun', 'Year', 'year', 'PY', 'Publication Year'] if col in data.columns), None)
    journal_col = next((col for col in ['Jurnal', 'Source title', 'Journal', 'SO'] if col in data.columns), None)
    author_col = next((col for col in ['Penulis', 'Authors', 'Author', 'AU'] if col in data.columns), None)
    title_col = next((col for col in ['Judul', 'Title', 'title', 'Document Title', 'TI'] if col in data.columns), None)
    citation_col = next((col for col in ['Citasi', 'Cited by', 'citedby-count', 'TC'] if col in data.columns), None)
    abstract_col = next((col for col in ['Abstract', 'abstract', 'Description', 'AB'] if col in data.columns), None)
    affiliation_col = next((col for col in ['Negara Afiliasi', 'Affiliation', 'Affiliations', 'C1'] if col in data.columns), None)

    # ---------------------------------------------------------
    # MENU 2: OVERVIEW & TRENDS
    # ---------------------------------------------------------
    if menu_selection == "📊 Overview & Trends":
        st.title("📊 Dataset Overview")
        st.markdown("Profil deskriptif data eksploratori yang menyajikan informasi statistik kunci.")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📄 Total Dokumen", len(data))
        m2.metric("📋 Total Metadata (Kolom)", len(data.columns))
        
        if year_col:
            data['Year_Numeric'] = pd.to_numeric(data[year_col], errors='coerce')
            min_y = int(data['Year_Numeric'].min()) if not pd.isna(data['Year_Numeric'].min()) else "N/A"
            max_y = int(data['Year_Numeric'].max()) if not pd.isna(data['Year_Numeric'].max()) else "N/A"
            m3.metric("📅 Rentang Tahun Publikasi", f"{min_y} - {max_y}")
            data = data.drop(columns=['Year_Numeric'])
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
        
        tab_stats1, tab_stats2, tab_stats3, tab_stats4 = st.tabs(["📊 Produksi & Sitasi", "🏢 Hukum Kinerja (Lotka & Bradford)", "🗺️ Peta Produksi Negara", "🧠 Topic Modeling (LDA)"])
        
        # SUBTAB: PRODUKSI
        with tab_stats1:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if year_col:
                    st.markdown("**1. Produksi Dokumen Per Tahun**")
                    st.area_chart(data[year_col].value_counts().sort_index(), color="#4a90e2") 
                
                if citation_col and title_col:
                    st.markdown("**2. Top 5 Dokumen Paling Banyak Dikutip (High-Impact Papers)**")
                    data[citation_col] = pd.to_numeric(data[citation_col], errors='coerce').fillna(0).astype(int)
                    top_cited = data.sort_values(by=citation_col, ascending=False).head(5)
                    st.dataframe(top_cited[[title_col, citation_col]].reset_index(drop=True), use_container_width=True)

            with col_c2:
                if author_col:
                    st.markdown("**3. Penulis Paling Produktif (Berdasarkan Frekuensi Kemunculan)**")
                    all_authors = data[author_col].dropna().astype(str).str.split(";").explode().str.strip()
                    st.bar_chart(all_authors[all_authors != ""].value_counts().head(10), color="#e74c3c")
                    
                if journal_col:
                    st.markdown("**4. Sumber / Jurnal Penerbit Teratas**")
                    st.bar_chart(data[journal_col].value_counts().head(10), color="#2ecc71")
                    
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
                            # Plot distribusi empiris
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
                            
                            # FIX: Dinamis min_df agar vocabulary tidak kosong jika dokumennya minim
                            min_df_val = 2 if len(docs_lda) > 5 else 1
                            
                            try:
                                tf_vectorizer = CountVectorizer(max_df=0.95, min_df=min_df_val, stop_words=custom_stop)
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
                st.markdown("**✂️ A. Pecah Sel Multi-Nilai (Split Multi-valued Cells)**")
                st.caption("Penting dilakukan untuk kolom 'Penulis' atau 'Author Keywords' agar setiap entitas dipisah menjadi barisnya sendiri untuk kalkulasi pemetaan jaringan yang akurat.")
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
                st.markdown("**✨ B. Transformasi Teks Instan**")
                col_case1, col_case2, col_case3 = st.columns(3)
                if col_case1.button("UPPERCASE", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).str.upper(), f"To UPPERCASE '{target_clean_col}'", target_col=target_clean_col)
                if col_case2.button("lowercase", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).str.lower(), f"To lowercase '{target_clean_col}'", target_col=target_clean_col)
                if col_case3.button("Title Case", use_container_width=True):
                    apply_transform(lambda col: col.astype(str).str.title(), f"To Title Case '{target_clean_col}'", target_col=target_clean_col)

                col_trim, col_blank = st.columns(2)
                with col_trim:
                    if st.button("✂️ Trim Extra Whitespace", use_container_width=True):
                        apply_transform(lambda col: col.astype(str).apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x), f"Trim Spasi '{target_clean_col}'", target_col=target_clean_col)
                with col_blank:
                    if st.button("🗑️ Drop Empty Rows", use_container_width=True):
                        def remove_blank_rows(df):
                            mask = df[target_clean_col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "tidak tersedia"])
                            return df[~mask]
                        apply_transform(remove_blank_rows, f"Hapus Baris Kosong '{target_clean_col}'", is_row_filter=True, target_col=target_clean_col)
                
                st.markdown("**🔍 C. Find & Replace (Regex Off)**")
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
                st.markdown("**🧠 D. Clustering / Penggabungan Varian Kata**")
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
                                        if method == "Levenshtein Distance" and levenshtein(kw1.lower(), kw2.lower(), lev_threshold) <= lev_threshold: match = True
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
                                        if AI_PROVIDER == "Mistral": ai_response = call_mistral_sync(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                        elif AI_PROVIDER == "Google Gemini": ai_response = call_gemini_sync(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                            
                                        try:
                                            match = re.search(r'\{.*\}', ai_response, re.DOTALL)
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

        if st.button("🚀 Eksekusi Mesin AI Batching", type="primary"):
            if not AI_API_KEY: 
                st.error(f"⚠️ Kredensial {AI_PROVIDER} API Key hilang. Harap lengkapi di menu Expand Settings Sidebar.")
            elif analysis_task == "Tanya Bebas (Custom Prompt)" and not custom_prompt.strip(): 
                st.warning("⚠️ Untuk Custom Prompt, area ketik instruksi mutlak diperlukan.")
            else:
                ai_data = data.copy()
                if sort_order == "Terbaru (Descending Year)" and year_col: 
                    ai_data = ai_data.sort_values(by=year_col, ascending=False)
                elif sort_order == "Terlama (Ascending Year)" and year_col: 
                    ai_data = ai_data.sort_values(by=year_col, ascending=True)
                elif sort_order == "High-Impact (Sitasi Terbanyak)" and citation_col:
                    ai_data[citation_col] = pd.to_numeric(ai_data[citation_col], errors='coerce').fillna(0)
                    ai_data = ai_data.sort_values(by=citation_col, ascending=False)
                
                docs_to_process = []
                sampled_titles = [] 
                
                for i in range(min(num_samples, len(ai_data))):
                    doc_text = str(ai_data[target_col].iloc[i])
                    if title_col:
                        doc_title = str(ai_data[title_col].iloc[i])
                        sampled_titles.append(f"**[{i+1}]** {doc_title}")
                        docs_to_process.append(f"Dokumen {i+1} [TI: {doc_title}]:\n{doc_text}\n\n")
                    else:
                        sampled_titles.append(f"**[{i+1}]** (Metadata Judul tidak direferensikan)")
                        docs_to_process.append(f"Dokumen {i+1}:\n{doc_text}\n\n")

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
                    user_prompt = f"Konstan. Berikut adalah Data Batch ({chunk_no} dari {total_chunks}). Evaluasi bagian corpus ini:\n\n{formatted_texts}"
                    
                    st.markdown(f"##### ⏳ Streaming Inference (Batch {chunk_no}/{total_chunks}) [Dokumen ke-{i+1} s/d {i+len(batch)}]...")
                    with st.container(border=True):
                        if AI_PROVIDER == "Mistral":
                            result_text = st.write_stream(stream_mistral(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                        elif AI_PROVIDER == "Google Gemini":
                            result_text = st.write_stream(stream_gemini(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                        
                        if "❌ Error" not in result_text:
                            full_report_text += f"\n\n=======================================\nSINTESIS BATCH {chunk_no} (DOKUMEN {i+1}-{i+len(batch)})\n=======================================\n" + result_text

                if full_report_text:
                    st.success("✅ Seluruh antrian Batch berhasil dieksekusi secara berurutan.")
                    st.download_button("📥 Ekspor Laporan Skrip (.txt)", data=full_report_text, file_name="laporan_ai_sintesis_full.txt", mime="text/plain", type="primary")


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
                    default_idx = cleanable_cols_b.index(author_col) if is_author_network and author_col in cleanable_cols_b else 0
                    net_col = st.selectbox("Entitas Pemetaan (Nodes):", cleanable_cols_b, index=default_idx)
                    default_delim_idx = 0 if is_author_network else 0
                    net_delim = st.selectbox("Format Pemisah Internal:", [";", ",", "|", "Tidak Ada"], index=default_delim_idx)

                st.markdown("---")
                
                # Membagi Form Konfigurasi Network Map dan Thematic Map Persis Biblioshiny
                col_set1, col_set2 = st.columns(2)
                
                with col_set1:
                    st.markdown("#### ⚙️ Co-occurrence Network Settings")
                    
                    st.markdown("<p style='color:#2c3e50; font-weight:bold; margin-bottom:0;'>Method Parameters</p>", unsafe_allow_html=True)
                    col_n1, col_n2 = st.columns(2)
                    net_layout = col_n1.selectbox("Network Layout", ["Automatic layout", "Circle layout", "Kamada-Kawai", "Fruchterman-Reingold"], index=0)
                    net_algo = col_n2.selectbox("Clustering Algorithm", ["Louvain", "Betweenness", "InfoMap", "Leading Eigenvector", "Leiden", "Spinglass", "Walktrap"], index=0, key="net_algo")
                    net_norm = col_n1.selectbox("Normalization Method", ["Association", "Equivalence", "Jaccard", "Salton", "Inclusion", "Raw"], index=0)
                    net_color_year = col_n2.selectbox("Node Color by Year", ["No", "Yes"], index=0)

                    st.markdown("<p style='color:#2c3e50; font-weight:bold; margin-bottom:0; margin-top:10px;'>Network Size</p>", unsafe_allow_html=True)
                    col_n3, col_n4 = st.columns(2)
                    net_n_nodes = col_n3.number_input("Number of Nodes", 10, 500, 50, step=10, key="net_n_nodes")
                    net_repulsion = col_n4.number_input("Repulsion Force", 0.05, 5.0, 0.5, step=0.05, key="net_repulsion")

                    st.markdown("<p style='color:#f39c12; font-weight:bold; margin-bottom:0; margin-top:10px;'>Filtering Options</p>", unsafe_allow_html=True)
                    col_n5, col_n6 = st.columns(2)
                    net_remove_isolates = col_n5.selectbox("Remove Isolated Nodes", ["Yes", "No"], index=0)
                    net_min_edges = col_n6.number_input("Minimum Number of Edges", 1, 20, 2, key="net_min_edges")

                with col_set2:
                    st.markdown("#### ⚙️ Parameters (Thematic Map)")
                    
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
                    theme_algo = col_t6.selectbox("Clustering Algorithm", ["Louvain", "Betweenness", "InfoMap", "Leading Eigenvector", "Leiden", "Spinglass", "Walktrap"], index=0, key="thm_algo")

                # REAKTIVITAS: Eksekusi Otomatis Berdasarkan Perubahan Parameter
                actual_net_delim = {";": ";", ",": ",", "|": "|"}.get(net_delim)
                
                with st.spinner("Mengeksekusi matriks jarak dan struktur graf secara otomatis..."):
                    # ---------------------------------------------------------
                    # PIPELINE 1: NETWORK MAP (Co-occurrence)
                    # ---------------------------------------------------------
                    df_mapped = preprocess_keywords(data, field=net_col, delimiter=actual_net_delim, is_author=is_author_network)
                    
                    G_raw_net, wc_net = build_cooccurrence(df_mapped, field=net_col, minfreq=1)
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

                    # ---------------------------------------------------------
                    # PIPELINE 2: THEMATIC MAP
                    # ---------------------------------------------------------
                    total_documents = len(df_mapped)
                    actual_min_freq_theme = max(1, math.ceil((theme_min_freq / 1000) * total_documents))

                    G_raw_theme, wc_theme = build_cooccurrence(df_mapped, field=net_col, minfreq=actual_min_freq_theme)
                    
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
                tab_net, tab_map, tab_evol, tab_three, tab_trend, tab_table = st.tabs([
                    network_title, "📍 Thematic Map", "⏳ Thematic Evolution", 
                    "🔀 Three-Fields Plot", "📈 Trend Topics", "📊 Network Analytics"
                ])
                
                # ===============================================
                # 1. NETWORK MAP RENDERER
                # ===============================================
                with tab_net:
                    if len(G_final_net.nodes()) == 0:
                        st.warning("Jaringan kosong. Silakan kurangi 'Minimum Number of Edges' atau tambah 'Number of Nodes' pada Pengaturan Network.")
                    else:
                        st.caption("Visualisasi interaktif graf berdasarkan pengaturan Co-occurrence Network di atas.")
                        
                        col_dl1, col_dl2 = st.columns([4, 1])
                        with col_dl2:
                            gexf_str = generate_gexf_string(G_final_net)
                            st.download_button(label="📥 Export to Gephi (.gexf)", data=gexf_str, file_name="network_graph.gexf", mime="application/xml")

                        edge_x, edge_y = [], []
                        for edge in G_final_net.edges():
                            x0, y0 = pos_net[edge[0]]
                            x1, y1 = pos_net[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])

                        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.8, color='rgba(200,200,200,0.5)'), hoverinfo='none', mode='lines')

                        node_x, node_y, node_text, node_hover, node_size, node_colors, text_sizes = [], [], [], [], [], [], []
                        color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
                        
                        # LOGIC: NODE COLOR BY YEAR
                        word_avg_years = {}
                        if net_color_year == "Yes" and year_col and year_col in df_mapped.columns:
                            word_years = defaultdict(list)
                            for idx_df, row in df_mapped.iterrows():
                                try:
                                    yr = int(row[year_col])
                                    for w in row[net_col]:
                                        if w in G_final_net.nodes():
                                            word_years[w].append(yr)
                                except: pass
                            word_avg_years = {w: np.mean(yrs) for w, yrs in word_years.items() if yrs}
                        
                        # Node mapping
                        node_to_comm_net = {}
                        for idx, comm in enumerate(communities_net):
                            for node in comm: node_to_comm_net[node] = idx

                        color_scale_values = []
                        
                        for node in G_final_net.nodes():
                            x, y = pos_net[node]
                            node_x.append(x); node_y.append(y); node_text.append(node)
                            freq = G_final_net.nodes[node]['freq']
                            
                            if net_color_year == "Yes" and node in word_avg_years:
                                avg_y = word_avg_years[node]
                                node_colors.append(avg_y)
                                node_hover.append(f"Entitas: <b>{node}</b><br>Frekuensi: {freq}<br>Rata-rata Tahun: {round(avg_y, 1)}")
                            else:
                                c_idx = node_to_comm_net.get(node, 0)
                                node_colors.append(color_palette[c_idx % len(color_palette)])
                                node_hover.append(f"Entitas: <b>{node}</b><br>Frekuensi Agregat: {freq}<br>Klaster: {c_idx+1}")
                                
                            node_size.append(min(max(freq * 1.5, 15), 65))
                            
                            base_font_size = 10 + math.sqrt(freq) * 2
                            text_sizes.append(base_font_size)

                        if net_color_year == "Yes" and word_avg_years:
                            marker_dict = dict(
                                color=node_colors, colorscale='Viridis', showscale=True, 
                                size=node_size, line_width=1, line_color='rgba(255,255,255,0.8)', opacity=0.9,
                                colorbar=dict(title="Avg. Pub Year")
                            )
                        else:
                            marker_dict = dict(color=node_colors, size=node_size, line_width=1, line_color='rgba(255,255,255,0.8)', opacity=0.9)

                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", 
                            hoverinfo='text', hovertext=node_hover, 
                            marker=marker_dict,
                            textfont=dict(size=text_sizes, color='black', family='Arial')
                        )

                        fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title_font_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='white', height=650))
                        st.plotly_chart(fig_net, use_container_width=True, config=PLOTLY_DL_CONFIG)

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
                        
                        mid_c = df_theme['Centrality'].mean() if not df_theme['Centrality'].empty else 0
                        mid_d = df_theme['Density'].mean() if not df_theme['Density'].empty else 0
                        
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
                    
                    if not year_col:
                        st.warning("Metadata Temporal (Tahun Publikasi) tidak ditemukan. Modul evolusi dinonaktifkan.")
                    else:
                        temp_df = data.copy()
                        temp_df['Year_Num'] = pd.to_numeric(temp_df[year_col], errors='coerce')
                        temp_df = temp_df.dropna(subset=['Year_Num'])
                        
                        if not temp_df.empty:
                            min_y = int(temp_df['Year_Num'].min())
                            max_y = int(temp_df['Year_Num'].max())
                            
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
                                    df_p1 = temp_df[temp_df['Year_Num'] <= cut_year]
                                    df_p2 = temp_df[temp_df['Year_Num'] > cut_year]
                                    
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
                    st.markdown("#### 🔀 Three-Fields Plot")
                    st.caption("Visualisasi komprehensif Penulis (Kiri) ➔ Kata Kunci (Tengah) ➔ Jurnal (Kanan).")
                    
                    if not author_col or not journal_col:
                        st.warning("Metadata Penulis atau Jurnal tidak lengkap dalam dataset.")
                    else:
                        try:
                            N_TOP = 15 
                            tf_df = data.copy()
                            
                            auth_list = tf_df[author_col].astype(str).str.split(";")
                            jour_list = tf_df[journal_col].astype(str)
                            
                            if actual_net_delim:
                                key_list = tf_df[net_col].astype(str).str.lower().str.split(actual_net_delim)
                            else:
                                key_list = tf_df[net_col].astype(str).str.lower().apply(lambda x: [x])
                            
                            all_a = [a.strip() for sublist in auth_list.dropna() for a in sublist if a.strip()]
                            top_a = [x[0] for x in Counter(all_a).most_common(N_TOP)]
                            
                            all_j = [j.strip() for j in jour_list.dropna() if j.strip()]
                            top_j = [x[0] for x in Counter(all_j).most_common(N_TOP)]
                            
                            all_k = [k.strip() for sublist in key_list.dropna() for k in sublist if k.strip() and k.strip() not in COMMON_STOPWORDS and k.strip() not in ["tidak tersedia", "n/a", "no title"]]
                            top_k = [x[0] for x in Counter(all_k).most_common(N_TOP)]
                            
                            labels = top_a + top_k + top_j
                            
                            link_counts_ak = defaultdict(int)
                            link_counts_kj = defaultdict(int)
                            
                            for i in range(len(tf_df)):
                                a_s = [a.strip() for a in auth_list.iloc[i] if a.strip() in top_a] if isinstance(auth_list.iloc[i], list) else []
                                k_s = [k.strip() for k in key_list.iloc[i] if k.strip() in top_k] if isinstance(key_list.iloc[i], list) else []
                                j_val = jour_list.iloc[i].strip()
                                j_s = [j_val] if j_val in top_j else []
                                
                                for a in a_s:
                                    for k in k_s:
                                        link_counts_ak[(a, k)] += 1
                                
                                for k in k_s:
                                    for j in j_s:
                                        link_counts_kj[(k, j)] += 1
                            
                            source_idx = []
                            target_idx = []
                            vals = []
                            
                            for (a, k), v in link_counts_ak.items():
                                source_idx.append(labels.index(a))
                                target_idx.append(labels.index(k))
                                vals.append(v)
                                
                            for (k, j), v in link_counts_kj.items():
                                source_idx.append(labels.index(k))
                                target_idx.append(labels.index(j))
                                vals.append(v)
                                
                            fig_three = go.Figure(data=[go.Sankey(
                                node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="#2ecc71"),
                                link=dict(source=source_idx, target=target_idx, value=vals, color="rgba(46, 204, 113, 0.4)")
                            )])
                            
                            fig_three.update_layout(title_text="Authors ➔ Keywords ➔ Journals", font_size=11, height=750)
                            st.plotly_chart(fig_three, use_container_width=True, config=PLOTLY_DL_CONFIG)
                            
                        except Exception as e:
                            st.error(f"Gagal memproses matriks 3 dimensi: {e}")

                # ===============================================
                # 5. TREND TOPICS (WORD DYNAMICS)
                # ===============================================
                with tab_trend:
                    st.markdown("#### 📈 Dinamika Topik Tren (Trend Topics)")
                    st.caption("Peta persebaran frekuensi kata kunci spesifik di sepanjang rentang waktu (tahun publikasi).")
                    
                    if not year_col:
                        st.warning("Metadata Tahun Publikasi tidak ditemukan.")
                    else:
                        temp_trend = data[[year_col, net_col]].copy()
                        temp_trend[year_col] = pd.to_numeric(temp_trend[year_col], errors='coerce')
                        temp_trend = temp_trend.dropna()
                        
                        if actual_net_delim:
                            temp_trend['Words'] = temp_trend[net_col].astype(str).str.lower().str.split(actual_net_delim)
                        else:
                            temp_trend['Words'] = temp_trend[net_col].astype(str).str.lower().apply(lambda x: [x])
                        
                        temp_trend = temp_trend.explode('Words')
                        temp_trend['Words'] = temp_trend['Words'].str.strip()
                        
                        valid_mask = (temp_trend['Words'] != "") & (~temp_trend['Words'].isin(COMMON_STOPWORDS)) & (temp_trend['Words'].str.len() > 3) & (~temp_trend['Words'].isin(["tidak tersedia", "n/a", "no title"]))
                        temp_trend = temp_trend[valid_mask]
                        
                        top_trend_words = temp_trend['Words'].value_counts().head(15).index.tolist()
                        temp_trend = temp_trend[temp_trend['Words'].isin(top_trend_words)]
                        
                        trend_grouped = temp_trend.groupby([year_col, 'Words']).size().reset_index(name='Frequency')
                        
                        if not trend_grouped.empty:
                            fig_trend = px.scatter(trend_grouped, x=year_col, y="Words", size="Frequency", color="Words",
                                                 title="Rentang Hidup dan Puncak Topik Riset",
                                                 labels={year_col: "Tahun Publikasi", "Words": "Kata Kunci Dominan"},
                                                 size_max=35, height=650)
                            
                            fig_trend.update_layout(xaxis=dict(tickformat="d"), showlegend=False)
                            for w in top_trend_words:
                                fig_trend.add_hline(y=w, line_width=0.5, line_color="lightgray", opacity=0.5)
                                
                            st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_DL_CONFIG)
                        else:
                            st.info("Data dinamis tidak mencukupi untuk pemplotan waktu.")

                # ===============================================
                # 6. TABLES AND NODE CENTRALITY METRICS
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

    # ---------------------------------------------------------
    # MENU 6: AI CHATBOT (TRUE RAG SEMANTIC SEARCH SYSTEM)
    # ---------------------------------------------------------
    elif menu_selection == "💬 AI Chatbot (RAG)":
        st.title("💬 Interactive AI Research Assistant (RAG System)")
        st.markdown("Sistem *Retrieval-Augmented Generation* tingkat lanjut. Mesin akan menggunakan Cosine Similarity (TF-IDF) untuk memindai ribuan jurnal Anda secara instan dan mencari wawasan terbaik sebelum merespon pertanyaan.")
        
        target_rag_col = abstract_col if abstract_col else title_col
        
        if not target_rag_col:
            st.error("⚠️ Kolom teks naratif (Abstrak/Judul) tidak ditemukan di dataset untuk diriset.")
        elif not HAS_SKLEARN:
            st.error("💡 Modul Machine Learning `scikit-learn` mutlak diperlukan untuk pencarian Semantic RAG. Jalankan `pip install scikit-learn`.")
        else:
            with st.expander("🛠️ Konfigurasi Mesin RAG (Information Retrieval)", expanded=False):
                st.write(f"Kolom target pencarian Vektor: **{target_rag_col}**")
                top_k = st.slider("Jumlah Jurnal Relevan yang Disuntikkan ke Memori AI (Top K):", 5, 30, 10)
                st.caption("Semakin banyak 'K', konteks AI semakin luas, namun berisiko melebihi batas Token API.")

            st.markdown("---")

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_chat_input := st.chat_input("Tanyakan sesuatu ke dataset Anda (Contoh: 'Jurnal mana saja yang membahas metode Machine Learning?')"):
                if not AI_API_KEY:
                    st.error(f"⚠️ Operasi diinterupsi. Masukkan {AI_PROVIDER} API Key pada prapengaturan (Settings Sidebar).")
                else:
                    st.session_state.chat_messages.append({"role": "user", "content": user_chat_input})
                    with st.chat_message("user"):
                        st.markdown(user_chat_input)

                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_chat_response = ""
                        
                        with st.spinner("Mencari jurnal yang paling relevan (TF-IDF Similarity Search)..."):
                            corpus = data[target_rag_col].fillna("").astype(str).tolist()
                            titles = data[title_col].fillna("No Title").astype(str).tolist() if title_col else ["No Title"] * len(corpus)
                            years = data[year_col].fillna("N/A").astype(str).tolist() if year_col else ["N/A"] * len(corpus)
                            
                            vectorizer = TfidfVectorizer(stop_words='english')
                            tfidf_matrix = vectorizer.fit_transform(corpus)
                            query_vec = vectorizer.transform([user_chat_input])
                            
                            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                            top_indices = similarities.argsort()[-top_k:][::-1]
                            
                            context_string = ""
                            for idx in top_indices:
                                if similarities[idx] > 0.01: 
                                    context_string += f"[{titles[idx]} (Tahun: {years[idx]})]\nIsi Teks: {corpus[idx]}\n\n"

                            if not context_string.strip():
                                context_string = "TIDAK ADA DATA JURNAL YANG RELEVAN DENGAN PERTANYAAN INI DI DATABASE."

                        with st.spinner("Berpikir & Menulis Jawaban..."):
                            rag_system_prompt = f"""Anda adalah 'BiblioBot', Asisten Riset Saintifik yang sangat teliti.
TUGAS UTAMA ANDA: Jawablah pertanyaan user HANYA berdasarkan cuplikan referensi JURNAL RELEVAN di bawah ini.
ATURAN EMAS: 
1. Jangan berhalusinasi. Jika tidak ada di teks referensi, katakan "Berdasarkan dataset, tidak ditemukan informasi tersebut."
2. SELALU SEBUTKAN NAMA JURNAL (di dalam tanda kurung siku []) jika Anda mengutip suatu fakta.
=== REFERENSI JURNAL RELEVAN (HASIL SEARCH TF-IDF) ===
{context_string}
======================
"""
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
                                    
                        response_placeholder.markdown(full_chat_response)
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_chat_response})

# ==============================
# FALLBACK: BELUM ADA DATA
# ==============================
else:
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.info("👈 Selamat Datang di **Biblio Analyzer Pro**. Silakan gunakan panel navigasi di bilah sebelah kiri untuk memasukkan Kredensial Akses Anda dan mengunggah/mengimpor set Data Riset (.csv/.json).")
    st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'><i>Engineered and enhanced by LLM Architecture. Built for modern enterprise-scale bibliometric studies.</i></p>", unsafe_allow_html=True)