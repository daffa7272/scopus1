import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import re
import unicodedata
import zlib
import json
from collections import Counter



# Untuk Word Cloud
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# ==============================
# CONFIG & STYLING
# ==============================
st.set_page_config(
    page_title="Scopus AI Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3em; 
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(128, 130, 137, 0.1);
        border: 1px solid rgba(128, 130, 137, 0.2);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        backdrop-filter: blur(5px);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# FUNGSI-FUNGSI PENDUKUNG AI (MISTRAL & GEMINI)
# ==============================

# --- MISTRAL AI ---
def stream_mistral(system_prompt, user_prompt, api_key, model):
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
    except Exception as e: yield f"❌ Terjadi kesalahan: {e}"

@st.cache_data(ttl=3600, show_spinner=False)
def call_mistral(system_prompt, user_prompt, api_key, model):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.1, "response_format": {"type": "json_object"}}
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=None)
        if response.status_code == 200: return response.json()["choices"][0]["message"]["content"]
        else: return f"Error: {response.text}"
    except Exception as e: return f"Error: {e}"

# --- GOOGLE GEMINI ---
def stream_gemini(system_prompt, user_prompt, api_key, model):
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
    except Exception as e: yield f"❌ Terjadi kesalahan: {e}"

@st.cache_data(ttl=3600, show_spinner=False)
def call_gemini(system_prompt, user_prompt, api_key, model):
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
# FUNGSI DATA WRANGLING
# ==============================
def clean_scopus_data(raw_data):
    cleaned_list = []
    entries = raw_data.get('search-results', {}).get('entry', [])
    for entry in entries:
        cleaned_list.append({
            "Judul": entry.get('dc:title', 'Tidak tersedia'),
            "Abstract": entry.get('dc:description', 'Tidak tersedia'),
            "Penulis": entry.get('dc:creator', 'Tidak tersedia'),
            "Jurnal": entry.get('prism:publicationName', 'Tidak tersedia'),
            "Tahun": str(entry.get('prism:coverDate', 'N/A'))[:4],
            "DOI": entry.get('prism:doi', 'Tidak tersedia'),
            "Citasi": entry.get('citedby-count', '0')
        })
    return pd.DataFrame(cleaned_list)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

COMMON_STOPWORDS = {"the", "and", "of", "to", "in", "a", "is", "that", "for", "on", "with", "as", "by", "an", "this", "from", "are", "we", "at", "be", "it", "tidak", "tersedia", "yang", "dan", "di", "dari", "ke", "untuk"}

def get_top_words(text_series, top_n=10):
    all_text = " ".join(text_series.dropna().astype(str)).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    filtered_words = [word for word in words if word not in COMMON_STOPWORDS]
    word_counts = Counter(filtered_words)
    return pd.DataFrame(word_counts.most_common(top_n), columns=['Kata', 'Frekuensi']).set_index('Kata')

# --- FUNGSI CLUSTERING OPENREFINE ---
def get_fingerprint(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) 
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn') 
    tokens = sorted(list(set(text.split()))) 
    return " ".join(tokens)

def get_ngram_fingerprint(text, n=2):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.replace(" ", "") 
    if len(text) < n: return text
    ngrams = sorted(list(set([text[i:i+n] for i in range(len(text)-n+1)])))
    return "".join(ngrams)

def get_soundex(token):
    token = str(token).upper()
    token = re.sub(r'[^A-Z]', '', token)
    if not token: return ""
    soundex = token[0]
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    for char in token[1:]:
        for key, code in dictionary.items():
            if char in key and code != soundex[-1]:
                soundex += code
    soundex = soundex.replace(".", "")[:4].ljust(4, "0")
    return soundex

def get_phonetic_fingerprint(text):
    tokens = re.sub(r'[^\w\s]', '', str(text)).split()
    return " ".join(sorted(list(set([get_soundex(t) for t in tokens]))))

def levenshtein(s1, s2, max_dist):
    if abs(len(s1) - len(s2)) > max_dist: return max_dist + 1
    if len(s1) < len(s2): return levenshtein(s2, s1, max_dist)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        if min(previous_row) > max_dist: return max_dist + 1
    return previous_row[-1]

def ppm_distance(s1, s2):
    c1 = len(zlib.compress(s1.encode('utf-8')))
    c2 = len(zlib.compress(s2.encode('utf-8')))
    c12 = len(zlib.compress((s1 + ' ' + s2).encode('utf-8')))
    return (c12 - min(c1, c2)) / max(c1, c2)

# ==============================
# INISIALISASI SESSION STATE 
# ==============================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'history_actions' not in st.session_state:
    st.session_state.history_actions = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = -1

if 'clustering_result' not in st.session_state:
    st.session_state.clustering_result = None
if 'preview_action' not in st.session_state:
    st.session_state.preview_action = None
if 'preview_original' not in st.session_state:
    st.session_state.preview_original = None
if 'preview_new' not in st.session_state:
    st.session_state.preview_new = None

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
# SIDEBAR - KONTROL & KREDENSIAL
# ==============================
with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=50)
    with col2:
        st.markdown("### Control Panel")
    
    st.divider()
    
    st.info("📂 **Pilih Sumber Data Riset**")
    input_option = st.radio("Metode Input:", ["Upload CSV/JSON", "Scopus API Search"], label_visibility="collapsed")
    
    st.divider()
    
    with st.expander("🔑 Konfigurasi API Keys & Model", expanded=True):
        SCOPUS_API_KEY = st.text_input("Scopus API Key", type="password", placeholder="Masukkan key Scopus").strip()
        
        st.markdown("**Konfigurasi AI Analysis**")
        AI_PROVIDER = st.selectbox("Penyedia AI:", ["Mistral", "Google Gemini"])
        
        if AI_PROVIDER == "Mistral":
            AI_API_KEY = st.text_input("Mistral API Key", type="password", placeholder="Masukkan key Mistral").strip()
            AI_MODEL = st.selectbox("Pilih Model (Terupdate):", ["mistral-small-latest", "open-mistral-nemo", "mistral-large-latest"])
            st.caption("✨ 'mistral-small-latest' adalah model teringan dan bersahabat dengan free-tier.")
            
        elif AI_PROVIDER == "Google Gemini":
            AI_API_KEY = st.text_input("Gemini API Key", type="password", placeholder="Masukkan key Gemini").strip()
            AI_MODEL = st.selectbox("Pilih Model (Terupdate):", ["gemini-2.5-flash", "gemini-2.5-pro"])
            st.caption("✨ 'gemini-2.5-flash' sangat cepat, efisien, dan memiliki kuota gratis yang besar.")
    
    with st.expander("⚙️ Pengaturan Tampilan", expanded=False):
        show_raw_data = st.checkbox("Tampilkan Tabel Data Mentah", value=True)

# ==============================
# HALAMAN UTAMA (MAIN CONTENT)
# ==============================
st.title("🔬 Scopus AI Research Analyzer")
st.markdown("Cari, ekstrak, dan analisis publikasi ilmiah secara otomatis menggunakan kecerdasan buatan.")

# --- OPSI 1: CARI VIA SCOPUS API ---
if input_option == "Scopus API Search":
    col_q, col_num, col_btn = st.columns([3, 1, 1])
    with col_q:
        query = st.text_input("🔍 Masukkan Query Scopus", placeholder="Contoh: TITLE-ABS-KEY(Artificial Intelligence)")
    with col_num:
        max_results = st.number_input("Jumlah Data", min_value=10, max_value=100, value=25, step=5)
    with col_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Cari Data", type="primary", use_container_width=True):
            if not SCOPUS_API_KEY: st.error("⚠️ Masukkan Scopus API Key.")
            elif not query: st.warning("⚠️ Masukkan query.")
            else:
                with st.spinner("Mengunduh data..."):
                    url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={max_results}&view=COMPLETE&apiKey={SCOPUS_API_KEY}"
                    resp = requests.get(url, headers={'Accept': 'application/json'})
                    
                    if resp.status_code in [401, 403]:
                        st.toast("⚠️ Akses institusi tidak terdeteksi. Beralih ke pencarian dasar (tanpa abstrak)...", icon="🔄")
                        url = f"https://api.elsevier.com/content/search/scopus?query={query}&count={max_results}&apiKey={SCOPUS_API_KEY}"
                        resp = requests.get(url, headers={'Accept': 'application/json'})

                    if resp.status_code == 200:
                        loaded_data = clean_scopus_data(resp.json())
                        st.session_state.history = [loaded_data]
                        st.session_state.history_actions = ["Data Awal (Scopus API)"]
                        st.session_state.current_step = 0
                        st.session_state.preview_action = None
                        st.success(f"✅ Ditarik {len(loaded_data)} dokumen.")
                        st.rerun()
                    else: st.error(f"❌ Error {resp.status_code}: {resp.text}")

# --- OPSI 2: UPLOAD FILE ---
else:
    st.markdown("##### Unggah Berkas Anda")
    uploaded_file = st.file_uploader("Mendukung format .csv atau .json", type=["csv", "json"], label_visibility="collapsed")
    if uploaded_file:
        if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
            try:
                loaded_data = pd.read_json(uploaded_file) if uploaded_file.name.endswith(".json") else pd.read_csv(uploaded_file)
                loaded_data = loaded_data.fillna("Tidak tersedia")
                st.session_state.history = [loaded_data]
                st.session_state.history_actions = [f"Data Awal ({uploaded_file.name})"]
                st.session_state.current_step = 0
                st.session_state.last_file = uploaded_file.name
                st.session_state.preview_action = None
                st.success(f"✅ Memuat {len(loaded_data)} baris.")
                st.rerun()
            except Exception as e: st.error(f"❌ Error: {e}")

# ==============================
# FILTER DINAMIS & AREA DASHBOARD
# ==============================
if st.session_state.current_step >= 0 and len(st.session_state.history) > 0:
    base_data = st.session_state.history[st.session_state.current_step].copy()
    data = base_data.copy() 

    year_col = next((col for col in ['Tahun', 'Year', 'year'] if col in data.columns), None)
    journal_col = next((col for col in ['Jurnal', 'Source title', 'Journal'] if col in data.columns), None)
    author_col = next((col for col in ['Penulis', 'Authors', 'Author'] if col in data.columns), None)
    title_col = next((col for col in ['Judul', 'Title', 'title', 'Document Title'] if col in data.columns), None)
    citation_col = next((col for col in ['Citasi', 'Cited by', 'citedby-count'] if col in data.columns), None)
    abstract_col = next((col for col in ['Abstract', 'abstract', 'Description'] if col in data.columns), None)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📅 Filter Tampilan Dataset")
        st.caption("Filter ini hanya mengubah tampilan, tidak menghapus data asli.")
        search_kw = st.text_input("🔍 Cari kata spesifik:", placeholder="Cari judul/abstrak...")
        if search_kw:
            mask = data.astype(str).apply(lambda x: x.str.contains(search_kw, case=False, na=False)).any(axis=1)
            data = data[mask]
        
        if year_col and len(data) > 0:
            data['Year_Numeric'] = pd.to_numeric(data[year_col], errors='coerce')
            min_year = int(data['Year_Numeric'].min()) if not pd.isna(data['Year_Numeric'].min()) else 2000
            max_year = int(data['Year_Numeric'].max()) if not pd.isna(data['Year_Numeric'].max()) else 2025
            if min_year < max_year:
                selected_years = st.slider("Rentang Tahun Publikasi:", min_year, max_year, (min_year, max_year))
                data = data[(data['Year_Numeric'] >= selected_years[0]) & (data['Year_Numeric'] <= selected_years[1])]
            data = data.drop(columns=['Year_Numeric'])

    if len(data) == 0:
        st.warning("⚠️ Tidak ada data yang cocok dengan filter pencarian Anda.")
    else:
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🤖 AI Analysis", "📈 Grafik & Tren", "🧹 Data Cleaning"])
        
        # --- TAB 1: TABEL DATA ---
        with tab1:
            st.markdown("### Ringkasan Dataset")
            m1, m2, m3 = st.columns(3)
            m1.metric("📄 Total Dokumen Terpilih", len(data))
            m2.metric("📋 Jumlah Kolom", len(data.columns))
            m3.metric("✅ Status Data", "Siap Dianalisis")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if show_raw_data:
                col_view, col_dl = st.columns([3, 1])
                with col_view:
                    all_cols = data.columns.tolist()
                    default_cols = [c for c in ['Title', 'Judul', 'Authors', 'Penulis', 'Year', 'Tahun', 'Source title', 'Jurnal', 'Cited by', 'Citasi'] if c in all_cols]
                    if not default_cols: default_cols = all_cols[:5]
                    selected_cols = st.multiselect("Pilih kolom yang ingin ditampilkan:", options=all_cols, default=default_cols)
                with col_dl:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(label="📥 Download CSV", data=convert_df_to_csv(data), file_name="scopus_dataset_filtered.csv", mime="text/csv", use_container_width=True)
                
                if selected_cols: st.dataframe(data[selected_cols], use_container_width=True)

        # --- TAB 2: ANALISIS MISTRAL AI ---
        with tab2:
            st.markdown("### Otak Analisis (AI Assistant)")
            col_cfg1, col_cfg2 = st.columns(2)
            with col_cfg1:
                target_col = st.selectbox("Pilih kolom teks target (Misal: Title/Abstract):", data.columns)
                analysis_task = st.selectbox("Pilih Tugas AI:", ["Buatkan Ringkasan Eksekutif", "Klasifikasikan Topik Utama", "Identifikasi Tren Riset (Research Gap)", "Ekstrak Kata Kunci Utama (Keywords)", "Tanya Bebas (Custom Prompt)"])
                custom_prompt = st.text_area("Ketik instruksi spesifik Anda di sini:", placeholder="Contoh: Carikan semua tantangan/limitasi penelitian...") if analysis_task == "Tanya Bebas (Custom Prompt)" else ""
            with col_cfg2:
                num_samples = st.slider("Jumlah sampel dokumen yang dibaca AI:", 1, min(100, len(data)), min(20, len(data)))
                sort_options = ["Bawaan (Sesuai Urutan File)"]
                if year_col: sort_options.extend(["Tahun Terbaru", "Tahun Terlama"])
                if citation_col: sort_options.append("Sitasi Terbanyak")
                sort_order = st.selectbox("Prioritas Pengambilan Sampel AI:", sort_options)

            if st.button("🚀 Mulai Analisis AI", type="primary"):
                if not AI_API_KEY: st.error(f"⚠️ Masukkan {AI_PROVIDER} API Key di pengaturan Sidebar (kiri) terlebih dahulu.")
                elif analysis_task == "Tanya Bebas (Custom Prompt)" and not custom_prompt.strip(): st.warning("⚠️ Ketik instruksi Anda.")
                else:
                    ai_data = data.copy()
                    if sort_order == "Tahun Terbaru" and year_col: ai_data = ai_data.sort_values(by=year_col, ascending=False)
                    elif sort_order == "Tahun Terlama" and year_col: ai_data = ai_data.sort_values(by=year_col, ascending=True)
                    elif sort_order == "Sitasi Terbanyak" and citation_col:
                        ai_data[citation_col] = pd.to_numeric(ai_data[citation_col], errors='coerce').fillna(0)
                        ai_data = ai_data.sort_values(by=citation_col, ascending=False)
                    
                    docs_to_process = []
                    sampled_titles = [] 
                    
                    for i in range(min(num_samples, len(ai_data))):
                        doc_text = str(ai_data[target_col].iloc[i])
                        if title_col:
                            doc_title = str(ai_data[title_col].iloc[i])
                            sampled_titles.append(f"**{i+1}.** {doc_title}")
                            docs_to_process.append(f"Dokumen {i+1} (Judul Asli: {doc_title}):\n{doc_text}\n\n")
                        else:
                            sampled_titles.append(f"**{i+1}.** (Tidak ada kolom judul)")
                            docs_to_process.append(f"Dokumen {i+1}:\n{doc_text}\n\n")

                    task_to_send = custom_prompt if analysis_task == "Tanya Bebas (Custom Prompt)" else analysis_task
                    system_prompt = f"""Sebagai pakar riset akademis senior, tugas Anda adalah: {task_to_send}
ATURAN SANGAT PENTING:
1. WAJIB menggunakan 'Judul Asli' persis seperti aslinya.
2. DILARANG KERAS menerjemahkan 'Judul Asli'.
3. Penjelasan analisis HARUS dalam bahasa Indonesia yang baku dan profesional."""

                    st.markdown("---")
                    st.markdown(f"### 📝 Laporan Hasil Analisis ({AI_PROVIDER})")
                    with st.expander(f"📄 Lihat Daftar {len(sampled_titles)} Judul Sampel", expanded=True):
                        for title in sampled_titles: st.markdown(title)
                    
                    # FITUR BARU: Pemrosesan AI secara Batching/Chunking agar tidak Error 503
                    BATCH_SIZE = 15 # Mengirim 15 dokumen sekaligus untuk menghindari Payload Too Large
                    full_report_text = ""
                    
                    for i in range(0, len(docs_to_process), BATCH_SIZE):
                        batch = docs_to_process[i:i+BATCH_SIZE]
                        formatted_texts = "".join(batch)
                        user_prompt = f"Berdasarkan data penelitian (Bagian {i//BATCH_SIZE + 1} dari {(len(docs_to_process)-1)//BATCH_SIZE + 1}), berikan analisis Anda:\n\n{formatted_texts}"
                        
                        st.markdown(f"##### ⏳ Memproses Dokumen {i+1} sampai {i+len(batch)}...")
                        with st.container(border=True):
                            if AI_PROVIDER == "Mistral":
                                result_text = st.write_stream(stream_mistral(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                            elif AI_PROVIDER == "Google Gemini":
                                result_text = st.write_stream(stream_gemini(system_prompt, user_prompt, AI_API_KEY, AI_MODEL))
                            
                            if "❌ Error" not in result_text:
                                full_report_text += f"\n\n--- Bagian {i//BATCH_SIZE + 1} ---\n\n" + result_text

                    if full_report_text:
                        st.success("✅ Analisis Batch Selesai!")
                        st.download_button("📥 Download Laporan Lengkap", data=full_report_text, file_name="laporan_analisis_ai_lengkap.txt", mime="text/plain", type="primary")

        # --- TAB 3: GRAFIK TREN ---
        with tab3:
            st.markdown("### Visualisasi Data & Tren")
            col_kw, col_cite = st.columns(2)
            with col_kw:
                text_col_to_analyze = abstract_col if abstract_col else title_col
                if text_col_to_analyze:
                    st.markdown(f"##### 🔠 Top 10 Kata (Dari '{text_col_to_analyze}')")
                    st.bar_chart(get_top_words(data[text_col_to_analyze]))
                    
                    # PERBAIKAN FITUR WORD CLOUD: Peringatan Intuitif jika belum terinstal
                    if HAS_WORDCLOUD:
                        st.markdown("---")
                        st.markdown(f"##### ☁️ Awan Kata (Word Cloud)")
                        with st.spinner("Membuat Word Cloud..."):
                            text_for_cloud = " ".join(data[text_col_to_analyze].dropna().astype(str).str.lower())
                            filtered_words_cloud = " ".join([w for w in re.findall(r'\b[a-z]{4,}\b', text_for_cloud) if w not in COMMON_STOPWORDS])
                            if filtered_words_cloud:
                                wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2', max_words=100).generate(filtered_words_cloud)
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis("off")
                                st.pyplot(fig)
                            else:
                                st.info("Tidak cukup kata untuk membuat Word Cloud.")
                    else:
                        st.markdown("---")
                        st.error("💡 Modul 'wordcloud' dan 'matplotlib' belum terinstal di sistem Anda.")
                        st.markdown("Fitur Awan Kata tidak dapat ditampilkan. Silakan buka terminal/Command Prompt Anda dan jalankan perintah berikut untuk menginstalnya:")
                        st.code("pip install wordcloud matplotlib", language="bash")
                        st.info("Setelah berhasil diinstal, muat ulang (refresh/restart) aplikasi Streamlit ini.")
                        
            with col_cite:
                if citation_col and title_col:
                    st.markdown("##### 🏆 Top 5 Dokumen Paling Banyak Dikutip")
                    data[citation_col] = pd.to_numeric(data[citation_col], errors='coerce').fillna(0).astype(int)
                    top_cited = data.sort_values(by=citation_col, ascending=False).head(5)
                    display_cols = [title_col, citation_col]
                    if year_col: display_cols.append(year_col)
                    st.dataframe(top_cited[display_cols].reset_index(drop=True), use_container_width=True)
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                if year_col:
                    st.markdown("##### 📈 Tren Publikasi per Tahun")
                    st.area_chart(data[year_col].value_counts().sort_index()) 
            with col_chart2:
                if journal_col:
                    st.markdown("##### 📖 Top 10 Jurnal Produktif")
                    st.bar_chart(data[journal_col].value_counts().head(10))
            if author_col:
                st.markdown("---")
                st.markdown("##### 👤 Top 10 Penulis Paling Berkontribusi")
                st.bar_chart(data[author_col].value_counts().head(10))

        # --- TAB 4: DATA CLEANING (OPENREFINE STYLE) ---
        with tab4:
            st.markdown("### 🧹 Pembersihan Data Interaktif & Bantuan AI")
            
            col_u, col_r, col_stat = st.columns([1, 1, 4])
            with col_u:
                if st.button("↩️ Undo", disabled=(st.session_state.current_step == 0), use_container_width=True):
                    st.session_state.current_step -= 1
                    st.session_state.preview_action = None
                    st.rerun()
            with col_r:
                if st.button("↪️ Redo", disabled=(st.session_state.current_step == len(st.session_state.history) - 1), use_container_width=True):
                    st.session_state.current_step += 1
                    st.session_state.preview_action = None
                    st.rerun()
            with col_stat:
                st.info(f"**Riwayat Aktif:** {st.session_state.history_actions[st.session_state.current_step]} (Langkah {st.session_state.current_step + 1} dari {len(st.session_state.history)})")
                
                # FITUR BARU: Export Log Riwayat (Macro)
                if len(st.session_state.history_actions) > 1:
                    log_data = json.dumps(st.session_state.history_actions, indent=2)
                    st.download_button("💾 Ekspor Resep Riwayat (JSON)", data=log_data, file_name="resep_pembersihan.json", mime="application/json")
            
            st.markdown("---")

            text_cols = base_data.select_dtypes(include=['object']).columns.tolist()
            exclude_cols = [c for c in [title_col, abstract_col, 'DOI'] if c]
            cleanable_cols = [c for c in text_cols if c not in exclude_cols]
            
            if not cleanable_cols:
                st.warning("⚠️ Tidak ada kolom berbasis teks yang bisa dibersihkan.")
            else:
                col_c1, col_c2 = st.columns([1, 2])
                with col_c1:
                    target_clean_col = st.selectbox("Pilih Kolom Target:", cleanable_cols)
                    
                    with st.expander("📊 Pratinjau Distribusi Nilai (Text Facet)", expanded=False):
                        st.caption("Lihat variasi unik apa saja yang paling sering muncul di kolom ini.")
                        facet_df = base_data[target_clean_col].value_counts().reset_index()
                        facet_df.columns = ['Nilai Unik', 'Jumlah Kemunculan']
                        st.dataframe(facet_df, height=200, use_container_width=True)

                    st.markdown("---")
                    
                    # FITUR BARU: Split Cells / Pecah Baris multi-nilai
                    st.markdown("**✂️ Pecah Sel Multi-Nilai (Split Cells):**")
                    st.caption("Memisahkan nilai yang digabung (misal: 'AI; Machine Learning') menjadi baris terpisah untuk klastering yang lebih akurat.")
                    col_sd1, col_sd2 = st.columns([3, 1])
                    with col_sd1:
                        split_delim = st.selectbox("Pilih Pemisah:", ["Titik Koma (;)", "Koma (,)", "Pemisah Garis Lurus (|)"], label_visibility="collapsed")
                    with col_sd2:
                        if st.button("Pecah", use_container_width=True):
                            actual_delim = { "Titik Koma (;)": ";", "Koma (,)": ",", "Pemisah Garis Lurus (|)": "|" }.get(split_delim)
                            def split_cells(df):
                                df_out = df.copy()
                                df_out[target_clean_col] = df_out[target_clean_col].astype(str).str.split(actual_delim)
                                df_out = df_out.explode(target_clean_col)
                                df_out[target_clean_col] = df_out[target_clean_col].str.strip()
                                df_out = df_out[df_out[target_clean_col] != ""]
                                return df_out
                            apply_transform(split_cells, f"Split Cells '{target_clean_col}' dengan '{actual_delim}'", is_row_filter=True, target_col=target_clean_col)

                    st.markdown("---")
                    st.markdown("**✨ Transformasi Teks Instan:**")
                    col_case1, col_case2, col_case3 = st.columns(3)
                    
                    if col_case1.button("UPPER", use_container_width=True):
                        apply_transform(lambda col: col.astype(str).str.upper(), f"Format UPPERCASE pada '{target_clean_col}'", target_col=target_clean_col)
                    if col_case2.button("lower", use_container_width=True):
                        apply_transform(lambda col: col.astype(str).str.lower(), f"Format lowercase pada '{target_clean_col}'", target_col=target_clean_col)
                    if col_case3.button("Title", use_container_width=True):
                        apply_transform(lambda col: col.astype(str).str.title(), f"Format Title Case pada '{target_clean_col}'", target_col=target_clean_col)

                    col_trim, col_blank = st.columns(2)
                    with col_trim:
                        if st.button("✂️ Trim Spasi Ekstra", use_container_width=True):
                            apply_transform(lambda col: col.astype(str).apply(lambda x: " ".join(x.split()) if isinstance(x, str) else x), f"Trim & Squish pada '{target_clean_col}'", target_col=target_clean_col)
                    with col_blank:
                        if st.button("🗑️ Hapus Baris Kosong", use_container_width=True):
                            def remove_blank_rows(df):
                                mask = df[target_clean_col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "tidak tersedia"])
                                return df[~mask]
                            apply_transform(remove_blank_rows, f"Hapus Baris Kosong di '{target_clean_col}'", is_row_filter=True, target_col=target_clean_col)
                    
                    st.markdown("**🔍 Temukan & Ganti (Find & Replace):**")
                    col_f, col_r = st.columns(2)
                    with col_f:
                        find_txt = st.text_input("Cari teks:", placeholder="Contoh: US")
                    with col_r:
                        replace_txt = st.text_input("Ganti dengan:", placeholder="Contoh: USA")
                    if st.button("Terapkan Ganti (Replace)", use_container_width=True):
                        if find_txt:
                            apply_transform(lambda col: col.astype(str).str.replace(find_txt, replace_txt, regex=False), f"Find & Replace '{find_txt}' di '{target_clean_col}'", target_col=target_clean_col)
                        else:
                            st.warning("Masukkan teks yang ingin dicari.")

                    st.markdown("---")
                    st.markdown("**🧠 Clustering Lanjutan:**")
                    method = st.selectbox("Metode Clustering:", ["Fingerprint", "N-Gram (2-gram)", "Phonetic (Soundex)", "Levenshtein Distance", "PPM (Compression Distance)"])
                    delimiter_opt = st.selectbox("Karakter Pemisah / Penggabung:", ["Titik Koma (;)", "Koma (,)", "Pemisah Garis Lurus (|)", "Tidak Ada"])
                    used_delim = {"Titik Koma (;)": ";", "Koma (,)": ",", "Pemisah Garis Lurus (|)": "|"}.get(delimiter_opt, None)
                    
                    lev_threshold, ppm_threshold = 2, 0.3
                    if method == "Levenshtein Distance": lev_threshold = st.slider("Toleransi Levenshtein:", 1, 5, 2)
                    elif method == "PPM (Compression Distance)": ppm_threshold = st.slider("Toleransi NCD/PPM:", 0.1, 0.9, 0.3, 0.05)
                    elif method == "Phonetic (Soundex)": st.info("Kelompokkan berdasarkan kemiripan PENGUCAPAN suara.")
                    
                    use_ai_suggestion = st.checkbox("🤖 Minta Pertimbangan AI (Cek Makna)", help="Kirim klaster yang terdeteksi ke AI untuk mengevaluasi apakah maknanya sama.")
                    
                    if st.button("🔍 Temukan Klaster", type="primary", use_container_width=True):
                        if use_ai_suggestion and not AI_API_KEY:
                            st.error(f"⚠️ {AI_PROVIDER} API Key diperlukan untuk fitur ini. Masukkan di panel sebelah kiri.")
                        else:
                            with st.spinner(f"Mengekstrak klaster pada kolom '{target_clean_col}'..."):
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
                                            elif method == "PPM (Compression Distance)" and ppm_distance(kw1.lower(), kw2.lower()) <= ppm_threshold: match = True
                                            if match:
                                                current_cluster.append(kw2)
                                                visited.add(kw2)
                                        if len(current_cluster) > 1:
                                            valid_clusters[f"Cluster_{cluster_id}"] = current_cluster
                                            cluster_id += 1
                                    progress_bar.empty()
                                
                                if not valid_clusters:
                                    st.session_state.clustering_result = None
                                    st.success("🎉 Tidak ada klaster duplikat ditemukan!")
                                else:
                                    ai_suggestions = {}
                                    
                                    if use_ai_suggestion:
                                        total_clusters = len(valid_clusters)
                                        st.info(f"⏳ Mengevaluasi seluruh {total_clusters} klaster dengan AI. Waktu tunggu tidak dibatasi, silakan main game sambil menunggu.")
                                        
                                        # FITUR BARU: Mini Game Flappy Bird HTML5 sebagai loading hiburan
                                        flappy_bird_html = """
                                        <!DOCTYPE html>
                                        <html>
                                        <head>
                                        <style>
                                          body { background-color: #70c5ce; color: #fff; font-family: 'Courier New', Courier, monospace; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; overflow: hidden; }
                                          canvas { background-color: #70c5ce; border: 4px solid #fff; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.3); }
                                          .ui { position: absolute; top: 20px; text-align: center; pointer-events: none; width: 100%; }
                                          .score { font-size: 40px; font-weight: bold; text-shadow: 2px 2px #000; }
                                          .msg { font-size: 18px; text-shadow: 1px 1px #000; margin-top: 5px; }
                                          #restartBtn { 
                                              display: none; 
                                              position: absolute; 
                                              top: 60%; 
                                              padding: 10px 20px; 
                                              font-size: 20px; 
                                              background: #f44336; 
                                              color: white; 
                                              border: none; 
                                              border-radius: 5px; 
                                              cursor: pointer; 
                                              box-shadow: 0 4px #d32f2f;
                                              pointer-events: auto;
                                          }
                                          #restartBtn:active { box-shadow: 0 0; transform: translateY(4px); }
                                        </style>
                                        </head>
                                        <body onclick="jump()" onkeydown="if(event.keyCode==32) jump()">
                                          <div class="ui">
                                            <div class="score" id="score">0</div>
                                            <div class="msg" id="msg">Klik atau Spasi untuk Terbang</div>
                                          </div>
                                          <canvas id="gameCanvas" width="320" height="480"></canvas>
                                          <button id="restartBtn" onclick="resetGame()">Restart Game</button>

                                          <script>
                                            const canvas = document.getElementById("gameCanvas");
                                            const ctx = canvas.getContext("2d");
                                            const scoreEl = document.getElementById("score");
                                            const msgEl = document.getElementById("msg");
                                            const restartBtn = document.getElementById("restartBtn");

                                            let bird, pipes, score, gameRunning, frames;
                                            const gravity = 0.25;
                                            const jumpForce = -4.5;
                                            const pipeWidth = 50;
                                            const pipeGap = 120;

                                            function init() {
                                                bird = { x: 50, y: 150, v: 0, r: 12 };
                                                pipes = [];
                                                score = 0;
                                                frames = 0;
                                                gameRunning = true;
                                                scoreEl.innerText = score;
                                                msgEl.innerText = "Klik atau Spasi untuk Terbang";
                                                restartBtn.style.display = "none";
                                                loop();
                                            }

                                            function jump() {
                                                if(gameRunning) bird.v = jumpForce;
                                                else if(restartBtn.style.display === "block") resetGame();
                                            }

                                            function resetGame() {
                                                init();
                                            }

                                            function loop() {
                                                if(!gameRunning) return;
                                                update();
                                                draw();
                                                requestAnimationFrame(loop);
                                            }

                                            function update() {
                                                frames++;
                                                bird.v += gravity;
                                                bird.y += bird.v;

                                                if(bird.y + bird.r > canvas.height || bird.y - bird.r < 0) gameOver();

                                                if(frames % 100 === 0) {
                                                    let h = Math.floor(Math.random() * (canvas.height - pipeGap - 100)) + 50;
                                                    pipes.push({ x: canvas.width, top: h });
                                                }

                                                pipes.forEach((p, i) => {
                                                    p.x -= 2;
                                                    if(p.x + pipeWidth < 0) {
                                                        pipes.splice(i, 1);
                                                        score++;
                                                        scoreEl.innerText = score;
                                                    }
                                                    // Collision
                                                    if(bird.x + bird.r > p.x && bird.x - bird.r < p.x + pipeWidth) {
                                                        if(bird.y - bird.r < p.top || bird.y + bird.r > p.top + pipeGap) gameOver();
                                                    }
                                                });
                                            }

                                            function draw() {
                                                ctx.clearRect(0,0, canvas.width, canvas.height);
                                                
                                                // Bird
                                                ctx.fillStyle = "#f1c40f";
                                                ctx.beginPath();
                                                ctx.arc(bird.x, bird.y, bird.r, 0, Math.PI*2);
                                                ctx.fill();
                                                ctx.strokeStyle = "#000";
                                                ctx.stroke();

                                                // Pipes
                                                ctx.fillStyle = "#2ecc71";
                                                pipes.forEach(p => {
                                                    ctx.fillRect(p.x, 0, pipeWidth, p.top);
                                                    ctx.strokeRect(p.x, 0, pipeWidth, p.top);
                                                    ctx.fillRect(p.x, p.top + pipeGap, pipeWidth, canvas.height);
                                                    ctx.strokeRect(p.x, p.top + pipeGap, pipeWidth, canvas.height);
                                                });
                                            }

                                            function gameOver() {
                                                gameRunning = false;
                                                msgEl.innerText = "GAME OVER!";
                                                restartBtn.style.display = "block";
                                            }

                                            init();
                                          </script>
                                        </body>
                                        </html>
                                        """
                                        
                                        # Buat penampung kosong untuk Game
                                        game_placeholder = st.empty()
                                        
                                        with game_placeholder.container():
                                            st.markdown("### 🎮 Sambil Menunggu, Main Flappy Bird!")
                                            st.caption("Klik pada area game atau gunakan Spasi untuk melompat.")
                                            components.html(flappy_bird_html, height=550)
                                            
                                        # Proses pemanggilan AI di latar belakang
                                        with st.spinner(f"🤖 {AI_MODEL} sedang mengevaluasi {total_clusters} variasi (Mode JSON Strict)..."):
                                            prompt_clusters = {}
                                            for idx, (key, items) in enumerate(valid_clusters.items()):
                                                prompt_clusters[f"Cluster_{idx}"] = items
                                            
                                            sys_prompt_ai = """Anda adalah pakar data bibliometrik. Evaluasi setiap klaster kata berikut.
Tentukan apakah variasi kata dalam satu klaster benar-benar bermakna/berentitas sama dan harus digabungkan.
Pilih atau buat satu kata 'standar' yang paling profesional untuk variasi tersebut.

PENTING: Output Anda HARUS murni JSON dengan skema seperti berikut:
{
  "Cluster_X": {
    "gabung": true/false,
    "standar": "Nama Standar Profesional",
    "alasan": "Alasan singkat"
  }
}"""
                                            usr_prompt_ai = f"Data Klaster:\n{json.dumps(prompt_clusters, indent=2)}"
                                            
                                            if AI_PROVIDER == "Mistral":
                                                ai_response = call_mistral(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                            elif AI_PROVIDER == "Google Gemini":
                                                ai_response = call_gemini(sys_prompt_ai, usr_prompt_ai, AI_API_KEY, AI_MODEL)
                                                
                                            try:
                                                match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                                                if match:
                                                    ai_suggestions = json.loads(match.group(0))
                                                else:
                                                    ai_suggestions = json.loads(ai_response)
                                            except Exception as e:
                                                st.warning(f"⚠️ Gagal memproses JSON dari AI. Menggunakan pengaturan bawaan. Detail: {e}")

                                        # Setelah proses AI selesai, hancurkan/hapus tempat game agar menghilang dari layar
                                        game_placeholder.empty()

                                    cluster_data = []
                                    for idx, (key, items) in enumerate(valid_clusters.items()):
                                        c_id = f"Cluster_{idx}"
                                        standard = max(items, key=lambda x: kw_freq[x]) 
                                        gabung = True
                                        alasan = "-"
                                        
                                        if ai_suggestions and c_id in ai_suggestions:
                                            sugg = ai_suggestions[c_id]
                                            gabung = sugg.get("gabung", True)
                                            standar_ai = sugg.get("standar", standard)
                                            if standar_ai: standard = standar_ai
                                            alasan = sugg.get("alasan", "-")

                                        for item in items:
                                            # Memastikan jika Variasi Asli sama dengan Nilai Baru, maka selalu diceklis
                                            item_gabung = True if item == standard else gabung
                                            
                                            cluster_data.append({
                                                "Klaster": f"Klaster {idx+1}",
                                                "Gabung?": item_gabung,
                                                "Variasi Asli": item,
                                                "Nilai Baru (Edit Disini)": standard,
                                                "Frekuensi": kw_freq[item],
                                                "Saran AI": alasan
                                            })
                                        
                                    st.session_state.clustering_result = pd.DataFrame(cluster_data)
                                    st.session_state.target_clean_col = target_clean_col
                                    st.session_state.used_delim = used_delim

                # AREA TAMPILAN PRATINJAU & EDITOR KLASTERING
                with col_c2:
                    st.download_button(
                        label="📥 Download Data Saat Ini (.csv)",
                        data=convert_df_to_csv(base_data),
                        file_name="scopus_dataset_cleaned.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.markdown("---")
                    
                    if st.session_state.get('preview_action'):
                        st.success(f"✅ **Tindakan Berhasil:** {st.session_state.preview_action}")
                        st.markdown("##### 🔍 Pratinjau Perubahan (Sebelum vs Sesudah)")
                        orig_col = st.session_state.preview_original
                        new_col = st.session_state.preview_new
                        
                        if orig_col is not None and new_col is not None:
                            # PERBAIKAN FITUR SPLIT CELLS: Deteksi yang lebih cerdas ketika panjang baris berubah
                            if len(orig_col) != len(new_col):
                                if len(new_col) > len(orig_col):
                                    st.info(f"Berhasil dipecah (Split)! Jumlah baris bertambah dari **{len(orig_col)}** menjadi **{len(new_col)}** baris. Berikut cuplikannya:")
                                    preview_df = pd.DataFrame({"Hasil Pecahan (Baru)": new_col.head(10)})
                                    st.dataframe(preview_df, use_container_width=True)
                                else:
                                    deleted_idx = orig_col.index.difference(new_col.index)
                                    st.info(f"Terdapat **{len(deleted_idx)}** baris yang dihapus. Berikut cuplikannya:")
                                    preview_df = pd.DataFrame({"Baris yang Dihapus (Original)": orig_col.loc[deleted_idx]})
                                    st.dataframe(preview_df.head(), use_container_width=True)
                            else:
                                try:
                                    changed_mask = orig_col != new_col
                                    changed_idx = changed_mask[changed_mask].index
                                    if len(changed_idx) > 0:
                                        st.info(f"Terdapat **{len(changed_idx)}** baris yang berubah. Berikut cuplikannya:")
                                        preview_df = pd.DataFrame({"Sebelum (Original)": orig_col.loc[changed_idx[:5]], "Sesudah (Baru)": new_col.loc[changed_idx[:5]]})
                                        st.dataframe(preview_df, use_container_width=True)
                                    else:
                                        st.info("Tidak ada perubahan spesifik yang terdeteksi, atau teks sudah sesuai.")
                                except ValueError:
                                    st.info("💡 Struktur baris telah berubah (dipecah). Lihat data Anda di tabel utama.")
                        
                        if st.button("Tutup Pratinjau ✖️", use_container_width=False):
                            st.session_state.preview_action = None
                            st.rerun()
                        st.markdown("---")
                    
                    if st.session_state.clustering_result is not None:
                        st.write(f"⚠️ Ditemukan **{len(st.session_state.clustering_result['Klaster'].unique())}** klaster pada kolom **{st.session_state.target_clean_col}**.")
                        st.info("💡 **Petunjuk:** Hapus centang pada variasi spesifik yang TIDAK ingin Anda gabungkan. Anda juga bisa mengedit 'Nilai Baru'.")
                        
                        edited_df = st.data_editor(
                            st.session_state.clustering_result,
                            column_config={
                                "Klaster": st.column_config.TextColumn("Kelompok", disabled=True),
                                "Gabung?": st.column_config.CheckboxColumn("Merge?", default=True),
                                "Variasi Asli": st.column_config.TextColumn("Variasi Asli", disabled=True),
                                "Nilai Baru (Edit Disini)": st.column_config.TextColumn("Nilai Baru (New Cell Value)"),
                                "Frekuensi": st.column_config.NumberColumn("Muncul", disabled=True),
                                "Saran AI": st.column_config.TextColumn("Alasan / Analisis AI", disabled=True)
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        if st.button("Terapkan Penggabungan (Apply & Merge)", type="primary"):
                            mapping = {}
                            for index, row in edited_df.iterrows():
                                if row["Gabung?"] == True:
                                    original_val = row["Variasi Asli"]
                                    new_val = row["Nilai Baru (Edit Disini)"]
                                    mapping[original_val.strip()] = new_val
                            
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
                                apply_transform(cluster_apply, f"Penggabungan Klaster pada '{st.session_state.target_clean_col}'", target_col=st.session_state.target_clean_col)
                            else:
                                st.warning("Anda tidak memilih klaster satupun untuk digabung.")
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 Silakan gunakan panel di sebelah kiri untuk mengatur API Key, Penyedia AI, dan memasukkan data CSV/JSON Anda.")