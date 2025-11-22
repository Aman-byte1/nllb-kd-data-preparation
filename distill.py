import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import os
import logging
import yaml
import gc
import argparse
import time
from datasets import load_dataset
from huggingface_hub import login, HfApi, hf_hub_download
from comet import download_model, load_from_checkpoint

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

ISO_TO_NLLB = {
    'en': 'eng_Latn', 'fr': 'fra_Latn', 'ar': 'arb_Arab', 'es': 'spa_Latn',
    'pt': 'por_Latn', 'am': 'amh_Ethi', 'sw': 'swh_Latn', 'ha': 'hau_Latn',
    'zu': 'zul_Latn', 'so': 'som_Latn', 'yo': 'yor_Latn', 'af': 'afr_Latn',
    'arz': 'arz_Arab', 'wo': 'wol_Latn', 'ln': 'lin_Latn'
}

# ==========================================
# CLASS: RESEARCH STATISTICS TRACKER
# ==========================================
class ResearchStats:
    def __init__(self):
        self.stats = []

    def log(self, pair, original, after_heuristics, after_comet, final):
        self.stats.append({
            "Pair": pair,
            "Original_Rows": original,
            "Valid_Heuristics": after_heuristics,
            "Valid_Quality(QE)": after_comet,
            "Final_Strict_Count": final,
            "Yield_Rate_%": round((final / original * 100), 2) if original > 0 else 0
        })

    def print_report(self):
        print("\n" + "="*80)
        print("🧬 RESEARCH DATA GENERATION REPORT 🧬")
        print("="*80)
        df_stats = pd.DataFrame(self.stats)
        if not df_stats.empty:
            # Calculate total loss metrics
            df_stats['Lost_to_Heuristics'] = df_stats['Original_Rows'] - df_stats['Valid_Heuristics']
            df_stats['Lost_to_Quality'] = df_stats['Valid_Heuristics'] - df_stats['Valid_Quality(QE)']
            df_stats['Lost_to_StrictNull'] = df_stats['Valid_Quality(QE)'] - df_stats['Final_Strict_Count']
            
            print(df_stats.to_string(index=False))
            print("-" * 80)
            print(f"TOTAL GENERATED ROWS: {df_stats['Final_Strict_Count'].sum()}")
            print("-" * 80)
        else:
            print("No data processed.")

stats_tracker = ResearchStats()

# ==========================================
# UTILITIES
# ==========================================
def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def batch_generate(model, tokenizer, texts, tgt_lang_id, batch_size, beam_size=1):
    indices = sorted(range(len(texts)), key=lambda k: len(texts[k]), reverse=True)
    sorted_texts = [texts[i] for i in indices]
    results = [None] * len(texts)

    for i in range(0, len(sorted_texts), batch_size):
        batch = sorted_texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, forced_bos_token_id=tgt_lang_id, max_length=512, num_beams=beam_size, early_stopping=(beam_size > 1))
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j, txt in enumerate(decoded): results[indices[i+j]] = txt
        except Exception as e:
            logger.error(f"Generation Error: {e}")
            clean_vram()
            for j in range(len(batch)): results[indices[i+j]] = "" # Mark as empty (will be filtered later)
                
    return results

# ==========================================
# STEP 1: TRANSLATION
# ==========================================
def run_translation_step(config, pair):
    src_short, tgt_short = pair
    src_code = ISO_TO_NLLB.get(src_short, src_short)
    tgt_code = ISO_TO_NLLB.get(tgt_short, tgt_short)
    
    output_file = os.path.join(config['paths']['output_dir'], f"raw_{src_code}-{tgt_code}.csv")
    
    # RESUME LOGIC
    start_index = 0
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            start_index = len(existing_df)
            logger.info(f"Resuming {src_code}-{tgt_code} from row {start_index}")
        except:
            start_index = 0

    # Load Data
    ds = load_dataset(config['dataset']['hf_dataset'], split=config['dataset']['hf_split'])
    df = ds.to_pandas()
    mask = (df['src_lang'].isin([src_short, src_code])) & (df['tgt_lang'].isin([tgt_short, tgt_code]))
    df = df[mask]
    col_map = {'source_text': 'source', 'src': 'source', 'target_text': 'target', 'tgt': 'target'}
    df.rename(columns=col_map, inplace=True)
    
    if config['dataset']['max_rows']: df = df.head(config['dataset']['max_rows'])
    
    if len(df) <= start_index:
        return output_file, src_code, tgt_code, len(df)

    df_remaining = df.iloc[start_index:].copy()
    
    if start_index == 0:
        cols = ['original_source', 'original_target', 'translated_source_to_target', 'translated_target_to_source', 'src_lang_code', 'tgt_lang_code']
        pd.DataFrame(columns=cols).to_csv(output_file, index=False)

    # Load Model
    logger.info("Loading NLLB...")
    dtype = torch.float32 if config['model']['dtype'] == "float32" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['name'], torch_dtype=dtype, device_map="auto").eval()

    tgt_token_id_fwd = tokenizer.convert_tokens_to_ids(tgt_code)
    tgt_token_id_bwd = tokenizer.convert_tokens_to_ids(src_code)
    
    chunk_size = config['translation']['chunk_size']
    batch_size = config['translation']['batch_size']

    for i in tqdm(range(0, len(df_remaining), chunk_size), desc="Translating"):
        chunk = df_remaining.iloc[i:i+chunk_size].copy()
        srcs = chunk['source'].astype(str).tolist()
        tgts = chunk['target'].astype(str).tolist()
        
        tokenizer.src_lang = src_code
        trans_s2t = batch_generate(model, tokenizer, srcs, tgt_token_id_fwd, batch_size)
        
        tokenizer.src_lang = tgt_code
        trans_t2s = batch_generate(model, tokenizer, tgts, tgt_token_id_bwd, batch_size)
        
        save_df = pd.DataFrame({
            'original_source': srcs, 'original_target': tgts,
            'translated_source_to_target': trans_s2t, 'translated_target_to_source': trans_t2s,
            'src_lang_code': src_code, 'tgt_lang_code': tgt_code
        })
        save_df.to_csv(output_file, mode='a', header=False, index=False)

    del model, tokenizer
    clean_vram()
    return output_file, src_code, tgt_code, len(df)

# ==========================================
# STEP 2: FILTERING & STATS
# ==========================================
def run_filtering_step(config, raw_file, original_count, pair_name):
    if not raw_file or not os.path.exists(raw_file): return None
    
    logger.info(f"Filtering {pair_name}...")
    df = pd.read_csv(raw_file)
    
    # 1. Heuristics (CPU)
    # We create a mask but don't drop yet, to count properly
    def heuristic_check(src, mt):
        if pd.isna(mt) or str(mt).strip() == "": return False
        if len(str(src)) == 0: return False
        ratio = len(str(mt).split()) / max(1, len(str(src).split()))
        return (config['filtering']['min_len_ratio'] < ratio < config['filtering']['max_len_ratio'])

    df['check_s2t'] = df.apply(lambda x: heuristic_check(x['original_source'], x['translated_source_to_target']), axis=1)
    df['check_t2s'] = df.apply(lambda x: heuristic_check(x['original_target'], x['translated_target_to_source']), axis=1)
    
    # Keep row if BOTH passed heuristics (to save compute on COMET)
    # Note: User asked for strict filtering, but we can check specific directions for COMET first
    valid_heuristic_rows = df[(df['check_s2t']) & (df['check_t2s'])].copy()
    count_after_heuristics = len(valid_heuristic_rows)
    
    if valid_heuristic_rows.empty:
        stats_tracker.log(pair_name, original_count, 0, 0, 0)
        return None

    # 2. COMET (GPU)
    logger.info("Loading COMET...")
    qe_model_path = download_model(config['filtering']['qe_model'])
    qe_model = load_from_checkpoint(qe_model_path)
    qe_model.eval().cuda()
    
    # Score S2T
    inputs_s2t = [{"src": str(r['original_source']), "mt": str(r['translated_source_to_target'])} for _, r in valid_heuristic_rows.iterrows()]
    scores_s2t = qe_model.predict(inputs_s2t, batch_size=config['filtering']['batch_size'], gpus=1, progress_bar=True).scores
    
    # Score T2S
    inputs_t2s = [{"src": str(r['original_target']), "mt": str(r['translated_target_to_source'])} for _, r in valid_heuristic_rows.iterrows()]
    scores_t2s = qe_model.predict(inputs_t2s, batch_size=config['filtering']['batch_size'], gpus=1, progress_bar=True).scores
    
    del qe_model
    clean_vram()
    
    valid_heuristic_rows['score_s2t'] = scores_s2t
    valid_heuristic_rows['score_t2s'] = scores_t2s
    
    # 3. Quality Filter
    threshold = config['filtering']['qe_threshold']
    valid_quality = valid_heuristic_rows[
        (valid_heuristic_rows['score_s2t'] >= threshold) & 
        (valid_heuristic_rows['score_t2s'] >= threshold)
    ].copy()
    
    count_after_quality = len(valid_quality)
    
    # 4. FINAL STRICT NULL/NAN DROP
    # Even if it passed scoring, ensure no column is NaN
    final_df = valid_quality.dropna(how='any')
    count_final = len(final_df)

    # Cleanup columns
    final_df = final_df.drop(columns=['check_s2t', 'check_t2s', 'score_s2t', 'score_t2s'])
    
    # Log Stats
    stats_tracker.log(pair_name, original_count, count_after_heuristics, count_after_quality, count_final)
    
    if final_df.empty: return None
    
    return final_df

# ==========================================
# STEP 3: APPEND & UPLOAD
# ==========================================
def append_and_upload(config, new_df, repo_id, api):
    combined_filename = "combined_dataset.parquet"
    local_combined_path = os.path.join(config['paths']['output_dir'], combined_filename)
    
    # 1. Try to retrieve existing data from Local or Repo
    master_df = pd.DataFrame()
    
    # Check local first
    if os.path.exists(local_combined_path):
        logger.info("Loading local master dataset...")
        master_df = pd.read_parquet(local_combined_path)
    else:
        # Check remote (in case we switched machines)
        try:
            logger.info("Checking for existing remote dataset...")
            path = hf_hub_download(repo_id=repo_id, filename=combined_filename, repo_type="dataset")
            master_df = pd.read_parquet(path)
        except:
            logger.info("No existing dataset found. Creating new.")

    # 2. Concatenate
    logger.info(f"Appending {len(new_df)} rows to master (current size: {len(master_df)})...")
    if not master_df.empty:
        # Ensure columns match
        new_df = new_df[master_df.columns]
        master_df = pd.concat([master_df, new_df], ignore_index=True)
    else:
        master_df = new_df
        
    # 3. Save Local Master
    master_df.to_parquet(local_combined_path, index=False)
    
    # 4. Upload
    logger.info(f"⬆️ Uploading updated master file ({len(master_df)} total rows)...")
    api.upload_file(
        path_or_fileobj=local_combined_path,
        path_in_repo=combined_filename,
        repo_id=repo_id,
        repo_type="dataset"
    )
    logger.info("✅ Upload Complete.")

# ==========================================
# MAIN
# ==========================================
def main(config_path):
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    login(token=config['huggingface']['token'])
    api = HfApi()
    repo_id = f"{config['huggingface']['username']}/{config['huggingface']['repo_name']}"
    api.create_repo(repo_id=repo_id, private=config['huggingface']['private'], exist_ok=True, repo_type="dataset")

    for pair in config['language_pairs']:
        pair_str = f"{pair[0]}-{pair[1]}"
        try:
            # 1. Translate
            raw_csv, src_code, tgt_code, orig_count = run_translation_step(config, pair)
            
            # 2. Filter (Strict) & Get Stats
            clean_df = run_filtering_step(config, raw_csv, orig_count, pair_str)
            
            if clean_df is not None and not clean_df.empty:
                # 3. Append to Master File & Upload Immediately
                append_and_upload(config, clean_df, repo_id, api)
            
        except Exception as e:
            logger.error(f"Critical error on {pair_str}: {e}")
            clean_vram()

    # 4. Final Research Report
    stats_tracker.print_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
