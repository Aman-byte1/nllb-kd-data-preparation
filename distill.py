import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import argparse
from datasets import load_dataset
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'translation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class NLLBTranslator:
    def __init__(self, model_name="facebook/nllb-200-3.3B", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize NLLB translator with the 3.3B model"""
        logging.info(f"Loading model {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.device = device
        
        # NLLB language codes mapping
        self.lang_codes = {
            'en': 'eng_Latn',
            'ar': 'arb_Arab',
            'am': 'amh_Ethi',
            'sw': 'swh_Latn',
            'ha': 'hau_Latn',
            'zu': 'zul_Latn',
            'so': 'som_Latn',
            'yo': 'yor_Latn',
            'af': 'afr_Latn',
            'arz': 'arz_Arab',
            'fr': 'fra_Latn',
            'es': 'spa_Latn',
            'pt': 'por_Latn',
            'wo': 'wol_Latn',
            'ln': 'lin_Latn'
        }
    
    def translate_batch(self, 
                       texts: List[str], 
                       src_lang: str, 
                       tgt_lang: str,
                       max_length: int = 512,
                       num_beams: int = 5,
                       batch_size: int = 8) -> List[str]:
        """Translate a batch of texts"""
        src_code = self.lang_codes.get(src_lang, src_lang)
        tgt_code = self.lang_codes.get(tgt_lang, tgt_lang)
        
        self.tokenizer.src_lang = src_code
        translations = []
        
        # Get the target language token ID
        # Handle both old and new tokenizer API
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            tgt_token_id = self.tokenizer.lang_code_to_id[tgt_code]
        else:
            # For newer versions, use convert_tokens_to_ids
            tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Process in smaller batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # Decode
            batch_translations = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            translations.extend(batch_translations)
        
        return translations

class DataProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Language pairs sorted by data size (ascending)
        self.language_pairs = [
            ('wo', 'fr', 9_071),
            ('yo', 'en', 22_626),
            ('wo', 'en', 31_170),
            ('zu', 'en', 33_189),
            ('so', 'en', 43_657),
            ('am', 'en', 71_800),
            ('arz', 'en', 84_170),
            ('ha', 'en', 155_881),
            ('af', 'en', 161_644),
            ('sw', 'en', 314_300),
            ('es', 'en', 1_324_681),
            ('pt', 'en', 1_401_671),
            ('ar', 'en', 1_424_237),
            ('fr', 'en', 1_483_951),
            # Reverse directions (you'll need to check actual counts)
            ('en', 'ar', None),
            ('en', 'am', None),
            ('en', 'sw', None),
            ('en', 'ha', None),
            ('en', 'zu', None),
            ('en', 'so', None),
            ('en', 'yo', None),
            ('en', 'af', None),
            ('en', 'arz', None),
            ('en', 'fr', None),
            ('en', 'es', None),
            ('en', 'pt', None),
            ('fr', 'wo', None),
            ('fr', 'ln', None),
        ]
    
    def load_data(self, src_lang: str, tgt_lang: str, max_rows: int = None, 
                  hf_dataset: str = None, hf_split: str = "train") -> pd.DataFrame:
        """Load training data for a language pair from local files or Hugging Face"""
        
        # Try loading from Hugging Face if dataset name is provided
        if hf_dataset:
            try:
                logging.info(f"Loading dataset '{hf_dataset}' from Hugging Face (split: {hf_split})")
                dataset = load_dataset(hf_dataset, split=hf_split)
                
                # Convert to DataFrame
                df = dataset.to_pandas()
                
                # Filter by language pair if dataset contains multiple pairs
                if 'src_lang' in df.columns and 'tgt_lang' in df.columns:
                    # Map language codes if needed (e.g., 'en' -> 'eng_Latn')
                    src_code = self._get_lang_code(src_lang)
                    tgt_code = self._get_lang_code(tgt_lang)
                    
                    df = df[
                        ((df['src_lang'] == src_code) & (df['tgt_lang'] == tgt_code)) |
                        ((df['src_lang'] == src_lang) & (df['tgt_lang'] == tgt_lang))
                    ]
                    logging.info(f"Filtered to {len(df)} rows for {src_lang}→{tgt_lang}")
                
                # Rename columns if needed
                if 'source' not in df.columns and 'src' in df.columns:
                    df.rename(columns={'src': 'source'}, inplace=True)
                if 'target' not in df.columns and 'tgt' in df.columns:
                    df.rename(columns={'tgt': 'target'}, inplace=True)
                
                if max_rows:
                    df = df.head(max_rows)
                
                logging.info(f"Loaded {len(df)} rows from Hugging Face dataset")
                return df
                
            except Exception as e:
                logging.warning(f"Failed to load from Hugging Face: {str(e)}")
                if hf_dataset and not any(os.path.exists(os.path.join(self.data_dir, p)) 
                                         for p in [f"{src_lang}-{tgt_lang}.csv",
                                                   f"{src_lang}_{tgt_lang}.csv",
                                                   f"{src_lang}-{tgt_lang}.tsv",
                                                   f"{src_lang}_{tgt_lang}.tsv",
                                                   f"{src_lang}-{tgt_lang}.json"]):
                    raise  # Re-raise if no local files available as fallback
        
        # Try loading from local files
        file_patterns = [
            f"{src_lang}-{tgt_lang}.csv",
            f"{src_lang}_{tgt_lang}.csv",
            f"{src_lang}-{tgt_lang}.tsv",
            f"{src_lang}_{tgt_lang}.tsv",
            f"{src_lang}-{tgt_lang}.json",
        ]
        
        for pattern in file_patterns:
            filepath = os.path.join(self.data_dir, pattern)
            if os.path.exists(filepath):
                logging.info(f"Loading data from {filepath}")
                
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath, nrows=max_rows)
                elif filepath.endswith('.tsv'):
                    df = pd.read_csv(filepath, sep='\t', nrows=max_rows)
                elif filepath.endswith('.json'):
                    df = pd.read_json(filepath, lines=True)
                    if max_rows:
                        df = df.head(max_rows)
                
                return df
        
        logging.warning(f"No data file found for {src_lang}-{tgt_lang}")
        return pd.DataFrame()
    
    def _get_lang_code(self, lang: str) -> str:
        """Convert language code to NLLB format if needed"""
        lang_codes = {
            'en': 'eng_Latn',
            'ar': 'arb_Arab',
            'am': 'amh_Ethi',
            'sw': 'swh_Latn',
            'ha': 'hau_Latn',
            'zu': 'zul_Latn',
            'so': 'som_Latn',
            'yo': 'yor_Latn',
            'af': 'afr_Latn',
            'arz': 'arz_Arab',
            'fr': 'fra_Latn',
            'es': 'spa_Latn',
            'pt': 'por_Latn',
            'wo': 'wol_Latn',
            'ln': 'lin_Latn'
        }
        return lang_codes.get(lang, lang)
    
    def filter_translations(self, 
                          df: pd.DataFrame,
                          min_length: int = 1,
                          max_length: int = 512,
                          remove_duplicates: bool = True) -> pd.DataFrame:
        """Filter translated data"""
        initial_count = len(df)
        
        # Remove empty translations
        df = df[df['translation'].str.len() > min_length]
        
        # Remove too long translations
        df = df[df['translation'].str.len() < max_length]
        
        # Remove duplicates if specified
        if remove_duplicates:
            df = df.drop_duplicates(subset=['source', 'translation'])
        
        # Remove translations that are identical to source
        df = df[df['source'] != df['translation']]
        
        final_count = len(df)
        logging.info(f"Filtered from {initial_count} to {final_count} rows")
        
        return df
    
    def calculate_statistics(self, df: pd.DataFrame, authentic_df: pd.DataFrame = None) -> Dict:
        """Calculate statistics for the translated data"""
        stats = {
            'total_rows': len(df),
            'avg_source_length': df['source'].str.len().mean(),
            'avg_translation_length': df['translation'].str.len().mean(),
            'unique_sources': df['source'].nunique(),
            'unique_translations': df['translation'].nunique(),
        }
        
        if authentic_df is not None:
            # Check overlaps with authentic dataset
            overlaps = df.merge(
                authentic_df, 
                left_on=['source', 'translation'],
                right_on=['source', 'translation'],
                how='inner'
            )
            stats['overlap_with_authentic'] = len(overlaps)
            stats['overlap_percentage'] = (len(overlaps) / len(df)) * 100
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='NLLB Knowledge Distillation Data Generation')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing source data')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for translation')
    parser.add_argument('--num_beams', type=int, default=None, help='Number of beams for beam search')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum rows to process per language pair')
    parser.add_argument('--authentic_data_dir', type=str, default=None, help='Directory with authentic data for overlap checking')
    parser.add_argument('--hf_dataset', type=str, default=None, help='Hugging Face dataset name')
    parser.add_argument('--hf_split', type=str, default=None, help='Dataset split to use')
    parser.add_argument('--src_lang', type=str, default=None, help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default=None, help='Target language code')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded config from {args.config}")
        else:
            logging.warning(f"Config file {args.config} not found, using defaults")
    
    # Merge command-line args with config (CLI takes precedence)
    def get_param(cli_value, config_path, default=None):
        """Get parameter value from CLI, config, or default"""
        if cli_value is not None:
            return cli_value
        # Navigate nested config dict
        parts = config_path.split('.')
        value = config
        for part in parts:
            value = value.get(part, {}) if isinstance(value, dict) else {}
        return value if value != {} else default
    
    # Extract parameters
    data_dir = get_param(args.data_dir, 'data.input_dir', './data')
    output_dir = get_param(args.output_dir, 'data.output_dir', './distilled_data')
    batch_size = get_param(args.batch_size, 'translation.batch_size', 8)
    num_beams = get_param(args.num_beams, 'translation.num_beams', 5)
    max_rows = get_param(args.max_rows, 'dataset.max_rows', None)
    hf_dataset = get_param(args.hf_dataset, 'dataset.hf_dataset', None)
    hf_split = get_param(args.hf_split, 'dataset.hf_split', 'train')
    authentic_data_dir = get_param(args.authentic_data_dir, 'authentic_data_dir', None)
    model_name = config.get('model', {}).get('name', 'facebook/nllb-200-3.3B')
    
    # Initialize translator and processor
    translator = NLLBTranslator(model_name=model_name)
    processor = DataProcessor(data_dir, output_dir)
    
    # Determine language pairs to process
    if args.src_lang and args.tgt_lang:
        # Specific pair from CLI
        language_pairs = [(args.src_lang, args.tgt_lang, max_rows)]
        logging.info(f"Processing specific language pair: {args.src_lang}→{args.tgt_lang}")
    elif config.get('language_pairs'):
        # From config file - handle both [src, tgt] and [src, tgt, max_rows] formats
        language_pairs = []
        for pair in config['language_pairs']:
            if len(pair) == 2:
                # [src, tgt] format - use global max_rows
                language_pairs.append((pair[0], pair[1], max_rows))
            elif len(pair) == 3:
                # [src, tgt, max_rows] format - use per-pair max_rows
                language_pairs.append((pair[0], pair[1], pair[2]))
        logging.info(f"Processing {len(language_pairs)} language pairs from config")
    else:
        # Default pairs
        language_pairs = processor.language_pairs
    
    # Process each language pair
    results = []
    
    for src_lang, tgt_lang, pair_rows in language_pairs:
        # Skip if no expected rows AND not processing a specific language pair
        if pair_rows is None and not (args.src_lang and args.tgt_lang):
            continue  # Skip if we don't have data count
            
        logging.info(f"\n{'='*50}")
        if pair_rows:
            logging.info(f"Processing {src_lang} → {tgt_lang} (Max: {pair_rows} rows)")
        else:
            logging.info(f"Processing {src_lang} → {tgt_lang}")
        logging.info(f"{'='*50}")
        
        try:
            # Load data with per-pair max_rows
            pair_max_rows = pair_rows if pair_rows else max_rows
            df = processor.load_data(src_lang, tgt_lang, pair_max_rows, 
                                    hf_dataset=hf_dataset, 
                                    hf_split=hf_split)
            
            if df.empty:
                logging.warning(f"No data found for {src_lang}-{tgt_lang}")
                continue
            
            # Ensure columns are properly named
            if 'source' not in df.columns:
                # Try to identify source column
                df.rename(columns={df.columns[0]: 'source'}, inplace=True)
            if 'target' not in df.columns:
                # Try to identify target column
                if 'translation' in df.columns:
                    df.rename(columns={'translation': 'target'}, inplace=True)
                elif len(df.columns) > 1:
                    df.rename(columns={df.columns[1]: 'target'}, inplace=True)
            
            # Forward translation: src_lang → tgt_lang
            logging.info(f"Generating forward translations {src_lang}→{tgt_lang}")
            translations_fwd = []
            chunk_size = 100
            
            for i in tqdm(range(0, len(df), chunk_size), desc=f"Translating {src_lang}→{tgt_lang}"):
                chunk = df.iloc[i:i+chunk_size]
                chunk_translations = translator.translate_batch(
                    chunk['source'].tolist(),
                    src_lang,
                    tgt_lang,
                    num_beams=num_beams,
                    batch_size=batch_size
                )
                translations_fwd.extend(chunk_translations)
            
            # Backward translation: tgt_lang → src_lang
            logging.info(f"Generating backward translations {tgt_lang}→{src_lang}")
            translations_bwd = []
            
            for i in tqdm(range(0, len(df), chunk_size), desc=f"Translating {tgt_lang}→{src_lang}"):
                chunk = df.iloc[i:i+chunk_size]
                chunk_translations = translator.translate_batch(
                    chunk['target'].tolist(),
                    tgt_lang,
                    src_lang,
                    num_beams=num_beams,
                    batch_size=batch_size
                )
                translations_bwd.extend(chunk_translations)
            
            # Create output dataframe with both directions
            df_output = pd.DataFrame({
                'source': df['source'].values,
                'target': df['target'].values,
                'translation_source': translations_fwd[:len(df)],  # src→tgt
                'translation_target': translations_bwd[:len(df)]   # tgt→src
            })
            
            # Filter translations
            # Keep only rows where translations are valid (not empty, not too long, etc.)
            df_filtered = df_output[
                (df_output['translation_source'].str.len() > 1) &
                (df_output['translation_source'].str.len() < 512) &
                (df_output['translation_target'].str.len() > 1) &
                (df_output['translation_target'].str.len() < 512) &
                (df_output['translation_source'] != df_output['source']) &
                (df_output['translation_target'] != df_output['target'])
            ].drop_duplicates(subset=['source', 'target', 'translation_source', 'translation_target'])
            
            logging.info(f"Filtered from {len(df_output)} to {len(df_filtered)} rows")
            
            # Calculate statistics
            stats = {
                'total_rows': len(df_filtered),
                'avg_source_length': df_filtered['source'].str.len().mean(),
                'avg_target_length': df_filtered['target'].str.len().mean(),
                'avg_translation_source_length': df_filtered['translation_source'].str.len().mean(),
                'avg_translation_target_length': df_filtered['translation_target'].str.len().mean(),
                'unique_sources': df_filtered['source'].nunique(),
                'unique_targets': df_filtered['target'].nunique(),
            }
            
            # Save results
            output_file = os.path.join(args.output_dir, f"{src_lang}-{tgt_lang}_distilled.csv")
            df_filtered.to_csv(output_file, index=False)
            logging.info(f"Saved {len(df_filtered)} bidirectional translations to {output_file}")
            
            # Save statistics
            stats_file = os.path.join(args.output_dir, f"{src_lang}-{tgt_lang}_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            results.append({
                'language_pair': f"{src_lang}-{tgt_lang}",
                'original_rows': pair_rows,
                'processed_rows': len(df),
                'filtered_rows': len(df_filtered),
                **stats
            })
            
        except Exception as e:
            logging.error(f"Error processing {src_lang}-{tgt_lang}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'processing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nProcessing complete! Summary saved to {summary_file}")

if __name__ == "__main__":
    main()