import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

def load_huggingface_sentiment(dataset_name: str = "Sp1786/multiclass-sentiment-analysis-dataset"):
    """
    Load sentiment dataset from Hugging Face.
    
    Args:
        dataset_name: Hugging Face dataset identifier
    
    Returns:
        Tuple of (train_df, validation_df, test_df)
    """
    print(f"Loading dataset from Hugging Face: {dataset_name}...")
    
    try:
        # Load all splits
        dataset = load_dataset(dataset_name)
        
        train_df = pd.DataFrame(dataset['train'])
        
        print(f"✓ Successfully loaded dataset")
        print(f"  Train samples: {len(train_df)}")
        
        return train_df
    
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise


def map_label_to_sentiment(label_int: int) -> str:
    """
    Map integer label to sentiment string.
    Assuming: 0=negative, 1=neutral, 2=positive
    (Adjust mapping based on actual dataset)
    
    Args:
        label_int: Integer label (0, 1, 2)
    
    Returns:
        Sentiment string ('negative', 'neutral', 'positive')
    """
    label_map = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    return label_map.get(label_int, 'neutral')


def process_huggingface_sentiment(output_dir: str = 'data/processed'):
    """
    Process Hugging Face sentiment dataset.
    Saves only train split in required format.
    
    Args:
        output_dir: Directory to save processed data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Processing Hugging Face Sentiment Dataset")
    print("="*60 + "\n")
    
    # Load from Hugging Face
    train_df = load_huggingface_sentiment()
    
    print("\nDataset columns:", train_df.columns.tolist())
    print("Dataset shape:", train_df.shape)
    print("\nFirst few rows:")
    print(train_df.head())
    
    # Expected columns: id, text, label, sentiment
    # Keep only text and convert label to sentiment if needed
    
    if 'sentiment' in train_df.columns:
        # If sentiment column already exists, use it
        print("\n✓ Using existing 'sentiment' column")
        processed_df = train_df[['text', 'sentiment']].copy()
        processed_df.columns = ['text', 'label']
    elif 'label' in train_df.columns:
        # Convert integer label to sentiment string
        print("\n✓ Converting integer labels to sentiment strings")
        processed_df = train_df[['text', 'label']].copy()
        
        # Map labels
        processed_df['label'] = processed_df['label'].apply(map_label_to_sentiment)
    else:
        raise ValueError("Dataset must have either 'sentiment' or 'label' column")
    
    # Remove any rows with missing values
    processed_df = processed_df.dropna()
    
    print(f"\nProcessed samples: {len(processed_df)}")
    print("\nLabel distribution:")
    label_counts = processed_df['label'].value_counts()
    for label in ['positive', 'neutral', 'negative']:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(processed_df) if len(processed_df) > 0 else 0
        print(f"  {label:10s}: {count:6d} ({pct:5.1f}%)")
    
    # Save processed data
    output_file = output_dir / 'evaluation.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved evaluation dataset to: {output_file}")
    print(f"  Total samples: {len(processed_df)}")
    print(f"  Columns: {processed_df.columns.tolist()}")
    
    return processed_df


if __name__ == '__main__':
    import sys
    
    # Optional: specify custom output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed'
    
    try:
        dataset = process_huggingface_sentiment(output_dir)
        print("\n" + "="*60)
        print("✓ Processing complete!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)