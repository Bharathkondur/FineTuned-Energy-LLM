from datasets import load_dataset
import pandas as pd
import os

def main():
    print("Loading dataset...")
    try:
        ds = load_dataset("GainEnergy/oilandgas-engineering-dataset")
        print("Dataset loaded successfully.")
        print(ds)
        
        # Convert to pandas DataFrame for easier handling and saving
        if 'train' in ds:
            df = ds['train'].to_pandas()
            
            # Save to CSV
            csv_path = "oil_gas_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved dataset to {csv_path}")
            
            # Save to JSONL (useful for text data)
            json_path = "oil_gas_data.jsonl"
            df.to_json(json_path, orient='records', lines=True)
            print(f"Saved dataset to {json_path}")
            
            print("\nFirst 5 rows of the data:")
            print(df.head())
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
