import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import json
import random

# Load your data
with open('oil_gas_data.jsonl', 'r') as f:
    original_data = [json.loads(line) for line in f]

print(f"Loaded {len(original_data)} original entries")

# Setup Augmenters (no WordNet dependency)
aug_swap = naw.RandomWordAug(action='swap')  # Swap random words
aug_delete = naw.RandomWordAug(action='delete')  # Delete random words
aug_split = naw.SplitAug()  # Split words

augmenters = [aug_swap, aug_delete, aug_split]

augmented_data = []

for i, entry in enumerate(original_data):
    # Keep the original
    augmented_data.append(entry)
    
    # Create 2 new versions of each line using different augmenters
    for _ in range(2):
        aug = random.choice(augmenters)
        try:
            new_text = aug.augment(entry['text'])[0]
            augmented_data.append({"text": new_text, "label": entry['label']})
        except Exception as e:
            # If augmentation fails, just duplicate the original
            augmented_data.append(entry)
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(original_data)} entries...")

# Save the expanded dataset (now ~1,300 lines)
with open('energy_data_expanded.jsonl', 'w') as f:
    for entry in augmented_data:
        f.write(json.dumps(entry) + '\n')

print(f"\nDataset expanded from {len(original_data)} to {len(augmented_data)} lines!")
print(f"Saved to: energy_data_expanded.jsonl")