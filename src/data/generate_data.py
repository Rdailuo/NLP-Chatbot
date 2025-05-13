import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import json
import os
from pathlib import Path

# Initialize Faker
fake = Faker()

def generate_claim_narrative(is_fraudulent):
    """Generate a realistic claim narrative with potential fraud indicators."""
    if is_fraudulent:
        # Fraudulent claim patterns
        patterns = [
            "The {item} was {damage} when I {action}. I need immediate replacement.",
            "My {item} was stolen from my {location}. I don't have any {evidence}.",
            "The {incident} happened while I was {activity}. I need compensation for {damage}.",
            "I lost my {item} during {event}. I need to file this claim quickly.",
            "The {damage} occurred due to {cause}. I need this processed as soon as possible."
        ]
        items = ["laptop", "phone", "jewelry", "camera", "watch"]
        damages = ["damaged", "broken", "destroyed", "lost", "stolen"]
        actions = ["was moving", "was traveling", "was at a party", "was shopping"]
        locations = ["car", "hotel", "restaurant", "parking lot", "public transport"]
        evidence = ["receipts", "photos", "witnesses", "police report"]
        incidents = ["accident", "theft", "damage", "loss"]
        activities = ["traveling", "shopping", "at work", "at a restaurant"]
        events = ["vacation", "business trip", "moving", "party"]
        causes = ["unknown reasons", "mysterious circumstances", "unforeseen events"]
    else:
        # Legitimate claim patterns
        patterns = [
            "My {item} was {damage} due to {cause}. I have {evidence} to support this.",
            "During {event}, my {item} was {damage}. I can provide {evidence}.",
            "I noticed {damage} to my {item} after {incident}. I have {evidence}.",
            "The {item} was {damage} when {incident} occurred. I have {evidence}.",
            "My {item} was {damage} at {location}. I have {evidence} and {additional_info}."
        ]
        items = ["car", "house", "furniture", "electronics", "appliances"]
        damages = ["damaged by water", "damaged by fire", "damaged in accident", "damaged by storm"]
        causes = ["water leak", "electrical fire", "car accident", "severe weather"]
        evidence = ["photos", "repair estimates", "police report", "witness statements"]
        events = ["a storm", "a car accident", "a home incident", "a natural disaster"]
        incidents = ["the storm", "the accident", "the incident", "the disaster"]
        locations = ["home", "work", "parking lot", "garage"]
        additional_info = ["security camera footage", "insurance documentation", "maintenance records"]

    template = random.choice(patterns)
    narrative = template.format(
        item=random.choice(items),
        damage=random.choice(damages),
        action=random.choice(actions) if is_fraudulent else "",
        location=random.choice(locations),
        evidence=random.choice(evidence),
        incident=random.choice(incidents),
        activity=random.choice(activities) if is_fraudulent else "",
        event=random.choice(events),
        cause=random.choice(causes),
        additional_info=random.choice(additional_info) if not is_fraudulent else ""
    )
    
    return narrative

def generate_claim_data(num_claims=1000):
    """Generate synthetic insurance claims data."""
    data = []
    
    for _ in range(num_claims):
        # Determine if this claim is fraudulent (20% of claims are fraudulent)
        is_fraudulent = random.random() < 0.2
        
        # Generate claim date (within last 2 years)
        claim_date = fake.date_between(start_date='-2y', end_date='today')
        
        # Generate claim amount (higher for fraudulent claims)
        if is_fraudulent:
            claim_amount = random.uniform(5000, 50000)
        else:
            claim_amount = random.uniform(1000, 15000)
        
        # Generate claim features
        claim = {
            'claim_id': fake.uuid4(),
            'claim_date': claim_date.strftime('%Y-%m-%d'),
            'claim_amount': round(claim_amount, 2),
            'policy_holder_age': random.randint(18, 80),
            'policy_holder_tenure': random.randint(1, 20),  # years
            'claim_narrative': generate_claim_narrative(is_fraudulent),
            'claim_type': random.choice(['property', 'auto', 'health', 'liability']),
            'claim_status': random.choice(['pending', 'approved', 'rejected']),
            'is_fraudulent': is_fraudulent,
            'location': fake.city(),
            'previous_claims': random.randint(0, 5),
            'time_to_report': random.randint(1, 30),  # days
            'policy_type': random.choice(['basic', 'premium', 'standard']),
            'deductible_amount': random.choice([500, 1000, 2000, 5000]),
            'coverage_amount': random.choice([10000, 25000, 50000, 100000])
        }
        
        data.append(claim)
    
    return pd.DataFrame(data)

def save_data(df, filename):
    """Save the generated data to CSV and JSON formats."""
    # Create directories if they don't exist
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = f'data/raw/{filename}.csv'
    df.to_csv(csv_path, index=False)
    
    # Save as JSON
    json_path = f'data/raw/{filename}.json'
    df.to_json(json_path, orient='records', indent=2)
    
    print(f"Data saved to {csv_path} and {json_path}")

def main():
    # Generate training data
    print("Generating synthetic insurance claims data...")
    df = generate_claim_data(num_claims=1000)
    
    # Save the data
    save_data(df, 'insurance_claims')
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total claims: {len(df)}")
    print(f"Fraudulent claims: {df['is_fraudulent'].sum()} ({df['is_fraudulent'].mean()*100:.1f}%)")
    print(f"Average claim amount: ${df['claim_amount'].mean():.2f}")
    print(f"Average claim amount (fraudulent): ${df[df['is_fraudulent']]['claim_amount'].mean():.2f}")
    print(f"Average claim amount (legitimate): ${df[~df['is_fraudulent']]['claim_amount'].mean():.2f}")

if __name__ == "__main__":
    main() 