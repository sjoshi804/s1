import requests
import jsonlines
from datasets import load_dataset
from tqdm import tqdm
import concurrent.futures
from typing import List, Optional

def rephrase_trajectory(trajectory: str) -> Optional[str]:
    url = "http://0.0.0.0:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "messages": [
            {
                "role": "user", 
                "content": f"Rephrase this trajectory to be more concise, while keeping the content the same:\n\n{trajectory}"
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error making request: {e}")
        return None

def process_batch(trajectories: List[str]) -> List[Optional[str]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(rephrase_trajectory, trajectories))

ds = load_dataset("simplescaling/s1K-1.1_tokenized")

BATCH_SIZE = 10
with jsonlines.open("short_trajectories.jsonl", "w") as writer:
    for i in tqdm(range(0, len(ds["train"]), BATCH_SIZE)):
        batch_trajectories = [
            ds["train"][j]["deepseek_thinking_trajectory"] 
            for j in range(i, min(i + BATCH_SIZE, len(ds["train"])))
        ]
        short_trajectories = process_batch(batch_trajectories)
        
        for short_trajectory in short_trajectories:
            if short_trajectory is not None:
                writer.write({"short_trajectory": short_trajectory})
