import os
import urllib.request

HF_BASE = "https://huggingface.co/Nikkitha-23/crop-yield-model/resolve/main"

FILES = {
    "models/random_forest.pkl" : f"{HF_BASE}/random_forest.pkl",
    "models/encoders.pkl"      : f"{HF_BASE}/encoders.pkl",
    "models/feature_names.pkl" : f"{HF_BASE}/feature_names.pkl",
}

def setup():
    if os.path.exists("models/random_forest.pkl"):
        print("✅ Model already exists!")
        return

    print("📥 Downloading model from Hugging Face...")
    os.makedirs("models", exist_ok=True)

    for filepath, url in FILES.items():
        print(f"Downloading {filepath}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ {filepath} downloaded!")

    print("🎉 All models downloaded successfully!")

if __name__ == "__main__":
    setup()
