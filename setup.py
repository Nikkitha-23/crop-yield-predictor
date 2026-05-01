import os
import urllib.request

HF_BASE = "https://huggingface.co/Nikkitha-23/crop-yield-model/resolve/main"

FILES = {
    "models/random_forest.pkl"      : f"{HF_BASE}/random_forest.pkl",
    "models/encoders.pkl"           : f"{HF_BASE}/encoders.pkl",
    "models/feature_names.pkl"      : f"{HF_BASE}/feature_names.pkl",
    "models/fertilizer_model.pkl"   : f"{HF_BASE}/fertilizer_model.pkl",
    "models/fertilizer_encoders.pkl": f"{HF_BASE}/fertilizer_encoders.pkl",
    "models/fertilizer_features.pkl": f"{HF_BASE}/fertilizer_features.pkl",
}

def setup():
    # Check if ALL models exist locally
    all_exist = all(os.path.exists(f) for f in FILES.keys())

    if all_exist:
        print("✅ All models already exist!")
        return

    print("📥 Downloading models from Hugging Face...")
    os.makedirs("models", exist_ok=True)

    for filepath, url in FILES.items():
        if os.path.exists(filepath):
            print(f"✅ {filepath} already exists — skipping!")
            continue
        print(f"Downloading {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ {filepath} done!")
        except Exception as e:
            print(f"❌ Failed: {filepath} — {e}")

    print("🎉 Setup complete!")

if __name__ == "__main__":
    setup()