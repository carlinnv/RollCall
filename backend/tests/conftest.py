"""Pin settings for tests so a developer's .env never changes results or calls the network.

Empty keys mean Gemini is off unless a test turns it on with a stand-in (the `gemini` fixture in
test_api.py). load_dotenv and gemini_classifier.load_env_file never override variables that exist."""
import os

for key, value in {"DEMO_STORY": "full", "CLUSTER_METHOD": "auto", "DATASET": "demo", "DEMO_MODE": "0",
                   "CLAIM_CHECK_NLI": "0", "GEMINI_API_KEY": "", "GEMINI_KEY": "", "GOOGLE_API_KEY": "",
                   "PHOTON_API_KEY": ""}.items():
    os.environ[key] = value
os.environ.pop("CLASSIFIER", None)
