"""Pin demo settings for tests so a developer's .env (DEMO_STORY=live, CLASSIFIER=gemini, ...) never
changes results or calls the network. load_dotenv / load_env_file never override these."""
import os

for key, value in {"DEMO_STORY": "full", "CLASSIFIER": "keyword", "CLUSTER_METHOD": "auto",
                   "DATASET": "demo", "DEMO_MODE": "0", "CLAIM_CHECK_NLI": "0"}.items():
    os.environ[key] = value
