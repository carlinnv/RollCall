"""Pin settings for tests so a developer's .env never changes results or calls the network.

Empty keys mean Gemini is off unless a test turns it on with a stand-in (the `gemini` fixture in
test_api.py). load_dotenv and gemini_classifier.load_env_file never override variables that exist."""
import os

for key, value in {"DEMO_STORY": "full", "CLUSTER_METHOD": "auto", "DATASET": "demo", "DEMO_MODE": "0",
                   "CLAIM_CHECK_NLI": "0", "GEMINI_API_KEY": "", "GEMINI_KEY": "", "GOOGLE_API_KEY": "",
                   "PHOTON_API_KEY": "",
                   # Never touch a real TigerData database. Empty (not unset) so .env cannot fill it in.
                   "TIGER_DATABASE_URL": ""}.items():
    os.environ[key] = value
os.environ.pop("CLASSIFIER", None)

# Never read or write the real on-call roster (backend/.data/oncall.json, which holds staff phone
# numbers), and never hold messages for quiet hours unless a test sets the clock itself.
import tempfile  # noqa: E402

os.environ["ONCALL_FILE"] = os.path.join(tempfile.mkdtemp(prefix="rollcall-oncall-"), "oncall.json")
os.environ["QUIET_HOURS"] = "off"
for key in ("SCHOOL_DAYS", "SCHOOL_HOURS", "SCHOOL_TIMEZONE"):
    os.environ.pop(key, None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _empty_roster(monkeypatch, tmp_path):
    """Every test starts with an empty on-call roster of its own."""
    from app.services import oncall
    from app.store import store

    monkeypatch.setattr(store, "roster", oncall.Roster(tmp_path / "oncall.json"))
