# Photon setup (iMessage channel)

How to get real iMessages flowing into the app on your machine. Takes about 10 minutes.

You can skip all of this. Without it the app still runs, and the "Simulate family reply" box on the case page does the same job.

## What you need

- An iPhone or Mac that can send iMessages
- Python 3.11 or later
- Access to the team's Photon organization (Carlin has invited you; check your email)

## Steps

### 1. Get three values from the Photon dashboard

Sign in at https://app.beta.photon.codes and pick the team's project in the switcher at the top of the sidebar. If you cannot see it, accept Carlin's invitation first.

**a. Project ID**

Open **Settings** and copy the Project ID. It starts with `pho_prj_`.

**b. API key**

1. Open **API keys** and type a name for the key (your own name is fine).
2. Open the **Create key** menu and choose **Custom**.
3. Tick `project:read` and `project:write`, then create the key.
4. Copy it now. It is shown only once.

The key is a secret. Do not paste it in a group chat or commit it.

**c. Agent number**

1. Open **iMessage**.
2. Under "Add the number you or your user will send from", enter your own phone number and select **Add**.
3. Complete **Assign a number from the shared pool**.
4. Copy the assigned number. This is the agent number you will text.

The agent only answers numbers that have been added here, so this part cannot be skipped. The project has 25 of these slots in total, so add one number each.

### 2. Create your `.env`

From the repo root:

```bash
cp .env.example .env
```

Open `.env` and fill in four values. Phone numbers need `+` and the country code, with no spaces or dashes.

```
PHOTON_PROJECT_ID=pho_prj_...        # step 1a
PHOTON_API_KEY=...                   # step 1b
PHOTON_AGENT_HANDLE=+1XXXXXXXXXX     # step 1c, the agent number
PHOTON_TEST_SENDER=+1XXXXXXXXXX      # your own phone number
```

`.env` is git-ignored. Leave it that way.

### 3. Install the backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you already had a `.venv`, run the `pip install` line again. This branch added a dependency.

### 4. Check the connection

```bash
.venv/bin/python scripts/photon_echo.py
```

When it prints `Ready`, text anything to the agent number from your phone. Within a few seconds you should get your message echoed back, and the script exits.

If that works, setup is done.

### 5. Run the app

```bash
.venv/bin/uvicorn app.main:app --port 8000 --reload
```

Look for this line in the log:

```
photon channel listening on +1XXXXXXXXXX (1 reachable families)
```

Start the UI in a second terminal (`cd frontend && npm install && npm run dev`) and open http://localhost:5173.

## Try it

Your phone plays the parent of one demo student: Sofia (S002), a Spanish-speaking family on bus Route 7.

1. Text the agent number: `El autobús no llegó a la parada`
2. You get a short thank-you reply in Spanish.
3. Refresh the case queue. Sofia's absence now shows "transportation" with your text as the evidence.

To play a different student, add `PHOTON_DEMO_STUDENT_ID=S014` (or any student ID) to `.env` and restart the API.

## Things that will surprise you

- **You must text first.** The agent cannot message a number that has not texted it. Send one message before testing anything outbound.
- **The page does not update by itself.** Refresh the browser after you send a text.
- **Text only.** Photos and voice notes are ignored.
- **`STOP` works.** If you text `STOP`, the agent goes silent for your number until you restart the API.
- **A safety message silences the agent.** After a text about bullying or feeling unsafe, you get no more automatic replies until you click **Reset demo data**.
- **Restarting the API resets the data.** Everything is in memory. Saving a backend file triggers a restart.

## If something goes wrong

| What you see | Fix |
| :--- | :--- |
| `Missing in .env: ...` | A value in step 2 is empty or misspelled. |
| `PHOTON_PROJECT_ID must start with pho_prj_` | You copied the wrong value. Repeat step 1a. |
| `Photon API error 401` or `403` | The API key is wrong, expired, or missing `project:read` / `project:write`. Create a new one (step 1b). |
| Echo script times out | Check you texted the agent number from the exact number in `PHOTON_TEST_SENDER`, and that you added that same number in step 1c. On iPhone, Settings → Apps → Messages → Send & Receive shows which number you send from. |
| No `photon channel listening` line | One of the four `.env` values is missing, or `.env` is not in the repo root. |
| Your text gets "This number is not linked to a student record" | `PHOTON_TEST_SENDER` does not match the number you are texting from. |
| `curl: Failed to connect to localhost port 8000` | The API is not running. Do step 5. |
| A check-in returns `queued` | You have not texted the agent yet, or the student is not the one your phone is linked to. |

## More detail

- [PHOTON_FEATURES.md](PHOTON_FEATURES.md): every feature on the channel, its files and its rules
- [Photon 2.0 docs](https://photon.codes/docs/beta/get-started/imessage)
