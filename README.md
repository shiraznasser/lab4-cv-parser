# HR Candidate Profile Parser — Streamlit GUI

A Streamlit web app that extracts a structured candidate profile from a CV (PDF) using an LLM and LangChain's `StructuredOutputParser`. Built as Lab 4, on top of the Lab 3 output-parsing exercise.

<img width="3838" height="1847" alt="image" src="https://github.com/user-attachments/assets/870b69a0-4437-4333-9af1-5642c3c6db70" />

## What it does

1. Upload a CV as a PDF.
2. The app extracts the text with `PyPDFLoader`.
3. The text is sent to `mistralai/Mistral-Nemo-Instruct-2407` with a prompt asking for a structured JSON profile.
4. The model's response is parsed into a clean JSON object with:
   - `full_name`
   - `email`
   - `education` — list of `{degree, institution, year}`
   - `skills` — list of strings
   - `experience` — list of `{role, company, years}`
5. The JSON is displayed in the app.

## Files

- `app.py` — the Streamlit application.
- `requirements.txt` — Python dependencies.

## Running it

This app needs a GPU with enough memory to load a 12B parameter model in float16 (roughly 24 GB). It was developed and run on Kaggle notebooks.

### On Kaggle

1. Enable internet in the notebook settings.
2. Write `app.py` and `requirements.txt` into the working directory (e.g. with `%%writefile`), or clone this repo.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt pyngrok
   ```
4. Run the app and expose it with ngrok:
   ```python
   from pyngrok import ngrok
   import subprocess

   ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
   process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
   public_url = ngrok.connect(8501)
   print("Your app is live at:", public_url)
   ```
5. Open the printed URL, upload a CV PDF, and click **Parse CV**.

### Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a CUDA-capable GPU for the model to load at a reasonable speed; CPU inference will work but will be slow.

## Notes

- The model samples its output, so occasionally the JSON block comes back malformed. If parsing fails, the app shows the raw model output instead so you can inspect it.
- An ngrok authtoken is required to expose the app publicly from a notebook environment — get one free at [ngrok.com](https://ngrok.com).
