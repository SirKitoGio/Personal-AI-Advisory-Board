# Deployment Architecture: NVIDIA-Only Cloud Mode

## 1. Overview
This guide maps out the "Pure Cloud" deployment strategy for the Local AI Advisor. To ensure 24/7 availability and zero dependency on local hardware, we bypass Ollama and utilize high-performance NVIDIA-hosted models.

## 2. Infrastructure
*   **Host:** Streamlit Community Cloud (Recommended)
*   **Engine:** Python 3.10+
*   **LLM Provider:** NVIDIA API (integrate.api.nvidia.com)

## 3. Environment Variables (Required)
You MUST set these in the Streamlit Cloud Dashboard (Settings > Secrets):

```toml
IS_PROD = "true"
NVIDIA_API_KEY = "your_nvapi_key_here"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
```

## 4. Execution Steps (The "Code Way")
1.  **Push to GitHub:** Ensure `.env` and `data/` are in `.gitignore`.
2.  **Connect Streamlit Cloud:**
    *   Point to your repository.
    *   Set Main file path: `src/app.py`.
3.  **Configure Secrets:** Copy the values from your local `.env` to the Streamlit Cloud "Secrets" section.
4.  **Deploy:** The app will automatically detect `IS_PROD=true` and switch to the optimized NVIDIA-only interface.

## 5. Why NVIDIA-Only?
*   **Zero Latency:** No local tunnels (ngrok) required.
*   **High Performance:** Access to Llama 3.1 405B and Nemotron 120B.
*   **Scalability:** The app remains alive even when your M4 Mac is offline.

---
*Maintained by the Relentless Architect (Keith Alan Speirs)*
