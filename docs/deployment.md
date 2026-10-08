# Deployment Guide

This guide covers deploying the RAG-Based Drug-Drug Interaction Decision Support System for a public, evidence-only demo.

## Architecture & Configuration

This project can be run entirely in **evidence-only mode** (without any LLM dependency). 

- **Frontend:** Any static host (Vercel, Netlify, Cloudflare Pages).
- **Backend:** A Dockerized container (Render, Heroku, AWS App Runner).

### Backend Configuration

Deploy the backend using the provided `backend/Dockerfile`.

#### Required Environment Variables
- `LLM_PROVIDER=none` (Disables the LLM layer, relying solely on RAG retrieval).
- `FRONTEND_ORIGIN=https://<your-public-frontend-url.com>` (Configures CORS to allow requests from your frontend).

#### Storage & Persistence
**CRITICAL:** The application is designed to ingest new drugs *on demand* if they are not already in the database. When a user checks a new drug, the backend calls RxNorm and DailyMed, processes the label, and saves it. 
- If deployed statelessly (without a persistent volume), any new drugs processed by users will be lost whenever the container restarts.
- **For a fully functional deployment:** Mount a **Persistent Volume** to `/app/backend/data` inside the container. This ensures `ddi.db` (SQLite) and `chroma_store/` (ChromaDB) survive restarts.

*(Note: The provided Dockerfile builds an initial seed database into the image, so it works out-of-the-box statelessly for the seed drugs, but will lose any newly queried drugs upon restart).*

#### Hardware Requirements
- **RAM:** A minimum of **1GB RAM** is strongly recommended. The `sentence-transformers` model operates in memory. Small free-tier instances (e.g., 256MB or 512MB) may experience Out-Of-Memory (OOM) crashes.

### Frontend Configuration

The frontend is a static React application built with Vite.

#### Required Environment Variables (Build-Time)
- `VITE_API_BASE_URL=https://<your-public-backend-url.com>`

To build the frontend for production:
```bash
cd frontend
npm install
npm run build
```
Upload the resulting `dist/` directory to your static host.

## Running Locally via Docker

To validate the deployment image locally:

1. **Build the image (from the repository root):**
   ```bash
   docker build -t ddi-backend -f backend/Dockerfile .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8001:8001 \
       -e LLM_PROVIDER=none \
       -e FRONTEND_ORIGIN=http://localhost:5173 \
       ddi-backend
   ```

3. **Verify Health:**
   ```bash
   curl http://localhost:8001/api/health
   ```

4. **Verify Evidence-Only RAG:**
   Point your frontend (running via `npm run dev`) at the containerized backend. Try checking an interaction between Warfarin and Ibuprofen. You should receive the retrieved evidence with a clean fallback message, and no LLM API calls will be made.
