# Deploy this app to Hugging Face Spaces

This folder contains the Dockerfile used when deploying the Streamlit app to **Hugging Face Spaces**. The Space runs the same app as local Docker, but:

- **HTS tab** shows a message only: *"High-Throughput Characterization is only available on local installs (Docker or pip)."* No batch runs on the Space to keep usage fair.
- The app listens on **port 7860** (required by HF).

## One-command deploy (recommended)

After you commit and push to GitHub, from **repo root**:

```bash
./scripts/push_hf_space.sh --pull
```

- Builds a fresh bundle, syncs to a cached clone under `/tmp`, commits, and pushes to your Space **`main`** branch.
- **Pushing triggers an automatic rebuild** on Hugging Face (that is the restart).
- Requires `hf auth login` (or git credentials for `huggingface.co`).
- Override Space URL: `HF_SPACE_URL=https://huggingface.co/spaces/USER/NAME ./scripts/push_hf_space.sh`

Optional custom message: `./scripts/push_hf_space.sh -m "fix: faster dgrid"`

## Quick deploy (manual bundle)

1. From the **repo root**, run:
   ```bash
   ./scripts/deploy_to_hf_space.sh
   ```
   This creates **`build_space/`** with everything needed: Dockerfile at root + app code.

2. Create a new **Space** on Hugging Face:
   - https://huggingface.co/new-space
   - Pick **Docker** as the SDK.
   - Create the Space (it will be empty).

3. Push the contents of **`build_space/`** into the Space repo:
   ```bash
   cd build_space
   git init
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git add .
   git commit -m "Initial Space deployment"
   git push space main
   ```
   (Use your Space URL and branch; HF often uses `main`.)

4. The Space will build and run. Open the Space URL when the build finishes.

## Manual deploy (no script)

- Copy **`spaces/Dockerfile`** to the **root** of the repo (or of a clone).
- Rename or replace the existing root `Dockerfile` if you need to keep the local one.
- Push that repo to a Hugging Face Space configured for **Docker**. The Space must have this Dockerfile at root and all app code (python_cli, mgr_src, cifs, requirements.docker.lock, etc.) so `COPY . /app` works.

## Encouraging local use

The in-app HTS tab message tells users to use **Docker Desktop** or **pip install** (Docker Hub / PyPI when public) for full High-Throughput and to save Space resources.
