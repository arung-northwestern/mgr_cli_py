## MGR Standalone (Docker Desktop GUI-Only)

This guide is for collaborators who prefer not to use the terminal.

### 1) Install Docker Desktop
- Download: https://www.docker.com/products/docker-desktop/
- Install and launch Docker Desktop.

### 2) Sign in to Docker Hub (GUI)
- In Docker Desktop, click the avatar icon (top-right).
- Sign in with your Docker Hub account.

### 3) Pull the image (GUI)
- Open Docker Desktop → Images.
- Click **Pull**.
- Enter the image name you were given, for example:
  `your-username/mgrcli_dev:latest`
- Click **Pull**.

### 4) Run the app (GUI)
- In Docker Desktop → Images, find the image → click **Run**.
- In **Optional settings**:
  - **Ports**: set Host `8501` → Container `8501`
  - **Volumes**: add a host folder mapped to `/results`
- Click **Run**.

### 5) Open the app
- Open your browser and go to: `http://localhost:8501`

### (Optional) CLI for terminal users
Terminal users can install the CLI from the repo using `uv`:
- Install uv (macOS/Linux): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- From the repo root: `uv pip install -e .`
