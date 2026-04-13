# DeepVision CrowdMonitor - Milestone 4

This milestone introduces a modern real-time Web Dashboard, Email Alerts, and Docker/GPU Deployment methodologies.

## Dashboard Highlights
- Real-time side-by-side visualization of the original video stream alongside the generated density map.
- Aesthetic dashboard with glowing status indicators driven by live API.
- Fully modular backend utilizing Flask and SSE patterns.

## Getting Started Locally
1. Install dependencies: `pip install -r requirements.txt Flask`
2. Configure dummy parameters in `alert_utils.py` with your real SMTP constraints if you want to deploy emails.
3. Start the Web App: `python app.py`
4. Access at `http://localhost:5000` or `http://127.0.0.1:5000`.

## Running with Docker (GPU Support)
This application includes a `Dockerfile` powered by PyTorch's official CUDA base.
Ensure you have [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) operational on WSL2 to leverage GPU.

```bash
docker-compose up --build
```
Your dashboard will boot onto port `5000` with native CUDA injection, rapidly accelerating frame-rate inferencing.

## Customization
We implemented the alert via `send_email_alert` stored in `alert_utils.py`. The `THRESHOLD` can be adjusted inside `app.py`.
