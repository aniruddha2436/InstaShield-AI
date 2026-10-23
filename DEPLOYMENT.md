# Deploy InstaShield AI on Render

Status: deployment files are prepared; no public deployment has been created yet. A connected Render account and a source repository or image accessible to Render are required.

## Configuration

The Dockerfile builds React/Vite and serves the built assets from FastAPI on the same origin as /api/analyze. It binds to 0.0.0.0 and Render's PORT. The native XGBoost JSON model and metadata are included; the original CSV, notebook, joblib snapshot, local environments, tests and browser history are not copied into the runtime image. Linux libgomp1 is installed for XGBoost. The container runs as a non-root user.

render.yaml defines one Docker web service with the free plan and /api/health as its health check. No database, Instagram API credentials, or separate frontend service is needed. Confirm the plan shown in Render before creating any paid resources. Free-service availability, startup delays and limits depend on Render's current account and plan policies.

## Deploy through the dashboard

1. Put this project at the root of a GitHub or GitLab repository accessible to your Render account. Include Dockerfile, render.yaml, backend, and frontend. Exclude .venv, node_modules, .env and any other private files.
2. In Render, create a new Blueprint from that repository and select render.yaml. Alternatively create a Web Service, choose Docker, set Dockerfile Path to ./Dockerfile, and set health check path to /api/health.
3. Review the plan and create the service. Leave the Docker Command override empty; the Dockerfile already starts the server.
4. Wait for the build and readiness checks to pass. Open the actual onrender.com URL returned by Render.
5. Check the homepage, /api/health, and both notebook examples. The bio input sends its calculated count; only model features are analyzed.

If this project is in a repository subdirectory, set Render's root directory to that folder so Dockerfile and render.yaml paths resolve correctly.

## Local container verification (optional)

With Docker Desktop running, from this project folder:

~~~powershell
docker build -t instashield-ai .
docker run --rm -p 8000:10000 instashield-ai
~~~

Open http://127.0.0.1:8000. This single address serves both frontend and backend.

## Verification and limitations

The existing frontend production build and nine model/API tests passed before this hosting change. Docker is not installed in the authoring environment, so the Linux container build must still be verified by Render. Local regression-check attempts during deployment preparation timed out; do not treat that as a passed container or routing check. A live Render URL will be provided only after an actual successful deployment.

History is stored in browser localStorage and remains optional. Localhost history does not transfer to the hosted address or other browsers. The app does not persist a backend history database. Public hosting does not add Instagram access or identity verification.

References: [Render Docker deployment](https://render.com/docs/docker), [Render Blueprints](https://render.com/docs/blueprint-spec).

## Combined v4 + v5 candidate

A candidate artifact is available separately. The Dockerfile still copies only the selected active model pair; it does not automatically deploy candidate_v4_v5 or rollback_v4. Follow UPDATE_V4_V5.md to review, select, test, push and verify the candidate, or restore the rollback model.
