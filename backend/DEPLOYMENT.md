# Customer Churn AI - Fixed Deployment

## Backend (Render)

The backend is optimized for reliable inference:
- Model, scaler and encoder are loaded once at startup.
- Reference/background data is cached in memory.
- `/predict` does not run SHAP.
- `/explain` uses a cached SHAP GradientExplainer for the LSTM/GRU model.
- `/health` is configured as the Render health check.
- Gunicorn uses one worker, two threads and a 120-second timeout.
- Public `/train` is disabled unless `ENABLE_TRAINING=true`.
- scikit-learn is pinned to 1.7.2 to match the serialized preprocessing artifacts.

Render settings are already included in `backend/render.yaml`.

If creating the service manually:
- Root Directory: `backend`
- Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start Command: `gunicorn --workers 1 --threads 2 --timeout 120 --graceful-timeout 30 --keep-alive 5 app:app`
- Health Check Path: `/health`
- Environment Variable: `FRONTEND_ORIGIN=https://churn-chi.vercel.app`
- Environment Variable: `ENABLE_TRAINING=false`

## Frontend (Vercel)

The frontend calls:
`https://churn-dki6.onrender.com`

The frontend now:
- warms the Render backend through `/health` when the page opens,
- uses a controlled 90-second request timeout for cold starts,
- reports backend errors clearly,
- keeps prediction and explanation separate.

## Deployment order

1. Push the updated backend files to GitHub.
2. Redeploy/restart the Render service and wait for `/health` to return HTTP 200.
3. Deploy the updated frontend to Vercel.
4. Open the Vercel site and wait for the backend warm-up.
5. Test Predict Churn.
6. Test Explain Prediction.

## Important

Render Free services can sleep after inactivity. The page warm-up reduces the impact on the first user request, but a sleeping free service can still have a cold-start delay. A paid/non-sleeping Render instance is the final solution if consistently instant first-request latency is required.
