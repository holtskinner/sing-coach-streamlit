# AriaCoach - Singing Teacher

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run)

A [Streamlit](https://streamlit.io/) app that records your singing, sends it to
Gemini for coaching feedback, and speaks the feedback back using Gemini TTS.
Runs locally or on Cloud Run.

## Run locally

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

2. Set the environment variables Vertex AI needs:

   ```bash
   export GOOGLE_CLOUD_PROJECT='<Your Google Cloud Project ID>'  # Change this
   export GOOGLE_CLOUD_LOCATION='global'
   export GOOGLE_GENAI_USE_ENTERPRISE=true
   ```

3. Authenticate to your Google Cloud project:

   ```sh
   gcloud config set project $GOOGLE_CLOUD_PROJECT
   gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
   gcloud auth application-default login -q
   ```

4. Run the app:

   ```bash
   uv run streamlit run app.py
   ```

   Open the URL it prints, record yourself singing, and AriaCoach will respond.

## Deploy to Cloud Run

1. Set the project and region:

   ```bash
   export GOOGLE_CLOUD_PROJECT='<Your Google Cloud Project ID>'  # Change this
   export GOOGLE_CLOUD_REGION='us-central1'
   ```

2. Deploy:

   ```bash
   gcloud run deploy aria-coach \
     --port=8080 \
     --source=. \
     --allow-unauthenticated \
     --region=$GOOGLE_CLOUD_REGION \
     --project=$GOOGLE_CLOUD_PROJECT \
     --set-env-vars=GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_REGION
   ```

On success you get a URL to the deployed service.
