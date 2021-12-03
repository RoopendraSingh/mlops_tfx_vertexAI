FROM gcr.io/tfx-oss-public/tfx:1.3.2
RUN mkdir -p custom_components
COPY . ./custom_components/
RUN pip install --upgrade google-cloud-aiplatform==1.6.0 google-cloud-storage==1.42.3
