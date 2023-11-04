FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR app/
COPY ["Pipfile", "Pipfile.lock", "bin/model-randomforest.pkl", "bin/dv-randomforest.pkl", "server/webservice.py", "server/predict.html", "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "webservice:app"]
