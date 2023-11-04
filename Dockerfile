FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR app/
COPY . ./

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "server.webservice:app"]
