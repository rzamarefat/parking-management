FROM python:3.12.5-alpine3.19
COPY ../requirements.txt /app/requirements.txt
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "run_producer.py"]
