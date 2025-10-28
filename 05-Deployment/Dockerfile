FROM python:3.8.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies, fallback if lock mismatched
RUN pipenv install --system --deploy || pipenv install --system

# Ensure the scikit-learn version matches your model
RUN pip install --upgrade scikit-learn==1.6.1

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
