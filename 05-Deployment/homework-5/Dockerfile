FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && uv pip install --all-extras .

# Copy FastAPI app and model
COPY main.py pipeline_v1.bin ./

EXPOSE 9696

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
