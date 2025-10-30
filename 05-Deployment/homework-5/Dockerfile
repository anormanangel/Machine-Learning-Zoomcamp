FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv and project dependencies (no bash subshell)
RUN pip install uv && uv pip install --all-extras .

# Copy your FastAPI app
COPY main.py .

EXPOSE 9696

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9696"]
