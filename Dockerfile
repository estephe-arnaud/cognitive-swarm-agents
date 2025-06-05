# syntax=docker/dockerfile:1

# --- Stage 1: Build virtual environment with dependencies ---
FROM python:3.11-slim as builder

# Set up Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_NO_INTERACTION=1
RUN pip install --upgrade pip && \
    pip install poetry

# Copy only the dependency files to leverage Docker cache
WORKDIR /app
COPY poetry.lock pyproject.toml ./

# Install dependencies
RUN poetry install --no-root --no-dev --sync

# --- Stage 2: Final application image ---
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv
# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code
COPY src ./src
COPY scripts ./scripts
COPY config ./config

# Set the default command to run the main workflow
# You can override this command when running the container
CMD ["python", "-m", "scripts.run_makers", "--help"]

# For example, to run a specific script:
# docker run makers-app python -m scripts.run_ingestion --max_results 1
# docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -e MONGODB_URI=$MONGODB_URI makers-app python -m scripts.run_makers --query "My query"