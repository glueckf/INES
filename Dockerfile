FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN mkdir -p /app/src/res

WORKDIR /app/src

CMD ["python", "start_simulation.py"]