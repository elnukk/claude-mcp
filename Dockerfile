FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose both ports
EXPOSE 7860 8000

# Start both services
CMD uvicorn server:app --host 0.0.0.0 --port 8000 & python app.py