FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose both ports
EXPOSE 7860 8000

CMD ["python", "app.py"]