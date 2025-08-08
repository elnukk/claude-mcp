FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# gradio port
EXPOSE 7860

CMD ["python", "app.py"]