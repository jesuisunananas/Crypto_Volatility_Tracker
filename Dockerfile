FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py app.py
COPY model.py model.py

EXPOSE 5000

CMD ["python", "app.py"]