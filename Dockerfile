FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ ./model/

EXPOSE 5000

ENV FASTFOREX_API_KEY=your_api_key_here

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]

