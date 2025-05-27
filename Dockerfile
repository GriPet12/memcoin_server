FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY memcoin_graduation_model.pkl .
COPY features.txt .

EXPOSE 5000

CMD ["python", "app.py"]