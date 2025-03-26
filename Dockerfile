FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install python-dotenv

CMD ["streamlit", "run", "script3.py"]
