# Note : The Dockerfile must be placed inside the Waste Classifier Application in order to run smoothly when performing Build-Run

FROM python:3.10

WORKDIR /Waste Classifier App

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "Waste Classifier App.py", "--server.address=0.0.0.0"]
