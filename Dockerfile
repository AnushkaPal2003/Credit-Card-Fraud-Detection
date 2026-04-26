FROM python:3.10

WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project files
COPY . .

# expose mlflow port (optional for later UI)
EXPOSE 5000

# run script
CMD ["python", "credit.py"]