💳 Credit Card Fraud Detection

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using unsupervised anomaly detection.

🚀 Live App

https://credit-card-fraud-detection-balawe2klts3je3i7bnmqw.streamlit.app/

💻 GitHub

https://github.com/AnushkaPal2003/Credit-Card-Fraud-Detection

🛠 Tech Stack

Python

Scikit-learn (Isolation Forest)

MLflow

Streamlit

Docker

🧠 Approach

Treat fraud detection as an anomaly detection problem

Train Isolation Forest on normal transaction behavior

Detect rare fraudulent patterns

Track experiments using MLflow

Deploy as a Streamlit app and Docker container

🐳 Run with Docker
docker pull anushkapal/detectfraud
docker run -p 8501:8501 anushkapal/detectfraud


Open: http://localhost:8501

📊 Output
True fraud ratio      : 0.00173
Detected anomaly ratio: 0.00170

👩‍💻 Author

Anushka Pal
Aspiring Data Scientist