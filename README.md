# Cyber Threat Predictor

A Django web application that demonstrates a machine-learning workflow for classifying network-traffic records as one of two attack types — **Packet Drop** or **Packet Hijacking** — using classical scikit-learn models. Built as an academic / learning project.

> **Educational proof-of-concept.** The model trains on a dataset that is **not included** in this repository, and the accuracy figures shown in the app are illustrative, not a validated benchmark (see [Limitations](#limitations)). Don't use this for real security decisions.

---

## What it does

Two Django apps split the experience by role:

- **Remote_User** — end users register, sign in, submit a network-traffic record, and receive a predicted label (*Packet Drop* / *Packet Hijacking*).
- **Service_Provider** — analyst/admin side: trains the classifiers, browses the prediction ledger, compares per-model accuracy, views charts, and exports results to Excel.

**Prediction pipeline:** a text field is vectorized with `CountVectorizer`, split with `train_test_split`, and classified with several scikit-learn models — Multinomial Naive Bayes, Linear SVM, Logistic Regression, Decision Tree, and Extra Trees.

---

## Setup

Requires **Python 3** and **MySQL**.

```bash
# 1. Dependencies (no requirements.txt is committed — see Limitations)
pip install django pandas scikit-learn numpy mysqlclient python-dotenv xlwt

# 2. Database — create the schema and import the dump
mysql -u root -e "CREATE DATABASE cyber_threat_predictor"
mysql -u root cyber_threat_predictor < Database/database.sql

# 3. Secrets — Django reads SECRET_KEY from a .env file (python-dotenv)
echo "SECRET_KEY=your-django-secret-key" > "cyber threat predictor/.env"

# 4. Run
cd "cyber threat predictor"
python manage.py migrate
python manage.py runserver
```

DB connection settings live in `cyber threat predictor/cyber_threat_predictor/settings.py` (defaults to `root`@`127.0.0.1:3306`, empty password — change for your environment).

**Dataset:** the training code reads `IIoT_Network_Datasets.csv` (and `Labled_data.csv`). These are **not included** — supply them in the Django project directory before training/predicting.

---

## Tech stack

Django 3 · scikit-learn · pandas · NumPy · MySQL · python-dotenv · xlwt (Excel export)

---

## Limitations

Read these before judging the project:

- **Dataset not included.** The app cannot train or predict until you supply `IIoT_Network_Datasets.csv`. The repo ships only a MySQL schema dump (`Database/database.sql`), not the ML data.
- **Accuracy is not a validated benchmark.** Figures shown in the app are computed at runtime on a random split and should be treated as illustrative only.
- **Models retrain on every request** instead of loading a saved artifact — fine for a demo, not for production.
- **Single feature.** Prediction uses one vectorized text field, not the full set of network attributes the form collects.
- **Security posture is academic.** `DEBUG=True`, permissive defaults, and minimal credential handling — a learning scaffold, not hardened code.
- **No tests, no CI, no `requirements.txt`.**

---

## Repo hygiene

This repository previously committed a virtualenv (`venv/`), IDE config (`.idea/`), and OS metadata (`.DS_Store`). Those are removed from the tree and added to `.gitignore`. They remain in earlier git history — a history rewrite (e.g. `git filter-repo`) would be a separate, deliberate step.
