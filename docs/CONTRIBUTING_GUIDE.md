# 🤝 Contributing Guide

Welcome to the NYC Taxi ETA Prediction project.

This document explains how team members should contribute to the repository.

---

# Branching Strategy

⚠️ Do NOT work directly on the `main` branch.

Every new feature should be developed in its own feature branch.

Example:

```
main
│
├── feature/data-ingestion
├── feature/data-validation
├── feature/model-training
└── feature/api
```

---

# Workflow

## Step 1

Pull the latest changes.

```bash
git checkout main
git pull origin main
```

---

## Step 2

Create a new feature branch.

Example:

```bash
git checkout -b feature/data-ingestion
```

---

## Step 3

Work only in your feature branch.

Commit your changes regularly.

Example:

```bash
git add .
git commit -m "Implement data ingestion pipeline"
```

---

## Step 4

Push your branch.

```bash
git push origin feature/data-ingestion
```

---

## Step 5

Create a Pull Request (PR)

Open a Pull Request from:

```
feature/data-ingestion
        ↓
main
```

Do NOT merge your own Pull Request.

Let other review and merge it.

---

# Commit Message Guidelines

Use clear and meaningful commit messages.

Good examples:

```
Add data ingestion pipeline

Implement data validation

Add MLflow experiment tracking

Fix preprocessing bug

Update project documentation
```

Avoid messages like:

```
update

changes

fix

final

latest
```

---

# Repository Rules

✅ Always pull latest changes before starting work.

✅ Work only in your assigned module.

✅ Create a separate feature branch for every module.

✅ Keep commits small and meaningful.

✅ Raise a Pull Request after completing your work.

❌ Never commit directly to `main`.

❌ Never commit datasets.

❌ Never commit virtual environments.

❌ Never commit secrets or credentials.

---

# Project Structure

```
main
│
├── src/
├── scripts/
├── tests/
├── configs/
├── docs/
├── models/
├── reports/
├── data/
└── notebooks/
```

---

# Need Help?

If you're unsure about anything, let's discuss it.