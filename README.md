# SHL Assessment Recommender

An AI-powered conversational recommendation system that helps recruiters and hiring managers discover the most relevant SHL Individual Test Solutions through natural language interaction.

## Overview

Traditional assessment catalogs rely heavily on keyword search and rigid filters, which assumes users already know the exact terminology or assessment names they need.

This project solves that problem by providing a conversational recommendation agent that:

* Understands vague hiring intents
* Asks clarifying questions when needed
* Recommends only valid SHL Individual Test Solutions
* Prevents hallucinated recommendations
* Supports refinement and comparison workflows
* Returns structured API responses

The system is built using FastAPI, Gemini API, and a validated SHL assessment catalog.

---

# Features

* Conversational SHL assessment recommendation agent
* FastAPI backend with REST endpoints
* Hallucination prevention through catalog validation
* Multi-turn hiring conversations
* Grounded recommendations using only SHL catalog data
* Dockerized deployment
* Render cloud deployment support
* Offline evaluation framework with Recall@10 metrics
* Swagger/OpenAPI documentation

---

# Tech Stack

* Python 3.11+
* FastAPI
* Gemini API
* Pydantic
* HTTPX
* Docker
* Render

---

# Project Structure

```text
shl-assessment-recommender/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── catalog.py
│
├── tests/
│   └── evaluate.py
│
├── Dockerfile
├── render.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# API Endpoints

## Health Check

```http
GET /health
```

### Example Response

```json
{
  "status": "ok"
}
```

---

## Conversational Recommendation Endpoint

```http
POST /chat
```

### Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a senior Java backend engineer with AWS and Docker experience"
    }
  ]
}
```

### Example Response

```json
{
  "reply": "Based on your need for a senior Java backend engineer with AWS and Docker experience, I recommend the following SHL Individual Test Solutions.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    },
    {
      "name": "Docker (New)",
      "url": "https://www.shl.com/products/product-catalog/view/docker-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

---

# Recommendation Validation Layer

One of the core components of the project is a validation layer that prevents hallucinated recommendations.

Every recommendation returned by the LLM is validated against the internal SHL catalog before being sent to the user.

The validation layer:

* Rejects hallucinated assessments
* Ensures URLs exist in catalog
* Ensures names match catalog entries
* Enriches recommendations with canonical catalog metadata
* Restricts outputs to SHL Individual Test Solutions only

---

# Conversational Capabilities

The agent supports:

* Role-based recommendations
* Skill-specific recommendations
* Leadership and personality assessment flows
* Clarification questions
* Refinement requests
* Assessment comparison
* Multi-turn conversations

Example intents:

* "Hiring a senior Java backend engineer"
* "Need customer service agents with strong communication skills"
* "Looking for graduate management trainees"
* "Compare cognitive and personality assessments"

---

# Evaluation Framework

The project includes an offline evaluation harness using reference hiring conversations.

Metrics used:

* Recall@10
* Precision

## Final Evaluation Results

| Metric            | Score |
| ----------------- | ----- |
| Average Recall@10 | 0.769 |
| Average Precision | 0.858 |

The evaluation framework measures recommendation quality against manually curated expected assessment sets.

---

# Local Setup

## 1. Clone Repository

```bash
git clone https://github.com/Muheet-Mehraj/shl-assessment-recommender.git
cd shl-assessment-recommender
```

---

## 2. Create Virtual Environment

### Windows (Git Bash)

```bash
python -m venv venv
source venv/Scripts/activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## 5. Run Application

```bash
uvicorn app.main:app --reload
```

---

# API Documentation

Once running locally:

```text
http://127.0.0.1:8000/docs
```

---

# Running Evaluation

```bash
python -m tests.evaluate
```

This generates:

* Recall@10 metrics
* Precision metrics
* Missed recommendations
* Extra recommendations
* Evaluation artifact JSON

---

# Docker Deployment

## Build Image

```bash
docker build -t shl-assessment-recommender .
```

## Run Container

```bash
docker run -p 8000:8000 shl-assessment-recommender
```

---

# Live Deployment

## Production API

[https://shl-assessment-recommender-pen6.onrender.com](https://shl-assessment-recommender-pen6.onrender.com)

## Swagger Docs

[https://shl-assessment-recommender-pen6.onrender.com/docs](https://shl-assessment-recommender-pen6.onrender.com/docs)

---

# Design Decisions

## Why Validation Layer?

LLMs can hallucinate product names or generate invalid URLs.

The validation layer ensures:

* grounded recommendations
* deterministic outputs
* safe production behavior
* catalog consistency

---

## Why Conversational Retrieval?

Recruiters often describe hiring needs naturally rather than using exact assessment terminology.

The conversational agent improves usability by mapping vague intents into grounded SHL assessment recommendations.

---

# Future Improvements

Potential extensions include:

* vector search retrieval
* embedding-based ranking
* session memory
* assessment comparison scoring
* frontend UI
* streaming responses
* caching layer
* analytics dashboard

---

# Author

Muheet Mehraj

B.Tech CSE Student | AI/ML Enthusiast | Backend & LLM Systems

GitHub:
[https://github.com/Muheet-Mehraj](https://github.com/Muheet-Mehraj)
