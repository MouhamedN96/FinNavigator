# FinNavigator Backend - Missing Components
# This file documents all backend requirements

## 1. Database Schema (PostgreSQL)

### Users Table
- id, email, password_hash, created_at, updated_at
- api_key, rate_limit_tier
- preferences (JSON)

### Portfolios Table
- id, user_id, name, created_at
- positions (JSON)

### Alerts Table
- id, user_id, type, conditions, recipients
- enabled, cooldown_minutes
- last_triggered, trigger_count

### Tasks Table
- id, user_id, task_type, schedule, config
- enabled, last_run, next_run

### Execution History Table
- id, task_id, status, started_at, completed_at
- output, error

## 2. API Endpoints (FastAPI)

### Auth
- POST /auth/register
- POST /auth/login
- POST /auth/refresh

### Agents
- POST /agent/query
- POST /agent/chat
- GET /agent/history

### Portfolios
- GET /portfolio
- POST /portfolio
- PUT /portfolio/{id}
- DELETE /portfolio/{id}

### Alerts
- GET /alerts
- POST /alerts
- PUT /alerts/{id}
- DELETE /alerts/{id}

### Tasks (Cron)
- GET /tasks
- POST /tasks
- PUT /tasks/{id}
- DELETE /tasks/{id}
- POST /tasks/{id}/trigger

### Webhooks
- POST /webhooks/trigger/{task_id}

## 3. Required Environment Variables

```
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/finavigator
REDIS_URL=redis://localhost:6379

# API Keys
NVIDIA_API_KEY=
SEC_API_KEY=
TELEGRAM_BOT_TOKEN=
DISCORD_BOT_TOKEN=
SLACK_BOT_TOKEN=

# Auth
JWT_SECRET_KEY=
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## 4. Dependencies to Add

fastapi>=0.110.0
uvicorn>=0.27.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
pydantic>=2.6.0
python-jose>=3.3.0
passlib>=1.7.4
bcrypt>=4.1.0
redis>=5.0.0
httpx>=0.28.0