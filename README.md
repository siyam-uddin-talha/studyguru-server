# StudyGuru Pro - AI-Powered Study Assistant

StudyGuru Pro is an AI-powered study assistant that helps students analyze documents, images, and study materials using advanced AI technology. Built with FastAPI, Strawberry GraphQL, and OpenAI integration.

## Features

- **AI Document Analysis** - Upload images or PDFs and get intelligent analysis
- **Points-Based System** - Flexible pricing with points for AI usage
- **Multiple Question Types** - Supports MCQ and written question analysis
- **Language Detection** - Automatically detects and responds in the document's language
- **File Compression** - Automatic file compression before storage
- **Subscription Plans** - Free, Basic, and Pro plans with different point allocations

## Tech Stack

- **FastAPI** - Modern, fast web framework for building APIs
- **Strawberry GraphQL** - Modern GraphQL library for Python
- **SQLAlchemy** - Async ORM for database operations
- **Alembic** - Database migration tool
- **OpenAI API** - AI-powered document analysis
- **Paddle** - Payment processing
- **AWS S3** - File storage
- **UV** - Fast Python package installer and resolver

## Project Structure

```
app/
├── api/                    # REST API routes
├── core/                   # Core configuration and database
├── graphql/               # GraphQL schema and resolvers
│   ├── resolvers/         # GraphQL resolvers
│   └── types/             # GraphQL type definitions
├── helpers/               # Utility functions
├── models/                # SQLAlchemy models
├── services/              # Business logic services
└── workers/               # Background tasks
alembic/                   # Database migrations
```

## Installation

1. **Install UV (if not already installed)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**

```bash
git clone <repository-url>
cd studyguru-pro
```

3. **Install dependencies**

```bash
uv sync
```

4. **Environment Configuration**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Database Setup**

```bash
# Run init
alembic revision --autogenerate -m "init"

# Run migrations
uv run alembic upgrade head

# Seed the database
uv run python app/database/seed.py
```

6. **Run the application**

```bash
# Development
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 5000

# Production
uv run uvicorn app.main:app --host 0.0.0.0 --port 5000
```

## API Endpoints

### GraphQL

- **POST** `/graphql` - GraphQL endpoint

### REST Endpoints

- **POST** `/webhook/paddle` - Paddle webhooks
<!-- - **POST** `/doc-material/upload` - Upload and analyze documents -->

## Subscription Plans

### Free Plan

- **Price**: $0
- **Points**: 30 points on signup
- **Token Limit**: 1,000 tokens per analysis

### Basic Plan

- **Price**: $1/month
- **Points**: 100 points per month

### Pro Plan

- **Price**: $5/month
- **Points**: 700 points per month

### Point Add-ons

- **Price**: $0.01 per point
- **Minimum**: 100 points

## Points System

- Points are used for AI document analysis
- 1 point ≈ 100 tokens (adjustable)
- Free users get 30 points on signup
- Paid plans receive monthly point allocations
- Additional points can be purchased as add-ons

## Document Analysis

The AI analysis provides:

1. **Language Detection** - Automatically detects document language
2. **Question Type Classification** - MCQ vs written questions
3. **Structured Responses**:
   - **MCQ**: JSON array with questions, options, answers, and explanations
   - **Written**: Organized explanatory content
4. **Metadata**: Title, summary, token usage

## GraphQL Schema

### Key Queries

- `account` - Get current user account and points
- `subscriptions` - Get available subscription plans
- `docMaterials` - Get user's analyzed documents
- `pointsHistory` - Get points transaction history

### Key Mutations

- `register` - User registration
- `login` - User login
- `uploadDocument` - Upload and analyze document
- `purchasePoints` - Buy additional points

## Database Models

- **User** - User accounts with points tracking
- **Subscription** - Available plans and add-ons
- **Interaction** - Analyzed documents and results
- **PointTransaction** - Points usage history
- **Media** - File storage information

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black .
uv run isort .
```

### Type Checking

```bash
uv run mypy .
```

## Deployment

1. Set environment variables for production
2. Set `ENVIRONMENT=production` in `.env`
3. Configure proper database URL
4. Set up SSL certificates
5. Use a production ASGI server:

```bash
uv run gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Environment Variables

Key environment variables (see `.env.example`):

- `DATABASE_URL` - MySQL connection string
- `JWT_SECRET_KEY` - JWT signing secret
- `PADDLE_API_KEY` - Paddle payment API key
- `OPENAI_API_KEY` - OpenAI API key
- `AWS_*` - S3 storage configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]
