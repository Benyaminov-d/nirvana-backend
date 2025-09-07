# Nirvana App - Environment Variables Split

This directory contains the properly split environment variables for the three separate repositories:

- `nirvana-backend` - Backend API and business logic
- `nirvana-website` - Frontend SPA (React/Vite)  
- `nirvana-nginx` - Reverse proxy, SSL, and orchestration

## Files Overview

### Production Environment Files (with actual values)
- `nirvana-backend.env` - All backend-related environment variables
- `nirvana-website.env` - Frontend-specific environment variables
- `nirvana-nginx.env` - Nginx and deployment configuration
- `shared.env` - Variables that may be needed across repositories

### Template Files (for developers)
- `nirvana-backend.env.template` - Backend template without sensitive data
- `nirvana-website.env.template` - Website template
- `nirvana-nginx.env.template` - Nginx template

## Environment Variables Categorization

### Backend Variables (nirvana-backend)
- **Database**: PostgreSQL connection, credentials
- **Redis**: Cache configuration  
- **External APIs**: EODHD, OpenAI, Azure Functions
- **Authentication**: JWT secrets, basic auth, allowed origins
- **Business Logic**: Compass Score parameters, CVaR computation settings
- **Email**: SMTP configuration
- **Processing**: Service Bus, worker settings
- **Startup**: Bootstrap and initialization flags
- **Assistant**: AI system prompts

### Website Variables (nirvana-website)  
- **API Connection**: Backend URL configuration (`VITE_API_BASE`)
- **Development**: Vite dev server settings
- **Build**: Target and output configuration

### Nginx Variables (nirvana-nginx)
- **Domain**: SSL certificates, domain configuration
- **Service URLs**: Backend/frontend container references
- **Proxy**: Timeouts, rate limiting
- **Security**: Basic auth for protected routes
- **Infrastructure**: Database/Redis for orchestration

## Usage in Each Repository

### For Backend Repository (nirvana-backend)

```bash
# Copy template and customize
cp nirvana-backend.env.template .env
# Edit .env with your specific values
# Key variables to configure:
# - POSTGRES_PASSWORD
# - EODHD_API_KEY  
# - BASIC_AUTH_PASS
# - OPENAI_API_KEY (optional)
```

Docker Compose usage:
```yaml
services:
  backend:
    build: .
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql+psycopg2://nirvana:${POSTGRES_PASSWORD}@db:5432/nirvana
      REDIS_URL: redis://redis:6379/0
```

### For Website Repository (nirvana-website)

```bash
# Copy template and customize
cp nirvana-website.env.template .env
# Main variable to configure:
# - VITE_API_BASE (points to your backend)
```

For local development:
```env
VITE_API_BASE=http://localhost:8000
```

For production:
```env
VITE_API_BASE=https://api.nirvana.bm
```

### For Nginx Repository (nirvana-nginx)

```bash
# Copy template and customize  
cp nirvana-nginx.env.template .env
# Key variables to configure:
# - DOMAIN
# - CERTBOT_EMAIL
# - BASIC_AUTH_PASS
# - POSTGRES_PASSWORD (for orchestration)
```

This repository contains the full stack docker-compose.yml:
```yaml
services:
  backend:
    image: ghcr.io/your-org/nirvana-backend:latest
  frontend:
    image: ghcr.io/your-org/nirvana-website:latest
  nginx:
    image: nginx:alpine
    env_file: .env
```

## Migration Guide

### 1. For Backend Developers
1. Use `nirvana-backend.env` in your backend repository
2. Update your docker-compose.yml to reference only backend services (db, redis, backend)
3. Remove frontend-specific variables

### 2. For Frontend Developers  
1. Use `nirvana-website.env` in your website repository
2. Ensure `VITE_API_BASE` points to your backend (localhost for dev, production URL for prod)
3. Frontend docker-compose.yml should only contain the SPA service

### 3. For DevOps/Deployment
1. Use `nirvana-nginx.env` for full stack orchestration
2. Configure domain, SSL, and service discovery
3. This repository orchestrates all services via docker-compose

## Security Considerations

### Sensitive Variables
These variables contain sensitive data and should be secured:
- `POSTGRES_PASSWORD` - Database password
- `EODHD_API_KEY` - External API key
- `OPENAI_API_KEY` - AI service key  
- `BASIC_AUTH_PASS` - Authentication password
- `SB_CONNECTION` - Service Bus connection string
- `NVAR_FUNC_KEY` - Azure Function key

### Best Practices
1. **Never commit actual .env files** to Git
2. **Use templates** for repository setup
3. **Use secrets management** in production (Azure Key Vault, etc.)
4. **Rotate credentials** regularly
5. **Limit access** based on repository needs

## Cross-Repository Communication

### API Endpoints
- Backend exposes API at `/api/*` endpoints
- Frontend consumes API via `VITE_API_BASE` configuration
- Nginx routes `/api/*` to backend, everything else to frontend

### Service Discovery
- **Development**: Use `localhost` URLs
- **Docker Compose**: Use container names (`nirvana_backend`, `nirvana_frontend`)
- **Production**: Use actual domain names or load balancer URLs

### Shared Configuration
Variables that may be needed across repositories:
- `BASIC_AUTH_USER/BASIC_AUTH_PASS` - Used by both nginx and backend
- `NIR_ALLOWED_ORIGINS` - CORS configuration in backend
- `DOMAIN` - SSL and routing configuration

## Environment Validation

Each repository should validate required environment variables on startup:

### Backend Validation
```python
required_vars = [
    'DATABASE_URL', 'REDIS_URL', 'EODHD_API_KEY', 
    'BASIC_AUTH_USER', 'BASIC_AUTH_PASS'
]
```

### Website Validation  
```javascript
if (!import.meta.env.VITE_API_BASE) {
    throw new Error('VITE_API_BASE is required')
}
```

### Nginx Validation
```bash
if [ -z "$DOMAIN" ]; then
    echo "DOMAIN environment variable is required"
    exit 1
fi
```

This split allows each repository to be independently developed, deployed, and maintained while preserving the professional fintech application standards.


