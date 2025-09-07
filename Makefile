# Nirvana Backend - Makefile
# Repository: nirvana-backend

.PHONY: help setup dev start stop clean build logs test db-reset

# Default target
help:
	@echo "Nirvana Backend - Development Commands"
	@echo "======================================"
	@echo "Available targets:"
	@echo "  setup       - Setup development environment"
	@echo "  start       - Start backend services (access via nginx only)"
	@echo "  stop        - Stop all services"
	@echo "  build       - Build Docker images"
	@echo "  logs        - Show backend logs"
	@echo "  test        - Run backend tests"
	@echo "  db-reset    - Reset database (WARNING: deletes all data)"
	@echo "  clean       - Clean containers and volumes"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Docker and Docker Compose installed"
	@echo "  - .env file configured (copy from env.template)"
	@echo "  - pip.conf configured for Azure Artifacts"
	@echo "  - nirvana_network created (run 'make setup' in nginx repo)"

# Setup development environment
setup:
	@echo "=== Setting up Nirvana Backend ==="
	@if [ ! -f ".env" ]; then \
		echo "Creating .env from template..."; \
		cp nirvana-backend.env.template .env; \
		echo "Please edit .env file with your configuration"; \
	fi
	@echo "Building Docker images..."
	@docker compose build
	@echo "Setup complete!"

# Start development server with auto-reload
dev:
	@echo "=== Starting Development Server ==="
	@echo "Backend will be available at: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@docker compose up backend

# Start all services
start:
	@echo "=== Starting All Backend Services ==="
	@docker compose up -d
	@echo "Services started:"
	@echo "  - Database: localhost:5432"
	@echo "  - Redis: localhost:6379"
	@echo "  - Backend API: localhost:8000"

# Stop all services
stop:
	@echo "=== Stopping Backend Services ==="
	@docker compose down

# Build Docker images
build:
	@echo "=== Building Backend Images ==="
	@docker compose build --no-cache

# Build with network workaround
build-fix:
	@echo "=== Building Backend Images (Network Fix) ==="
	@docker build --network=host --build-arg DOCKER_BUILDKIT=0 -t nirvana-backend .

# Show backend logs
logs:
	@docker compose logs -f backend

# Run tests (when implemented)
test:
	@echo "=== Running Backend Tests ==="
	@docker compose exec backend python -m pytest tests/ -v

# Reset database (WARNING: deletes all data)
db-reset:
	@echo "=== Resetting Database ==="
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ]
	@docker compose down
	@docker volume rm nirvana-backend_db_data || true
	@docker compose up -d db
	@echo "Database reset complete"

# Clean containers and volumes
clean:
	@echo "=== Cleaning Backend Environment ==="
	@docker compose down -v
	@docker system prune -f
	@echo "Cleanup complete"

# Health check
health:
	@echo "=== Backend Health Check ==="
	@curl -s http://localhost:8000/api/health | jq . || echo "Backend not responding"

# Database shell
db-shell:
	@echo "=== Connecting to Database ==="
	@docker compose exec db psql -U nirvana -d nirvana