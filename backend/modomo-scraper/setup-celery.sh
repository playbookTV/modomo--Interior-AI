#!/bin/bash

# Celery Setup Script for Modomo Dataset Scraper
# This script helps set up and start Celery workers for development

set -e

echo "🔧 Modomo Celery Setup Script"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Redis is running
check_redis() {
    print_status "Checking Redis connection..."
    
    if command -v redis-cli >/dev/null 2>&1; then
        if redis-cli ping >/dev/null 2>&1; then
            print_success "Redis is running and accessible"
            return 0
        else
            print_error "Redis is not running or not accessible"
            return 1
        fi
    else
        print_warning "redis-cli not found. Please ensure Redis is installed and running"
        return 1
    fi
}

# Check if required environment variables are set
check_env() {
    print_status "Checking environment variables..."
    
    local missing_vars=()
    
    if [ -z "$REDIS_URL" ]; then
        missing_vars+=("REDIS_URL")
    fi
    
    if [ -z "$SUPABASE_URL" ]; then
        missing_vars+=("SUPABASE_URL")
    fi
    
    if [ -z "$SUPABASE_ANON_KEY" ]; then
        missing_vars+=("SUPABASE_ANON_KEY")
    fi
    
    if [ ${#missing_vars[@]} -eq 0 ]; then
        print_success "All required environment variables are set"
        return 0
    else
        print_error "Missing environment variables: ${missing_vars[*]}"
        echo "Please set the following variables:"
        for var in "${missing_vars[@]}"; do
            echo "  export $var=your_value"
        done
        return 1
    fi
}

# Install Celery dependencies
install_deps() {
    print_status "Installing Celery dependencies..."
    
    # Try minimal requirements first to avoid conflicts
    if [ -f "requirements-minimal-celery.txt" ]; then
        print_status "Trying minimal Celery requirements (avoids gradio conflicts)..."
        if pip install -r requirements-minimal-celery.txt; then
            print_success "Minimal Celery dependencies installed successfully"
            return 0
        else
            print_warning "Minimal requirements failed, trying full requirements..."
        fi
    fi
    
    # Fallback to full requirements
    if [ -f "requirements-celery.txt" ]; then
        print_status "Installing full Celery requirements..."
        if pip install -r requirements-celery.txt; then
            print_success "Celery dependencies installed"
        else
            print_error "Full requirements failed. Trying core packages only..."
            pip install "celery[redis]>=5.3.0" flower>=2.0.0 "websockets>=11.0,<12.0"
            print_success "Core Celery packages installed"
        fi
    else
        print_warning "requirements-celery.txt not found. Installing core Celery packages..."
        pip install "celery[redis]>=5.3.0" flower>=2.0.0 "websockets>=11.0,<12.0"
        print_success "Core Celery packages installed"
    fi
}

# Start Redis if not running
start_redis() {
    if ! check_redis; then
        print_status "Attempting to start Redis..."
        
        # Try different ways to start Redis
        if command -v brew >/dev/null 2>&1; then
            print_status "Starting Redis with Homebrew..."
            brew services start redis
        elif command -v systemctl >/dev/null 2>&1; then
            print_status "Starting Redis with systemctl..."
            sudo systemctl start redis
        elif command -v docker >/dev/null 2>&1; then
            print_status "Starting Redis with Docker..."
            docker run -d --name modomo-redis -p 6379:6379 redis:7-alpine
        else
            print_error "Could not start Redis automatically. Please start Redis manually."
            exit 1
        fi
        
        # Wait a moment and check again
        sleep 2
        if check_redis; then
            print_success "Redis started successfully"
        else
            print_error "Failed to start Redis"
            exit 1
        fi
    fi
}

# Start Celery worker
start_worker() {
    local worker_type=${1:-"general"}
    
    print_status "Starting Celery $worker_type worker..."
    
    case $worker_type in
        "general")
            celery -A celery_app worker --loglevel=info --concurrency=2 &
            ;;
        "ai")
            celery -A celery_app worker --loglevel=info --concurrency=1 -Q ai_processing,detection &
            ;;
        "scraping")
            celery -A celery_app worker --loglevel=info --concurrency=1 -Q scraping,import &
            ;;
        "color")
            celery -A celery_app worker --loglevel=info --concurrency=2 -Q color_processing,classification &
            ;;
        *)
            print_error "Unknown worker type: $worker_type"
            exit 1
            ;;
    esac
    
    local worker_pid=$!
    echo $worker_pid > ".celery-${worker_type}-worker.pid"
    print_success "Celery $worker_type worker started (PID: $worker_pid)"
}

# Start Celery beat
start_beat() {
    print_status "Starting Celery beat scheduler..."
    celery -A celery_app beat --loglevel=info &
    local beat_pid=$!
    echo $beat_pid > ".celery-beat.pid"
    print_success "Celery beat started (PID: $beat_pid)"
}

# Start Flower monitoring
start_flower() {
    print_status "Starting Flower monitoring interface..."
    celery -A celery_app flower --port=5555 &
    local flower_pid=$!
    echo $flower_pid > ".celery-flower.pid"
    print_success "Flower started (PID: $flower_pid)"
    print_status "Access Flower at: http://localhost:5555"
}

# Stop all Celery processes
stop_celery() {
    print_status "Stopping Celery processes..."
    
    # Stop workers
    for pidfile in .celery-*-worker.pid; do
        if [ -f "$pidfile" ]; then
            local pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                print_success "Stopped worker (PID: $pid)"
            fi
            rm -f "$pidfile"
        fi
    done
    
    # Stop beat
    if [ -f ".celery-beat.pid" ]; then
        local pid=$(cat ".celery-beat.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            print_success "Stopped beat scheduler (PID: $pid)"
        fi
        rm -f ".celery-beat.pid"
    fi
    
    # Stop flower
    if [ -f ".celery-flower.pid" ]; then
        local pid=$(cat ".celery-flower.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            print_success "Stopped Flower (PID: $pid)"
        fi
        rm -f ".celery-flower.pid"
    fi
    
    # Clean up any remaining celery processes
    pkill -f "celery.*worker" 2>/dev/null || true
    pkill -f "celery.*beat" 2>/dev/null || true
    pkill -f "celery.*flower" 2>/dev/null || true
    
    print_success "All Celery processes stopped"
}

# Show status of Celery processes
show_status() {
    print_status "Celery Status:"
    echo ""
    
    # Check workers
    echo "Workers:"
    for pidfile in .celery-*-worker.pid; do
        if [ -f "$pidfile" ]; then
            local pid=$(cat "$pidfile")
            local worker_type=$(basename "$pidfile" .pid | sed 's/.celery-\(.*\)-worker/\1/')
            if kill -0 "$pid" 2>/dev/null; then
                echo "  ✅ $worker_type worker (PID: $pid) - RUNNING"
            else
                echo "  ❌ $worker_type worker (PID: $pid) - STOPPED"
                rm -f "$pidfile"
            fi
        fi
    done
    
    # Check beat
    if [ -f ".celery-beat.pid" ]; then
        local pid=$(cat ".celery-beat.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ✅ Beat scheduler (PID: $pid) - RUNNING"
        else
            echo "  ❌ Beat scheduler (PID: $pid) - STOPPED"
            rm -f ".celery-beat.pid"
        fi
    else
        echo "  ❌ Beat scheduler - NOT STARTED"
    fi
    
    # Check flower
    if [ -f ".celery-flower.pid" ]; then
        local pid=$(cat ".celery-flower.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ✅ Flower monitoring (PID: $pid) - RUNNING"
            echo "      URL: http://localhost:5555"
        else
            echo "  ❌ Flower monitoring (PID: $pid) - STOPPED"
            rm -f ".celery-flower.pid"
        fi
    else
        echo "  ❌ Flower monitoring - NOT STARTED"
    fi
    
    echo ""
    
    # Check Redis
    if check_redis; then
        echo "✅ Redis - RUNNING"
    else
        echo "❌ Redis - NOT RUNNING"
    fi
}

# Main script logic
case "${1:-help}" in
    "setup")
        print_status "Setting up Celery environment..."
        check_env || exit 1
        install_deps
        start_redis
        print_success "Celery environment setup complete!"
        ;;
    "start")
        worker_type=${2:-"general"}
        check_env || exit 1
        start_redis
        start_worker "$worker_type"
        ;;
    "start-all")
        check_env || exit 1
        start_redis
        start_worker "general"
        sleep 1
        start_worker "ai" 
        sleep 1
        start_worker "scraping"
        sleep 1
        start_beat
        sleep 1
        start_flower
        print_success "All Celery services started!"
        ;;
    "stop")
        stop_celery
        ;;
    "restart")
        stop_celery
        sleep 2
        check_env || exit 1
        start_redis
        start_worker "general"
        start_beat
        start_flower
        ;;
    "status")
        show_status
        ;;
    "flower")
        check_env || exit 1
        start_redis
        start_flower
        ;;
    "docker")
        print_status "Starting Celery with Docker Compose..."
        docker-compose -f docker-compose.celery.yml up -d
        print_success "Celery services started with Docker"
        print_status "Access Flower at: http://localhost:5555"
        ;;
    "docker-stop")
        print_status "Stopping Celery Docker services..."
        docker-compose -f docker-compose.celery.yml down
        print_success "Celery Docker services stopped"
        ;;
    "logs")
        if [ -f "docker-compose.celery.yml" ]; then
            docker-compose -f docker-compose.celery.yml logs -f
        else
            print_error "docker-compose.celery.yml not found"
        fi
        ;;
    "help"|*)
        echo "Modomo Celery Setup Script"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  setup                 Set up Celery environment (install deps, start Redis)"
        echo "  start [worker_type]   Start a specific worker (general, ai, scraping, color)"
        echo "  start-all            Start all workers, beat, and flower"
        echo "  stop                 Stop all Celery processes"
        echo "  restart              Restart all Celery processes"
        echo "  status               Show status of all Celery processes"
        echo "  flower               Start only Flower monitoring"
        echo "  docker               Start Celery with Docker Compose"
        echo "  docker-stop          Stop Celery Docker services"
        echo "  logs                 Show Docker logs"
        echo "  help                 Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 setup             # Initial setup"
        echo "  $0 start-all         # Start everything"
        echo "  $0 start ai          # Start only AI worker"
        echo "  $0 docker            # Use Docker Compose"
        echo ""
        ;;
esac