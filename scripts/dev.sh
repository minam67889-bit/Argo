#!/usr/bin/env bash
# Argo development helpers
set -e

case "$1" in
  install)
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    ;;
  run)
    echo "🚀 Starting Argo server on http://localhost:${ARGO_PORT:-8000}"
    python -m app.main
    ;;
  cli)
    shift
    echo "💻 Running CLI agent..."
    python -m cli.agent "$@"
    ;;
  test)
    echo "🧪 Running tests..."
    python tests/run_all.py
    ;;
  clean)
    echo "🧹 Cleaning generated files..."
    rm -rf data/ workspaces/ __pycache__/ app/__pycache__/ app/*/__pycache__/ cli/__pycache__/ tests/__pycache__/
    echo "Done."
    ;;
  *)
    echo "Usage: $0 {install|run|cli|test|clean}"
    echo ""
    echo "Commands:"
    echo "  install    Install Python dependencies"
    echo "  run        Start the web server"
    echo "  cli [...]  Run the CLI agent with optional task"
    echo "  test       Run all tests"
    echo "  clean      Remove generated files"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  export LLM_API_KEY=sk-or-..."
    echo "  $0 run"
    echo "  $0 cli \"fix the bug in main.py\""
    echo "  $0 test"
    exit 1
    ;;
esac
