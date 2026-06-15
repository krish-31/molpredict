#!/bin/bash
# start.sh — Start both the FastAPI backend and Vite React frontend concurrently.

# 1. Kill any existing processes running on ports 8000 and 5173
echo "Checking for processes on ports 8000 and 5173..."
npx kill-port 8000 5173 2>/dev/null || true

# 2. Activate Python environment if needed and launch backend
echo "Starting FastAPI backend on http://localhost:8000..."
if [ -n "$CONDA_PREFIX" ]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python"
fi
$PYTHON_BIN -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# 3. Start Vite frontend
echo "Starting React/Vite frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# 4. Wait for termination
trap "echo 'Stopping servers...'; kill $BACKEND_PID; kill $FRONTEND_PID; exit" INT TERM EXIT
wait
