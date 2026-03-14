#!/bin/bash
# Manava Analytics — Setup Script
# Run once: bash setup.sh

echo ""
echo "╔══════════════════════════════════════╗"
echo "  MANAVA Analytics — Setup"
echo "╚══════════════════════════════════════╝"
echo ""

echo "► Installing Python packages..."
pip3 install flask flask-cors pandas numpy scikit-learn statsmodels openpyxl python-pptx reportlab --quiet

echo ""
echo "✓ All packages installed!"
echo ""
echo "╔══════════════════════════════════════╗"
echo "  HOW TO START MANAVA EACH SESSION"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Terminal 1 — Ollama (AI chat):"
echo "  pkill ollama && sleep 2 && OLLAMA_ORIGINS=\"*\" OLLAMA_HOST=\"0.0.0.0:11434\" ollama serve"
echo ""
echo "  Terminal 2 — Analytics server:"
echo "  cd ~/Desktop/myai && python3 server.py"
echo ""
echo "  Terminal 3 — Web server:"
echo "  cd ~/Desktop/myai && python3 -m http.server 8080"
echo ""
echo "  Then open Chrome → http://localhost:8080"
echo ""
echo "  Click the ⚡ pulse icon in the top bar to open Data Analyst Mode"
echo ""
