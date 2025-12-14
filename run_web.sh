#!/bin/bash
echo "Mengecek Flask..."
if ! python -c "import flask" &> /dev/null; then
    echo "Menginstall Flask..."
    pip install flask
fi

echo ""
echo "=================================================="
echo "  WEB SERVER BRAIN TUMOR DETECTION BERJALAN!  "
echo "=================================================="
echo "Buka browser Anda dan akses alamat di bawah ini:"
echo "http://localhost:5000"
echo "=================================================="
echo "Tekan CTRL+C untuk berhenti."
echo ""

python app.py
