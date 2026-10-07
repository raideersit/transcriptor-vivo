#!/usr/bin/env bash
# Instalador del Transcriptor en vivo (Ubuntu / Linux Mint / Debian)
# Uso:  bash instalar.sh
set -e

DIR="$HOME/.local/share/transcriptor"
BIN="$HOME/.local/bin"
AQUI="$(cd "$(dirname "$0")" && pwd)"

echo "==> Instalando dependencias del sistema (pide tu contraseña)..."
sudo apt update
sudo apt install -y python3-venv python3-pip pulseaudio-utils ffmpeg

echo "==> Creando entorno de Python en $DIR ..."
mkdir -p "$DIR" "$BIN"
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --upgrade pip -q
"$DIR/venv/bin/pip" install faster-whisper numpy -q

cp "$AQUI/transcriptor.py" "$DIR/transcriptor.py"

echo "==> Creando el comando 'transcribir' ..."
cat > "$BIN/transcribir" <<EOF
#!/usr/bin/env bash
exec "$DIR/venv/bin/python" "$DIR/transcriptor.py" "\$@"
EOF
chmod +x "$BIN/transcribir"

echo "==> Descargando el modelo base.en (una sola vez, ~150 MB)..."
"$DIR/venv/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
     echo "   (Se agregó ~/.local/bin al PATH: cierra y abre la terminal)";;
esac

echo
echo "Listo. Abre una terminal y escribe:  transcribir"
