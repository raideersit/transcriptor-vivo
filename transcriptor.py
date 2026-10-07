#!/usr/bin/env python3
"""
Transcriptor en vivo (Linux)
----------------------------
Captura el audio que suena en el equipo (parlantes/audífonos, NO el micrófono),
lo transcribe en inglés con Whisper (faster-whisper, en CPU) y lo va guardando
en un archivo .txt mientras escuchas.

Uso rápido:
    python3 transcriptor.py                  # escucha y transcribe en vivo
    python3 transcriptor.py --modelo small.en  # más preciso, más lento
    python3 transcriptor.py --archivo audio.mp3  # transcribe un archivo completo
    python3 transcriptor.py --listar-fuentes     # muestra las fuentes de audio

Detener: Ctrl+C (transcribe lo último que quedó en memoria y cierra).
"""

import argparse
import datetime as dt
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

RATE = 16000                 # Whisper trabaja a 16 kHz mono
BYTES_POR_MUESTRA = 2        # s16le
FRAME = int(0.1 * RATE)      # ventanas de 100 ms para medir energía


# --------------------------------------------------------------------------
# Captura de audio
# --------------------------------------------------------------------------
def listar_fuentes():
    if not shutil.which("pactl"):
        print("No se encontró 'pactl'. Instala: sudo apt install pulseaudio-utils")
        return
    salida = subprocess.run(["pactl", "list", "short", "sources"],
                            capture_output=True, text=True).stdout
    print("Fuentes de audio disponibles (las que terminan en '.monitor' son")
    print("lo que suena por los parlantes/audífonos):\n")
    for linea in salida.strip().splitlines():
        partes = linea.split("\t")
        if len(partes) >= 2:
            marca = "  <-- salida del sistema" if partes[1].endswith(".monitor") else ""
            print(f"  {partes[1]}{marca}")
    print("\nPor defecto se usa @DEFAULT_MONITOR@ (la salida que estés usando ahora).")


def comando_captura(fuente, archivo_prueba=None):
    """Devuelve el comando que entrega audio crudo s16le 16 kHz mono por stdout."""
    if archivo_prueba:
        # Modo prueba: reproduce un archivo "en tiempo real" como si sonara en el PC
        return ["ffmpeg", "-loglevel", "error", "-re", "-i", archivo_prueba,
                "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"]
    if not shutil.which("parec"):
        sys.exit("No se encontró 'parec'. Instala: sudo apt install pulseaudio-utils")
    return ["parec", f"--device={fuente}", "--format=s16le", f"--rate={RATE}",
            "--channels=1", "--raw", "--latency-msec=100"]


def hilo_lector(proc, cola, detener):
    """Lee el audio del proceso de captura sin bloquear la transcripción."""
    bloque = FRAME * BYTES_POR_MUESTRA
    while not detener.is_set():
        datos = proc.stdout.read(bloque)
        if not datos:
            break
        cola.put(datos)
    cola.put(None)  # fin de la captura


# --------------------------------------------------------------------------
# Fragmentación: cortar el audio en pausas para no partir palabras
# --------------------------------------------------------------------------
def energia_por_frame(audio):
    n = len(audio) // FRAME
    if n == 0:
        return np.array([])
    ventanas = audio[: n * FRAME].reshape(n, FRAME)
    return np.sqrt(np.mean(ventanas ** 2, axis=1))


def buscar_corte(audio, min_seg, max_seg, umbral):
    """
    Devuelve el índice (en muestras) donde cortar, o None si conviene esperar.
    - Desde min_seg, corta en la última pausa encontrada.
    - Si se llega a max_seg sin pausa, corta en el punto más bajo de volumen.
    """
    duracion = len(audio) / RATE
    if duracion < min_seg:
        return None
    rms = energia_por_frame(audio)
    desde = int(min_seg * 0.75 * 10)         # no cortar demasiado al inicio
    silencio = rms < umbral
    # Buscar la última pausa real: al menos 2 ventanas seguidas (200 ms) en silencio
    for i in range(len(rms) - 2, desde - 1, -1):
        if silencio[i] and silencio[i + 1]:
            return (i + 1) * FRAME           # cortar en medio de la pausa
    if duracion >= max_seg:
        zona = rms[int(min_seg * 10): int(max_seg * 10)]
        return (int(min_seg * 10) + int(np.argmin(zona))) * FRAME + FRAME // 2
    return None


# --------------------------------------------------------------------------
# Transcripción
# --------------------------------------------------------------------------
def cargar_modelo(nombre, hilos):
    from faster_whisper import WhisperModel
    print(f"Cargando modelo '{nombre}' (la primera vez se descarga)...", flush=True)
    return WhisperModel(nombre, device="cpu", compute_type="int8", cpu_threads=hilos)


def transcribir(modelo, audio, idioma, contexto, beam):
    segmentos, _ = modelo.transcribe(
        audio,
        language=idioma,
        beam_size=beam,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        initial_prompt=contexto or None,
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segmentos).strip()


def ruta_salida(salida):
    if salida:
        return os.path.abspath(os.path.expanduser(salida))
    carpeta = os.path.expanduser("~/Transcripciones")
    os.makedirs(carpeta, exist_ok=True)
    nombre = dt.datetime.now().strftime("transcripcion_%Y-%m-%d_%H-%M.txt")
    return os.path.join(carpeta, nombre)


def fmt_tiempo(seg):
    m, s = divmod(int(seg), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --------------------------------------------------------------------------
# Modos
# --------------------------------------------------------------------------
def modo_archivo(args):
    modelo = cargar_modelo(args.modelo, args.hilos)
    salida = args.salida or os.path.splitext(args.archivo)[0] + ".txt"
    print(f"Transcribiendo {args.archivo} ...", flush=True)
    segmentos, _ = modelo.transcribe(args.archivo, language=args.idioma,
                                     beam_size=5, vad_filter=True)
    with open(salida, "w", encoding="utf-8") as f:
        for s in segmentos:
            linea = s.text.strip()
            if args.marcas:
                linea = f"[{fmt_tiempo(s.start)}] {linea}"
            print(linea, flush=True)
            f.write(linea + "\n")
    print(f"\nListo. Texto guardado en: {salida}")


def modo_vivo(args):
    modelo = cargar_modelo(args.modelo, args.hilos)
    salida = ruta_salida(args.salida)
    proc = subprocess.Popen(comando_captura(args.fuente, args.probar_con),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cola, detener = queue.Queue(), threading.Event()
    threading.Thread(target=hilo_lector, args=(proc, cola, detener), daemon=True).start()

    def al_salir(*_):
        detener.set()
        proc.terminate()
    signal.signal(signal.SIGINT, al_salir)
    signal.signal(signal.SIGTERM, al_salir)

    print("\n" + "=" * 60)
    print(" Escuchando el audio del sistema. Reproduce lo que quieras.")
    print(f" Guardando en: {salida}")
    print(" Ctrl+C para detener.")
    print("=" * 60 + "\n", flush=True)

    buffer = np.zeros(0, dtype=np.float32)
    inicio_buffer = 0.0         # segundo (desde el inicio) donde empieza el buffer
    contexto = ""
    fin = False

    with open(salida, "a", encoding="utf-8") as f:
        while not fin:
            # Juntar todo el audio que haya llegado
            try:
                datos = cola.get(timeout=0.5)
            except queue.Empty:
                datos = b""
            trozos = []
            while datos is not None:
                if datos:
                    trozos.append(datos)
                try:
                    datos = cola.get_nowait()
                except queue.Empty:
                    break
            if datos is None:
                fin = True
            if trozos:
                nuevo = np.frombuffer(b"".join(trozos), dtype=np.int16)
                buffer = np.concatenate([buffer, nuevo.astype(np.float32) / 32768.0])

            corte = len(buffer) if (fin and len(buffer)) else \
                buscar_corte(buffer, args.min_seg, args.max_seg, args.umbral)
            if corte is None:
                continue

            fragmento, buffer = buffer[:corte], buffer[corte:]
            t0 = inicio_buffer
            inicio_buffer += len(fragmento) / RATE

            # Saltar fragmentos que son puro silencio
            if np.sqrt(np.mean(fragmento ** 2)) < args.umbral:
                continue

            texto = transcribir(modelo, fragmento, args.idioma, contexto, args.beam)
            if not texto:
                continue
            contexto = (contexto + " " + texto)[-200:]
            linea = f"[{fmt_tiempo(t0)}] {texto}" if args.marcas else texto
            print(linea, flush=True)
            f.write(linea + "\n")
            f.flush()

    print(f"\nDetenido. Texto guardado en: {salida}")


def main():
    p = argparse.ArgumentParser(description="Transcribe en vivo el audio del sistema (inglés).")
    p.add_argument("--modelo", default="base.en",
                   help="tiny.en | base.en (por defecto) | small.en | distil-small.en")
    p.add_argument("--idioma", default="en", help="Idioma del audio (por defecto: en)")
    p.add_argument("--salida", help="Ruta del .txt (por defecto: ~/Transcripciones/...)")
    p.add_argument("--fuente", default="@DEFAULT_MONITOR@",
                   help="Fuente de captura (ver --listar-fuentes)")
    p.add_argument("--marcas", action="store_true", help="Agregar marcas de tiempo [mm:ss]")
    p.add_argument("--archivo", help="Transcribir un archivo de audio/video en vez de escuchar")
    p.add_argument("--listar-fuentes", action="store_true", help="Mostrar fuentes de audio")
    p.add_argument("--probar-con", metavar="ARCHIVO",
                   help="Prueba el modo en vivo reproduciendo un archivo en tiempo real")
    p.add_argument("--hilos", type=int, default=os.cpu_count() or 4, help="Hilos de CPU")
    p.add_argument("--beam", type=int, default=3, help="Precisión de búsqueda (1 = más rápido)")
    p.add_argument("--min-seg", type=float, default=6.0, help="Duración mínima de cada fragmento")
    p.add_argument("--max-seg", type=float, default=15.0, help="Duración máxima de cada fragmento")
    p.add_argument("--umbral", type=float, default=0.008, help="Volumen considerado silencio")
    args = p.parse_args()

    if args.listar_fuentes:
        listar_fuentes()
    elif args.archivo:
        modo_archivo(args)
    else:
        modo_vivo(args)


if __name__ == "__main__":
    main()
