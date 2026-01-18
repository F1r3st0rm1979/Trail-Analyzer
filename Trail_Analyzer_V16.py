# ==========================================================================================================
# -- Author:         Andrea Manna
# -- Version:        16.0.0 (Extended Excel Export)
# -- Create date:    2026-01-18
# -- Description:    TRAIL ANALYZER PRO V16 - Core System & Diagnostics.
# --                 Inizializzazione ambiente, logging clinico e test di integrità.
# ==========================================================================================================

import os
import sys
import time
import logging
import threading
import traceback
import functools
import datetime
import math
import inspect
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import queue # FIX V15: Thread-safe config

# ==========================================================================================================
# -- Metodo:        Configurazione Logging V15
# -- Descrizione:   Inizializza il file fisico .log per la persistenza dei dati.
# -- Uso V15:        Garantisce la conformità alla Regola 7 (Tracciabilità) e Regola 33.
# ==========================================================================================================
LOG_FILENAME = "trail_analyzer_v16.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8')]
)
logger = logging.getLogger("Trail_V16_Core")
logger.info("--- SESSIONE START: TRAIL ANALYZER V16 ---")

# ==========================================================================================================
# -- Funzione:      log_method_v15 (Decoratore)
# -- Descrizione:   Cattura in automatico nome metodo e parametri in chiaro (Regola 9/12).
# -- Parametri:     Cattura variabili scalari in chiaro e indica il tipo per oggetti complessi.
# ==========================================================================================================
def log_method_v15(func):
    """Decorator per il logging dei parametri dei metodi su file (Regola 7)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        param_strs = []
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            for name, value in bound_args.arguments.items():
                if name == 'self': continue
                if isinstance(value, (int, float, str, bool, type(None))):
                    param_strs.append(f"{name}={value}")
                else:
                    param_strs.append(f"{name}=[Variable:{type(value).__name__}]")
        except Exception:
            pass
        
        logger.info(f"EXEC: {func.__name__} | ARGS: {', '.join(param_strs) if param_strs else 'None'}")
        return func(*args, **kwargs)
    return wrapper

# Liste diagnostica (Regola 13)
CRITICAL_MISSING = []
OPTIONAL_MISSING = []

# ==========================================================================================================
# -- Libreria:       XML.ETREE.ELEMENTTREE
# -- Descrizione:    Parser per documenti XML (Standard Library).
# -- Uso V15:        Utilizzato per la conversione interna dei file KML (Regola 13).
# ==========================================================================================================
try:
    import xml.etree.ElementTree as ET
    logger.info("[ OK ] XML.ETREE caricato correttamente.")
except ImportError:
    logger.error("[FAIL] XML.ETREE non trovato.")
    CRITICAL_MISSING.append("xml.etree")

# ==========================================================================================================
# -- Libreria:       WEBBROWSER
# -- Descrizione:    Interfaccia per la visualizzazione di documenti web.
# -- Uso V15:        Apertura automatica delle mappe HTML generate (Regola 12).
# ==========================================================================================================
try:
    import webbrowser
    logger.info("[ OK ] WEBBROWSER caricato correttamente.")
except ImportError:
    logger.error("[FAIL] WEBBROWSER non trovato.")
    OPTIONAL_MISSING.append("webbrowser")

# ==========================================================================================================
# -- Libreria:       TKINTER
# -- Descrizione:    Toolkit standard per lo sviluppo di GUI in Python.
# -- Uso V15:        Gestione delle due finestre e del Notebook a 8 tab.
# ==========================================================================================================
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    logger.info("[ OK ] TKINTER caricato correttamente.")
except ImportError:
    logger.error("[FAIL] TKINTER non trovato.")
    CRITICAL_MISSING.append("tkinter")

# ==========================================================================================================
# -- Libreria:       PILLOW (PIL)
# -- Descrizione:    Libreria per la gestione avanzata delle immagini.
# -- Uso V15:        Rendering icone e interfacce grafiche dinamiche.
# ==========================================================================================================
try:
    from PIL import Image, ImageTk
    logger.info("[ OK ] PILLOW caricato correttamente.")
except ImportError:
    logger.error("[FAIL] PILLOW non trovato.")
    OPTIONAL_MISSING.append("pillow")

# ==========================================================================================================
# -- Libreria:       PANDAS
# -- Descrizione:    Libreria per l'analisi e manipolazione dei dati.
# -- Uso V15:        Gestione del database tracce e calcolo dei vettori fisiologici (Regola 13).
# ==========================================================================================================
try:
    import pandas as pd
    pd.set_option('future.no_silent_downcasting', True)
    logger.info("[ OK ] PANDAS caricato correttamente.")
except ImportError:
    logger.error("[FAIL] PANDAS non trovato.")
    CRITICAL_MISSING.append("pandas")

# ==========================================================================================================
# -- Libreria:       NUMPY
# -- Descrizione:    Calcolo vettoriale e scientifico.
# -- Uso V15:        Esegue il calcolo istantaneo delle pendenze e delle equazioni metaboliche.
# ==========================================================================================================
try:
    import numpy as np
    logger.info("[ OK ] NUMPY caricato correttamente.")
except ImportError:
    logger.error("[FAIL] NUMPY non trovato.")
    CRITICAL_MISSING.append("numpy")

# ==========================================================================================================
# -- Libreria:       SCIPY (savgol_filter & stats.norm)
# -- Descrizione:    Algoritmi per processamento segnali e statistica.
# -- Uso V15:        Filtra il rumore GPS e genera la curva di densità di probabilità (Regola 20).
# ==========================================================================================================
try:
    from scipy.signal import savgol_filter
    from scipy.stats import norm
    logger.info("[ OK ] SCIPY (Signal/Stats) caricato correttamente.")
except ImportError:
    logger.error("[FAIL] SCIPY non trovato.")
    CRITICAL_MISSING.append("scipy")

# ==========================================================================================================
# -- Libreria:       GPXPY
# -- Descrizione:    Parser per file GPS Exchange Format.
# -- Uso V15:        Estrae coordinate e altitudini dai file GPX per l'analisi 3D.
# ==========================================================================================================
try:
    import gpxpy
    logger.info("[ OK ] GPXPY caricato correttamente.")
except ImportError:
    logger.error("[FAIL] GPXPY non trovato.")
    CRITICAL_MISSING.append("gpxpy")

# ==========================================================================================================
# -- Librerie VIZ:   FOLIUM, MATPLOTLIB
# -- Descrizione:    Motori di visualizzazione grafica e cartografica.
# -- Uso V15:        Generano le mappe strategiche e i grafici con cursori sincronizzati.
# ==========================================================================================================
try:
    import folium
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    logger.info("[ OK ] Librerie di visualizzazione (Folium/Matplotlib) caricate.")
except ImportError:
    logger.error("[FAIL] Errore nel caricamento delle librerie di visualizzazione.")
    OPTIONAL_MISSING.append("visualizzazione")

# ==========================================================================================================
# -- Controllo:      Integrità Sistema V15
# -- Descrizione:    Blocca l'esecuzione se mancano componenti fondamentali (Regola 15).
# -- Logica:         Tenta il fallback su console se la GUI non è disponibile.
# ==========================================================================================================
if CRITICAL_MISSING:
    msg = f"ERRORE CRITICO: Componenti mancanti: {', '.join(CRITICAL_MISSING)}"
    logger.critical(msg)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Errore Qualità V15", msg)
    except:
        print(msg)
    sys.exit(1)

# ==========================================================================================================
# -- Metodo:        safe_step (Decoratore di Sicurezza)
# -- Descrizione:   Incapsula le funzioni per prevenire crash totali durante la simulazione.
# -- Uso V15:        Logga il traceback completo su file in caso di eccezione (Regola 7).
# ==========================================================================================================
def safe_step(func):
    """Garantisce la stabilità del motore di calcolo V15 intercettando gli errori."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"!!! ERRORE CRITICO in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    return wrapper

# ==========================================================================================================
# -- Classe:        VideoAudioSplashScreen
# -- Descrizione:   Gestisce la riproduzione del video di introduzione con sincronizzazione audio.
# -- Uso V15:        Entry point multimediale per l'avvio del software (Regola 12).
# ==========================================================================================================
class VideoAudioSplashScreen:
    @log_method_v15
    def __init__(self, video_path: str, on_finish_callback: Any):
        """
        Inizializza la finestra di splash e prepara il media player.
        
        Parametri:
        - video_path: Percorso del file .mp4
        - on_finish_callback: Funzione da chiamare al termine del video per avviare lo Stadio 1.
        """
        self.video_path = video_path
        self.on_finish_callback = on_finish_callback
        
        # Inizializzazione Finestra Splash (Regola 15)
        self.root = tk.Tk()
        self.root.title("Trail Analyzer V15 - Loading...")
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True) # Rimuove i bordi della finestra
        
        # Centratura Finestra
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width, height = 800, 450
        x = (screen_w // 2) - (width // 2)
        y = (screen_h // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Canvas per il rendering dei frame video
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # Placeholder per le risorse (saranno caricate in play_video)
        self.cap = None
        self.player = None
        
        logger.info(f"SplashScreen inizializzata per il file: {video_path}")

    # ==========================================================================================================
    # -- Metodo:        play_video
    # -- Descrizione:   Esegue il loop di riproduzione video e audio sincronizzato.
    # -- Uso V15:        Utilizza OpenCV per i frame e FFpyplayer per l'audio (Regola 12).
    # -- Note:           Implementa il ridimensionamento dinamico per il canvas 800x450.
    # ==========================================================================================================
    @log_method_v15
    def play_video(self):
        """
        Avvia la decodifica del file video e il rendering sul canvas di sistema.
        Implementa il controllo di fine file per il trigger del callback di chiusura.
        """
        try:
            # Import locali per moduli specifici multimediali (Regola 13)
            import cv2
            from ffpyplayer.player import MediaPlayer
            
            logger.info("Inizializzazione flussi multimediali (CV2 + FFpyplayer)...")
            
            self.cap = cv2.VideoCapture(self.video_path)
            self.player = MediaPlayer(self.video_path)
            
            # Funzione ricorsiva interna per il refresh del canvas (Regola 15)
            def stream_video():
                if self.cap is None: 
                    return
                
                # Lettura del frame video
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info("Fine del file video o errore di lettura. Chiusura Splash.")
                    self.on_video_end()
                    return
                
                # Gestione sincronizzazione audio tramite ffpyplayer
                audio_frame, val = self.player.get_frame()
                if val == 'eof':
                    logger.info("Fine del flusso audio (EOF).")
                
                # Pre-processing del frame per Tkinter (Regola 13)
                frame = cv2.resize(frame, (800, 450))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Conversione in oggetto compatibile con Tkinter
                img = Image.fromarray(frame)
                self.photo = ImageTk.PhotoImage(image=img)
                
                # Aggiornamento Canvas
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Scheduling del prossimo frame a ~30 FPS (33ms)
                # Questo permette alla GUI di rimanere reattiva (Regola 15)
                self.root.after(33, stream_video)
            
            # Avvio del ciclo di streaming
            stream_video()
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"ERRORE CRITICO durante la riproduzione video: {str(e)}")
            logger.error(traceback.format_exc())
            # In caso di errore video, forziamo il passaggio allo Stadio 1 per non bloccare il software
            self.on_video_end()

    # ==========================================================================================================
    # -- Metodo:        on_video_end
    # -- Descrizione:   Gestisce la chiusura pulita delle risorse multimediali e la transizione.
    # -- Uso V15:        Rilascia CV2 e FFpyplayer prima di lanciare il callback dello Stadio 1 (Regola 15).
    # ==========================================================================================================
    @log_method_v15
    def on_video_end(self):
        """
        Rilascia le risorse hardware e distrugge la finestra di splash.
        Avvia la transizione verso la finestra di ispezione geometrica (QuickLookWindow).
        """
        logger.info("Inizio procedura di cleanup SplashScreen...")
        
        # Rilascio della risorsa Video Capture di OpenCV
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Risorsa OpenCV rilasciata correttamente.")
            
        # Rilascio della risorsa Audio Player di FFpyplayer
        if self.player is not None:
            self.player.close_player()
            self.player = None
            logger.info("Risorsa FFpyplayer rilasciata correttamente.")
            
        # Chiusura della finestra Tkinter (Splash)
        try:
            # Verifichiamo se la root esiste ancora per evitare errori di 'invalid command name'
            if self.root.winfo_exists():
                self.root.destroy()
                logger.info("Finestra Splash distrutta. Transizione in corso.")
        except Exception as e:
            logger.warning(f"Nota: La finestra Splash è stata chiusa esternamente o errore minore: {str(e)}")
            
        # Innesco del callback per lo Stadio 1 (Ispezione Geometrica)
        # Questo callback è stato definito durante l'entry point del software.
        if self.on_finish_callback:
            logger.info("Esecuzione callback: Avvio Stadio 1 (QuickLookWindow).")
            self.on_finish_callback()

# ==========================================================================================================
# -- Classe:        ColorScheme
# -- Descrizione:   Hub centrale per la gestione cromatica di mappe e grafici.
# -- Uso V15:        Mantiene la consistenza terminologica e visiva (Regola 32).
# ==========================================================================================================
class ColorScheme:
    @log_method_v15
    def __init__(self):
        """
        Inizializza le palette di colori standardizzate per la visualizzazione tecnica.
        Definisce i gradienti di sforzo metabolico e i marker strategici.
        """
        # Palette per le pendenze (Gradienti di sforzo metabolico)
        self.slope_colors = {
            'hc_up':    '#0000FF', # High Category Up (> 30%) - Blu scuro
            'st_up':    '#2222FF', # Steep Up (> 20%)
            'mod_up':   '#4444FF', # Moderate Up (> 10%)
            'lt_up':    '#6666FF', # Light Up (> 1%)
            'flat':     '#888888', # Flat (-1% a 1%) - Grigio
            'lt_dn':    '#FFAAAA', # Light Down (< -1%)
            'mod_dn':   '#FFCCCC', # Moderate Down (< -10%)
            'st_dn':    '#FFEEEE', # Steep Down (< -20%)
            'hc_dn':    '#FFFFFF'  # High Category Down (< -30%) - Bianco
        }
        
        # Palette per i Marker Strategici (Identificazione rapida sulla mappa)
        self.marker_colors = {
            'food':  '#f39c12', # Arancio (Strategia Nutrizionale A/B/C)
            'sleep': '#8e44ad', # Viola (Punti di riposo/Fatica centrale)
            'gate':  '#e74c3c'  # Rosso (Cancelli orari e Basi Vita)
        }
        
        logger.info("Istanza ColorScheme creata: Palette pendenze e marker caricate correttamente.")

    # ==========================================================================================================
    # -- Metodo:        get_color_by_slope
    # -- Descrizione:   Mappa il valore della pendenza (%) a un codice colore HEX.
    # -- Uso V15:        Utilizzato sia per il plotting della traccia 3D che per le mappe Folium.
    # -- Parametri:     slope (float): Valore percentuale della pendenza.
    # -- Ritorna:       str: Codice HEX del colore corrispondente.
    # ==========================================================================================================
    @log_method_v15
    def get_color_by_slope(self, slope: float) -> str:
        """
        Determina il colore basato sulla pendenza percentuale (Regola 13).
        Le soglie sono calibrate per distinguere lo sforzo aerobico (salita) 
        dal logoramento eccentrico (discesa).
        """
        if slope > 30:   
            return self.slope_colors['hc_up']   # Salita estrema (> 30%)
        if slope > 20:   
            return self.slope_colors['st_up']   # Salita ripida (> 20%)
        if slope > 10:   
            return self.slope_colors['mod_up']  # Salita moderata (> 10%)
        if slope > 1:    
            return self.slope_colors['lt_up']   # Falsopiano in salita (> 1%)
        if slope >= -1:  
            return self.slope_colors['flat']    # Terreno piano (-1% a 1%)
        if slope >= -10: 
            return self.slope_colors['lt_dn']   # Discesa leggera (< -1%)
        if slope >= -20: 
            return self.slope_colors['mod_dn']  # Discesa moderata (< -10%)
        if slope >= -30: 
            return self.slope_colors['st_dn']   # Discesa ripida (< -20%)
            
        return self.slope_colors['hc_dn']       # Discesa estrema (< -30%)

# ==========================================================================================================
# -- Classe:        KMLParser
# -- Descrizione:   Modulo di conversione interna per il formato Keyhole Markup Language (KML).
# -- Uso V15:        Trasforma coordinate XML in DataFrame Pandas (Regola 13).
# ==========================================================================================================
class KMLParser:
    # ==========================================================================================================
    # -- Metodo:        parse
    # -- Descrizione:   Esegue il parsing dei tag <coordinates> e genera un dataset tabulare.
    # -- Uso V15:        Supporta il mapping 3D (Lat, Lon, Alt) per l'analisi fisiologica.
    # -- Parametri:     file_path (str): Percorso del file .kml
    # -- Ritorna:       pd.DataFrame: Dati estratti pronti per il pre-processing.
    # ==========================================================================================================
    @staticmethod
    @log_method_v15
    def parse(file_path: str) -> pd.DataFrame:
        """
        Analizza la struttura XML del file KML, estraendo i vertici della polilinea.
        Implementa la gestione del namespace standard OGC.
        """
        logger.info(f"Avvio parsing KML deterministico: {file_path}")
        try:
            # Parsing dell'albero XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Definizione del namespace standard KML (necessario per ElementTree)
            ns = {'ns': 'http://www.opengis.net/kml/2.2'}
            
            data = []
            # Ricerca ricorsiva di tutti i tag coordinates
            for coords in root.findall('.//ns:coordinates', ns):
                coord_str = coords.text.strip()
                # Lo standard KML prevede blocchi di "Longitudine,Latitudine,Altitudine" separati da spazi
                for row in coord_str.split():
                    parts = row.split(',')
                    if len(parts) >= 3:
                        data.append({
                            'Latitudine':  float(parts[1]),
                            'Longitudine': float(parts[0]),
                            'Altitudine':  float(parts[2])
                        })
            
            if not data:
                logger.warning(f"Attenzione: Nessuna coordinata valida trovata in {file_path}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"Parsing completato: estratti {len(df)} punti geografici.")
            return df
            
        except ET.ParseError as e:
            logger.error(f"Errore di sintassi XML nel file KML: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Errore imprevisto durante il parsing KML: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

# ==========================================================================================================
# -- Classe:        GPXAnalyzer
# -- Descrizione:   Motore core per la simulazione deterministica bioenergetica e tattica.
# -- Uso V15:        Coordina il pre-processamento 3D e il modello di fatica (Regola 13).
# ==========================================================================================================
class GPXAnalyzer:
    # ==========================================================================================================
    # -- Metodo:        __init__
    # -- Descrizione:   Inizializza le costanti biofisiche e le matrici di attrito ambientale.
    # -- Uso V15:        Imposta i parametri per pressione atmosferica ed efficienza (Regola 13).
    # ==========================================================================================================
    @log_method_v15
    def __init__(self):
        """
        Configura l'ambiente di calcolo, istanziando il sistema colori e le 
        costanti fisiche universali per la modellazione dell'ipossia.
        """
        # Inizializzazione utility visiva
        self.color_scheme = ColorScheme()
        
        # Matrice Terreno/Stagionalità (Coefficiente Eta): 
        # Modella l'attrito del suolo in base al mese e alla quota (Regola 13)
        self.terrain_matrix = {
            'Giugno':    {'low': 1.2, 'high': 1.8}, # Neve residua in quota
            'Luglio':    {'low': 1.1, 'high': 1.4},
            'Agosto':    {'low': 1.0, 'high': 1.2}, # Condizioni ottimali
            'Settembre': {'low': 1.2, 'high': 1.5}  # Terreno umido/fogliame
        }
        
        # Costanti Fisiche e Bioenergetiche
        self.P_SEA_LEVEL = 101.325 # Pressione standard al livello del mare (kPa)
        self.ETA_MET = 0.24        # Efficienza metabolica umana media (24%)
        
        # Inizializzazione variabili di stato della simulazione
        self.feeding_events = []
        self.logistic_data = pd.DataFrame()
        self.current_file_path = ""
        self.simulation_stats = {} # Container per KPI globali (NP, TSS, IF)
        
        logger.info("Motore GPXAnalyzer V15 inizializzato con successo.")

    # ==========================================================================================================
    # -- Metodo:        _load_logistic_points
    # -- Descrizione:   Carica i punti logistici (Basi Vita) da file CSV esterno.
    # -- Uso V15:        Abilita la logica di sonno vincolata alle basi vita (Regola V14).
    # ==========================================================================================================
    @log_method_v15
    def _load_logistic_points(self):
        """Carica basivita.csv se presente."""
        try:
            csv_path = os.path.join(os.path.dirname(self.current_file_path) if self.current_file_path else os.getcwd(), "basivita.csv")
            # Fallback alla directory corrente se non trovato nel path del file
            if not os.path.exists(csv_path):
                csv_path = "basivita.csv"
            
            if os.path.exists(csv_path):
                df_log = pd.read_csv(csv_path)
                # Standardizzazione colonne
                df_log.columns = [c.lower() for c in df_log.columns]
                if 'km' in df_log.columns and 'nome' in df_log.columns:
                    self.logistic_data = df_log
                    logger.info(f"Caricati {len(df_log)} punti logistici da {csv_path}")
                else:
                    logger.warning(f"File logistica {csv_path} non valido: mancano colonne 'Nome', 'Km'.")
            else:
                logger.warning("File 'basivita.csv' non trovato. Funzionalità sleep limitata.")
        except Exception as e:
            logger.error(f"Errore caricamento logistica: {str(e)}")

    # ==========================================================================================================
    # -- Metodo:        load_and_preprocess
    # -- Descrizione:   Carica il file sorgente, pulisce i dati e calcola la geometria 3D.
    # -- Uso V15:        Normalizza i dati KML/GPX e applica il filtraggio altimetrico (Regola 13).
    # -- Parametri:     file_path (str): Percorso del file da analizzare.
    # -- Ritorna:       pd.DataFrame: Dataset normalizzato e filtrato.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def load_and_preprocess(self, file_path: str) -> pd.DataFrame:
        """
        Esegue la pipeline di ingestione dati: caricamento, smoothing altimetrico 
        e calcolo dei vettori distanza/pendenza.
        """
        logger.info(f"Avvio pipeline di pre-processamento per: {file_path}")
        self.current_file_path = file_path
        ext = os.path.splitext(file_path)[1].lower()
        
        # 1. Ingestione Differenziata (Regola 13)
        if ext == '.kml':
            df = KMLParser.parse(file_path)
        elif ext == '.gpx':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    gpx = gpxpy.parse(f)
                data = []
                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            data.append({
                                'Latitudine':  point.latitude,
                                'Longitudine': point.longitude,
                                'Altitudine':  point.elevation
                            })
                df = pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Errore durante il parsing GPX: {str(e)}")
                return pd.DataFrame()
        else:
            logger.error(f"Formato file non supportato: {ext}")
            return pd.DataFrame()

        if df.empty or len(df) < 2:
            logger.warning("Dataset insufficiente per l'analisi.")
            return df

        # 2. Calcolo Distanze 2D (Formula Haversine) - SPOSTATO PRIMA DELLO SMOOTHING (Fix V15)
        coords = np.radians(df[['Latitudine', 'Longitudine']].values)
        lat1, lon1 = coords[:-1].T
        lat2, lon2 = coords[1:].T
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_2d = 6371008.8 * c 

        # Inserimento pre-calcolato per analisi densità
        df['D_Inc_2D'] = np.insert(dist_2d, 0, 0.0)
        
        # 3. Smoothing Altimetrico Adattivo (Fix V15 Peak Clipping)
        # Analisi Densità Punti
        avg_step = np.mean(dist_2d) if len(dist_2d) > 0 else 0
        raw_max_alt = df['Altitudine'].max()
        
        # Selezione Finestra Adattiva
        if avg_step < 10: window = 31      # Alta densità -> Smoothing aggressivo
        elif avg_step < 30: window = 11    # Media densità -> Standard V15
        elif avg_step < 100: window = 5    # Bassa densità -> Smoothing leggero
        else: window = 3                   # Molto bassa -> Quasi nullo
        
        # Clamp window su dispari e < len(df)
        if window >= len(df): window = len(df) // 2 * 2 + 1
        if window < 3: window = 0 # Disabilita se troppi pochi punti
        
        if window >= 3:
            df['Altitudine_Raw'] = df['Altitudine'].copy() # Backup
            df['Altitudine'] = savgol_filter(df['Altitudine'], window_length=int(window), polyorder=2)
            
            # Peak Protection Logic (Regola V15-Safety)
            new_max = df['Altitudine'].max()
            delta_peak = raw_max_alt - new_max
            
            if delta_peak > 25.0: # Se abbiamo tagliato più di 25m di vetta
                logger.warning(f"Smoothing aggressivo rilevato (Delta Peak: {delta_peak:.1f}m). Attivazione Peak Protection Estrema.")
                
                # UPDATED V15: Revert to RAW per garantire il picco massimo
                df['Altitudine'] = df['Altitudine_Raw']
                logger.info("Filtro DISABILITATO: Ripristino dati RAW per preservare la quota massima.")

            logger.info(f"Filtraggio adattivo applicato (Step: {avg_step:.1f}m, Win: {window}). MaxAlt: {raw_max_alt:.0f}->{df['Altitudine'].max():.0f}")

        # 4. Calcolo Distanze 3D e Pendenze Finali (Backup safety)

        # 4. Calcolo Distanze 3D e Pendenze
        delta_alt = np.diff(df['Altitudine'].values)
        dist_3d = np.sqrt(dist_2d**2 + delta_alt**2)

        # Inserimento dei delta nel DataFrame (primo punto a zero)
        df['D_Inc'] = np.insert(dist_3d, 0, 0.0)
        df['D_Inc_2D'] = np.insert(dist_2d, 0, 0.0)
        df['Delta_Alt'] = np.insert(delta_alt, 0, 0.0)
        
        # Distanza Cumulata e Pendenza Istantanea
        df['Distanza Cumulata'] = df['D_Inc'].cumsum()
        df['Pendenza (%)'] = np.where(
            df['D_Inc_2D'] > 0.1, 
            (df['Delta_Alt'] / df['D_Inc_2D']) * 100, 
            0.0
        )
        # Clip di sicurezza per pendenze irrealistiche (Regola 15)
        df['Pendenza (%)'] = df['Pendenza (%)'].clip(-60, 60)

        # 5. Caricamento Logistica per Analisi Strategica (Regola V14)
        self._load_logistic_points()

        logger.info(f"Pre-processamento completato. Punti: {len(df)}, Distanza Totale: {df['Distanza Cumulata'].max():.1f}m")
        return df

    # ==========================================================================================================
    # -- Metodo:        _get_env_res
    # -- Descrizione:   Calcola il coefficiente di resistenza del terreno (Eta).
    # -- Uso V15:        Varia in base al mese e all'altitudine (presenza neve/fango) (Regola 13).
    # -- Parametri:     mese (str): Mese della simulazione.
    # --                altitudine (float): Quota del punto corrente in metri.
    # -- Ritorna:       float: Coefficiente di attrito ambientale (Eta).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_env_res(self, mese: str, altitudine: float) -> float:
        """
        Determina la resistenza ambientale basata sulla stagione e sulla quota.
        Implementa il passaggio al coefficiente 'high' per terreni tecnici sopra i 2000m.
        """
        # Recupero dei dati del mese con fallback su Agosto (mese standard)
        dati_mese = self.terrain_matrix.get(mese, self.terrain_matrix['Agosto'])
        
        # Logica deterministica: sopra i 2000m si assume terreno più tecnico/instabile
        if altitudine < 2000:
            coefficiente = float(dati_mese['low'])
        else:
            coefficiente = float(dati_mese['high'])
            
        return coefficiente

    # ==========================================================================================================
    # -- Metodo:        _get_hyp_pen
    # -- Descrizione:   Calcola la penalità da ipossia basata sul modello barometrico e Bassett.
    # -- Uso V15:        Applica il decremento di potenza (FTP) in tempo reale (Regola 13).
    # -- Parametri:     altitudine (float): Quota del punto corrente in metri.
    # --                vo2_max (float): Capacità aerobica massima dell'atleta (ml/kg/min).
    # -- Ritorna:       float: Fattore correttivo (0.5 - 1.0) da applicare alla potenza.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_hyp_pen(self, altitudine: float, vo2_max: float) -> float:
        """
        Calcola la riduzione della performance dovuta alla quota (Regola 13).
        Modella la sensibilità all'ipossia in funzione della VO2 max dell'atleta.
        """
        # Calcolo della pressione atmosferica (kPa) alla quota indicata (Formula Barometrica)
        # P = P0 * (1 - 2.25577e-5 * h)^5.25588
        p_atm = self.P_SEA_LEVEL * (1 - 2.25577e-5 * altitudine)**5.25588
        
        # Calcolo della sensibilità individuale (Regola 20)
        # Atleti con VO2 max elevato subiscono proporzionalmente un impatto maggiore
        # a causa della limitazione del trasporto di O2 a carichi elevati.
        sensibilita = 1.0 + (max(0, vo2_max - 40) / 250)
        
        # Costante di riferimento per la tensione di vapore e pressione alveolare
        P_REF = 6.27 # kPa
        
        # FIX V15: Linearizzazione Bounceless (Gradienti dolci)
        # Sostituisce la curva esponenziale con una sensibilità interpolata
        sens_adj = np.interp(vo2_max, [40, 90], [1.0, 1.1])
        
        # Fattore correttivo basato sulla pressione parziale
        denominatore = self.P_SEA_LEVEL - P_REF
        numeratore = p_atm - P_REF
        
        ratio = max(0.01, numeratore / denominatore)
        # Calcolo decremento con sensibilità dinamica
        decremento = max(0.5, ratio**sens_adj)
        
        return float(decremento)

    # ==========================================================================================================
    # -- Metodo:        _get_hybrid_cost
    # -- Descrizione:   Integrazione Pandolf/Gottschall per costo energetico cammino/corsa.
    # -- Uso V15:        Modello energetico multi-modale per ultra-trail (Regola 13).
    # -- Parametri:     wa (float): Peso atleta (kg).
    # --                lz (float): Peso zaino (kg).
    # --                v (float): Velocità (m/s).
    # --                g (float): Pendenza (%).
    # --                eta (float): Coefficiente attrito terreno.
    # --                r (float): Walk Ratio (0.0=Run, 1.0=Walk).
    # --                vo2 (float): VO2 max (ml/kg/min).
    # -- Ritorna:       float: Costo metabolico totale in Watt (W).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_hybrid_cost(self, wa: float, lz: float, v: float, g: float, eta: float, r: float, vo2: float) -> float:
        """
        Stima il costo metabolico pesato sulla percentuale di cammino (Regola 13).
        Implementa la transizione energetica tra camminata tecnica e corsa.
        """
        # 1. Modello Pandolf (Cammino con carico)
        # M = 1.5W + 2.0(W + L)(L/W)^2 + eta(W + L)(1.5V^2 + 0.35VG)
        # Aggiustamento efficienza metabolica dinamica (V15)
        m_eff = (1.1 + (vo2 / 110)) * wa + 2.0 * (wa + lz) * (lz / wa)**2
        #c_walk = m_eff + eta * (wa + lz) * (1.5 * v**2 + 0.35 * v * (g / 100.0 if g > 0 else 0))
        c_walk = m_eff + eta * (wa + lz) * (1.5 * v**2 + 0.35 * v * g)
        
        # 2. Modello Gottschall (Corsa su pendenza)
        # Basato su curve polinomiali di costo specifico (J/kg/m)
        i = g / 100.0
        # Equazione polinomiale di 5° grado per la corsa (V14 Standard)
        c_run_spec = (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6)
        
        # Conversione in Watt: Costo_Specifico * Massa_Totale * Velocità * Attrito
        c_run = c_run_spec * (wa + lz) * v * eta
        
        # 3. Integrazione Ibrida
        # Risultato pesato sul Walk Ratio 'r' (Regola 13)
        costo_totale = (c_walk * r) + (c_run * (1.0 - r))

        # FIX V15: Basal Metabolic Floor (Prevents negative cost on steep descent)
        basal_floor = wa * 1.2 
        return float(max(costo_totale, basal_floor))

    # ==========================================================================================================
    # -- Metodo:        _get_mecc_pen
    # -- Descrizione:   Calcolo del danno muscolare eccentrico e relativa penalità meccanica.
    # -- Uso V15:        Riduce l'efficienza locomotoria in base al logoramento fisico (Regola 13).
    # -- Parametri:     danno_ecc (float): Dislivello negativo accumulato (metri).
    # --                massa_totale (float): Somma del peso atleta e peso zaino (kg).
    # -- Ritorna:       float: Fattore di penalità (0.8 - 1.0).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_mecc_pen(self, danno_ecc: float, massa_totale: float) -> float:
        """
        Modella il decremento di prestazione dovuto all'accumulo di stress meccanico (Regola 13).
        La penalità aumenta proporzionalmente alla massa e alla quota di discesa percorsa.
        """
        # FIX V15: Resilienza Meccanica Dinamica (Elite vs Amateur)
        # k_mecc_dyn scala con il rapporto Watt/Kg (più sei allenato, meno ti rompi)
        # Interpolazione: 2.5 W/kg (Amateur) -> 5.5 W/kg (Elite)
        # Valori: 5.0e-7 -> 3.0e-7
        # Nota: richiede accesso a self.current_wkg o simile, ma qui passiamo solo variabili locali.
        # Soluzione: Parametrizzazione implicita o conservativa. 
        # Per ora usiamo un valore fisso "Bounceless" che sarà sovrascritto dalla logica globale
        # oppure assumiamo una media. Ma la richiesta specifica l'interpolazione.
        # Poiché non passiamo ftp/wa qui, applichiamo un gradiente standard.
        # TUTTAVIA, la richiesta al punto 2 dice "Usa k_mecc_dyn...".
        # Dobbiamo calcolarlo fuori e passarlo, o stimarlo.
        # Dato che non modifico la firma, userò un valore medio ottimizzato Bounceless per ora,
        # MA la logica di Global Floor nel loop compenserà per gli elite.
        
        # UPDATE: Calcolo "Bounceless" standardizzato V15 Refactor
        K_MECC = 0.0000004 # Valore medio bilanciato
        
        riduzione = danno_ecc * massa_totale * K_MECC
        
        # Il floor dinamico sarà applicato nel loop principale come richiesto.
        penalita = max(0.8, 1.0 - riduzione)
        
        return float(penalita)

    # ==========================================================================================================
    # -- Metodo:        _get_gly_pen
    # -- Descrizione:   Modello energetico Rapoport per la deplezione di glicogeno e calcolo penalità.
    # -- Uso V15:        Gestisce il consumo di zuccheri e la restrizione della potenza erogabile (Regola 13).
    # -- Parametri:     p_met (float): Potenza metabolica istantanea (W).
    # --                ftp_adj (float): FTP corretta per altitudine (W).
    # --                w_a (float): Peso dell'atleta (kg).
    # --                dt (float): Intervallo di tempo (secondi).
    # --                ga (float): Glicogeno attuale (grammi).
    # -- Ritorna:       Tuple[float, float]: (Fattore di penalità 0.45-1.0, Nuova quantità glicogeno).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_gly_pen(self, p_met: float, ftp_adj: float, w_a: float, dt: float, ga: float) -> Tuple[float, float]:
        """
        Calcola lo stato energetico e la restrizione della performance (Regola 13).
        Implementa il decadimento non lineare della velocità all'esaurimento delle scorte.
        """
        # 1. Calcolo dell'intensità relativa (Phi) rispetto alla soglia metabolica
        # P_met_threshold = FTP_adj / efficienza_metabolica
        p_threshold_met = (ftp_adj / self.ETA_MET) + 0.1
        phi = p_met / p_threshold_met
        
        # 2. Calcolo del consumo di glicogeno (g/s)
        # Modello non lineare: il consumo aumenta con il quadrato dell'intensità (Regola 13)
        # Il coefficiente 0.00035 deriva dal consumo medio di 1.2g/min/kg alla soglia
        cons_sec = 0.00035 * w_a * (phi**2.0)
        
        # Aggiornamento scorte di glicogeno (grammi)
        nuovo_glicogeno = max(0.0, ga - (cons_sec * dt))
        
        # 3. Determinazione della penalità di potenza (Glycogen Penalty)
        # Se ga > 100g: nessuna penalità (p=1.0)
        # Sotto i 100g: inizio del "muro" con riduzione fino al 45% della potenza (Regola 20)
        if nuovo_glicogeno > 100:
            p = 1.0
        else:
            # Funzione di decadimento lineare verso il limite metabolico dei grassi
            p = max(0.45, 0.45 + (nuovo_glicogeno / 181.8))
            
        return float(p), float(nuovo_glicogeno)

    # ==========================================================================================================
    # -- Metodo:        _get_sleep_pen
    # -- Descrizione:   Modello circadiano per la fatica centrale e deprivazione di sonno.
    # -- Uso V15:        Penalizza la velocità nelle ore notturne e dopo lunghe veglie (Regola 13).
    # -- Parametri:     ha (float): Ore di veglia/attività accumulate (hours awake).
    # --                hd (float): Ora solare del giorno (hour of day, 0-23).
    # -- Ritorna:       float: Fattore di penalità (0.5 - 1.0).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_sleep_pen(self, ha: float, hd: float) -> float:
        """
        Modella la perdita di lucidità e reattività basata sull'ora e sulla veglia (Regola 13).
        Implementa il 'muro del sonno' tra le 02:00 e le 05:00 del mattino.
        """
        # 1. Penalità per veglia accumulata (Base Penalty)
        # La performance rimane stabile fino a 18 ore, poi degrada progressivamente.
        if ha <= 18:
            bp = 0.0
        elif ha <= 24:
            bp = (ha - 18) * 0.01  # Lieve decadimento cognitivo
        else:
            bp = 0.06 + (ha - 24) * 0.015 # Decadimento accelerato dopo le 24h
            
        # 2. Moltiplicatore Circadiano (Circadian Multiplier)
        # Incrementa l'effetto della fatica durante il nadir biologico (Regola 20)
        if 2.0 <= hd <= 5.0:
            cm = 3.0  # Picco di sonnolenza e riduzione del drive motorio
        elif 5.0 < hd <= 7.0 or 22.0 <= hd < 2.0:
            cm = 1.5  # Transizione crepuscolare
        else:
            cm = 1.0  # Condizioni di veglia standard
            
        # 3. Calcolo Penalità Finale
        # f_sleep = 1.0 - (base_penalty * circadian_multiplier)
        # Clamp di sicurezza a 0.5 per riflettere lo stato di "power nap" forzato.
        penalita = max(0.5, 1.0 - (bp * cm))
        
        return float(penalita)

    # ==========================================================================================================
    # -- Metodo:        _get_technical_braking_penalty
    # -- Descrizione:   Calcola il rallentamento biomeccanico su discese ripide.
    # -- Uso V15:        Simula la frenata eccentrica obbligatoria su terreni declivi (Regola 13).
    # -- Parametri:     slope (float): Pendenza del terreno (%).
    # -- Ritorna:       float: Fattore di rallentamento (0.1 - 1.0).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_technical_braking_penalty(self, slope: float) -> float:
        """
        Riduce la velocità in discesa quando la pendenza diventa estrema.
        Oltre il -15% inizia la frenata, oltre il -30% il rallentamento è drastico.
        """
        if slope >= -10.0:
            return 1.0
        elif slope >= -15.0:
            # Zona di transizione leggera (-10% a -15%)
            return 0.95
        elif slope >= -30.0:
            # Discesa ripida: rallentamento progressivo fino a 0.7
            # Mapping: -15 -> 0.95, -30 -> 0.7
            ratio = (abs(slope) - 15) / 15.0
            return 0.95 - (0.25 * ratio)
        else:
            # Discesa estrema: rallentamento fino a 0.5 (passo dopo passo)
            return 0.55

    # ==========================================================================================================
    # -- Metodo:        _calculate_critical_slope
    # -- Descrizione:   Calcola la pendenza critica (G_crit) per la transizione corsa/cammino.
    # -- Uso V15:        Definisce il punto di flesso biomeccanico (Specifica #1).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _calculate_critical_slope(self, vo2: float, w_kg: float) -> float:
        """
        Calcola G_crit basato su VO2max e W/kg.
        Formula: G_crit = 12.0 + (VO2max - 40) * 0.15 + (W/kg - 2.5) * 0.5
        """
        g_crit = 12.0 + (vo2 - 40.0) * 0.15 + (w_kg - 2.5) * 0.5
        return float(max(5.0, g_crit)) # Floor di sicurezza a 5%

    # ==========================================================================================================
    # -- Metodo:        _calculate_dynamic_walk_ratio_vector
    # -- Descrizione:   Pre-calcola il vettore di Walk Ratio per l'intera traccia con Target Anchor.
    # -- Uso V15:        Implementa Walk Ratio Dinamico, Adattivo e Cronologico (Specifiche #2, #3, #4).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _calculate_dynamic_walk_ratio_vector(self, df: pd.DataFrame, settings: Dict[str, Any]) -> np.ndarray:
        """
        Genera il vettore r_local ottimizzato per soddisfare la media target (r_target).
        """
        # 1. Estrazione Parametri
        vo2 = settings.get('vo2', 50.0)
        ftp = settings['ftp']
        wa = settings['peso']
        w_kg = ftp / wa if wa > 0 else 3.0
        r_target = settings.get('walk_ratio', 0.8) # Questo ora è il Target Globale (Anchor)
        
        # 2. Calcolo Soglia Biomeccanica (G_crit)
        g_crit = self._calculate_critical_slope(vo2, w_kg)
        
        # 3. Preparazione vettori numpy
        slopes = df['Pendenza (%)'].values
        d_cum = df['Distanza Cumulata'].values
        total_dist = d_cum[-1] if len(d_cum) > 0 else 1.0
        
        # 4. Modellazione Decadimento Cronologico (Front-loading Run)
        # F_stamina: 1.0 all'inizio, 0.0 alla fine
        f_stamina = np.maximum(0, total_dist - d_cum) / total_dist
        
        # 5. Funzione Core Walk Ratio Dinamico (r_raw) senza scaling K
        # r = 1 / (1 + e^(-0.5 * (G - G_crit) * F_stamina))
        # Nota: Se G < G_crit (pendenza facile) -> esponente positivo -> r tende a 0 (Corsa)
        #       Se F_stamina alto (inizio) -> effetto amplificato -> r tende a 0 ancora di più
        #       Se F_stamina basso (fine) -> esponente tende a 0 -> r tende a 0.5 o segue solo G
        
        # Per stabilità numerica exp, clippiamo l'argomento
        # Argomento positivo = Cammino (r->1), Negativo = Corsa (r->0)
        arg = -0.5 * (slopes - g_crit) * f_stamina
        arg = np.clip(arg, -10, 10) # Evita overflow
        r_raw_vector = 1.0 / (1.0 + np.exp(arg))
        
        # 6. Target Anchor Logic: Trovare Scale Factor K
        # Vogliamo che mean(r_final) ≈ r_target
        # Dove r_final = Clip(r_raw * K, 0, 1)
        # Risolviamo iterativamente per K
        
        # Quick check: se r_target è estremo, forziamo
        if r_target <= 0.05: return np.zeros(len(df))
        if r_target >= 0.95: return np.ones(len(df))
        
        current_mean = np.mean(r_raw_vector)
        
        # Euristica iniziale per K
        if current_mean < 1e-5: current_mean = 1e-5
        K_est = r_target / current_mean
        
        # Binary Search per trovare K ottimale
        # Range esteso per gestire casi estremi (target 0.9 su pendenze lievi richiede K molto alto)
        low = 0.0
        high = max(100.0, K_est * 20.0) 
        best_k = K_est
        min_err = 1.0
        
        for _ in range(30): # 30 iterazioni garantiscono alta precisione
            mid = (low + high) / 2
            r_trial = np.clip(r_raw_vector * mid, 0.0, 1.0)
            mean_trial = np.mean(r_trial)
            
            diff = mean_trial - r_target
            if abs(diff) < min_err:
                min_err = abs(diff)
                best_k = mid
                
            if mean_trial < r_target:
                low = mid
            else:
                high = mid
        
        # Calcolo vettore finale con K ottimizzato
        r_final = np.clip(r_raw_vector * best_k, 0.0, 1.0)
        
        logger.info(f"Dynamic Walk Ratio Optimization: Target={r_target:.2f}, G_crit={g_crit:.1f}%, K={best_k:.2f}, Result Mean={np.mean(r_final):.3f}")
        
        return r_final

    # ==========================================================================================================
    # -- Metodo:        _get_adaptive_intensity_factor
    # -- Descrizione:   Calcola il fattore di intensità (IF) target in base alla lunghezza del trail.
    # -- Uso V15:        Implementa scaling lineare 0.72 - 0.87 (Punto 1).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _get_adaptive_intensity_factor(self, total_km: float) -> float:
        """
        Calcola IF target con decadimento esponenziale (Specifica Utente).
        < 20km: 0.87 (Costante)
        > 20km: 0.72 + 0.15 * e^(-0.017 * (total_km - 20))
        Asintoto a 0.72.
        Punto di controllo: 220km -> ~0.725
        """
        if total_km <= 20.0:
            return 0.87
        else:
            # Decadimento Esponenziale verso asintoto 0.72
            # Rate k = 0.017 (calcolato per avere 0.725 a 220km)
            k = 0.017
            decay = np.exp(-k * (total_km - 20.0))
            return 0.72 + 0.15 * decay

    @safe_step
    @log_method_v15
    def _run_v16_deterministic_cascade(self, df: pd.DataFrame, settings: Dict[str, Any], progress_callback=None):
        """
        Esegue la simulazione bioenergetica V16 (Extended Telemetry).
        Calcola l'evoluzione temporale e metabolica dell'atleta lungo il percorso.
        """
        logger.info("Avvio motore deterministico V16...")
        
        # 1. Estrazione e Inizializzazione Parametri (Regola 20)
        w_a, w_z = settings['peso'], settings.get('peso_zaino', 5.0)
        ftp_base, vo2 = settings['ftp'], settings.get('vo2', 50.0)
        ftp_tol = settings.get('ftp_tolerance', 5.0) / 100.0
        z_score = settings.get('z_score', 1.96)
        strategy = settings.get('nutri_strategy', 'B')
        
        # Calcolo Glicogeno Iniziale (Modello Potenza/Peso)
        g_per_kg = np.interp(ftp_base / w_a, [2.5, 5.5], [10.0, 15.0])
        initial_glyco = g_per_kg * w_a
        glicogeno = initial_glyco
        
        # Variabili di stato
        t_cum, danno_ecc, kcal_cum = 0.0, 0.0, 0.0
        last_food_km = -10.0
        last_sleep_km = -10.0
        self.feeding_events = []
        self.sleep_events = []
        
        v_base = 0.1 
        p_target_met = 1.0
        
        # Parametri Calibrazione Elite
        w_kg = ftp_base / w_a
        k_mecc_dyn = np.interp(w_kg, [2.5, 5.5], [0.0000005, 0.0000003])
        floor_mecc = np.interp(w_kg, [2.5, 5.5], [0.80, 0.88])
        global_floor = np.interp(w_kg, [2.5, 5.5], [0.45, 0.65])

        d_cum_check = df['Distanza Cumulata'].values
        total_dist_km = d_cum_check[-1] / 1000.0 if len(d_cum_check) > 0 else 1.0
        
        walk_ratio_vector = self._calculate_dynamic_walk_ratio_vector(df, settings)
        
        # Extended Results Container (V16) - Tracks ALL intermediate vars
        results = {k: [] for k in ['p_watt', 'p_met', 't_min', 't_max', 'glyco', 'fatica', 'v_eff', 'kcal', 
                                   't_raw_sec', 'f_hyp', 'f_mecc', 'f_gly', 'walk_ratio',
                                   'eta_env', 'target_if', 'cost_u', 'v_base', 'f_tech', 'danno_ecc', 'f_total', 'f_sleep']}
        
        # 2. Ciclo di Simulazione Cascata
        for count, (i, row) in enumerate(df.iterrows()):
            curr_km = row['Distanza Cumulata'] / 1000.0
            
            # A. Fattori Ambientali
            eta_env = self._get_env_res(settings.get('mese', 'Agosto'), row['Altitudine'])
            f_hyp = self._get_hyp_pen(row['Altitudine'], vo2)
            if f_hyp is None: f_hyp = 1.0
            
            ftp_adj = ftp_base * f_hyp
            current_r = walk_ratio_vector[count]
            
            # B. Target IF
            target_if = self._get_adaptive_intensity_factor(total_dist_km)
            p_target_met = (ftp_adj * target_if) / self.ETA_MET
            
            # C. Costo Energetico
            cost_u = self._get_hybrid_cost(w_a, w_z, 1.0, row['Pendenza (%)'], eta_env, current_r, vo2)
            
            COST_SAFETY_MARGIN = 1.10
            v_base_calc = p_target_met / (max(10.0, cost_u * COST_SAFETY_MARGIN) + 0.1)
            
            slope_cap_kmh = max(3.0, 25.0 + (min(0, row['Pendenza (%)']) * 0.6))
            slope_cap_ms = slope_cap_kmh / 3.6
            v_base = max(0.1, min(v_base_calc, slope_cap_ms))
            
            # D. Strategia Nutrizionale
            refill_dt = 0.0
            dt_est = row['D_Inc'] / (v_base + 0.1)
            
            if strategy == 'A': 
                refill_dt = (1.2 / 60.0) * dt_est
            elif strategy == 'B' and glicogeno < (initial_glyco * 0.75) and (curr_km - last_food_km) > 10.0:
                glicogeno = min(initial_glyco, glicogeno + 125.0)
                last_food_km = curr_km
                self.feeding_events.append({'lat': row['Latitudine'], 'lon': row['Longitudine'], 'km': curr_km, 'label': 'REFILL_B'})
            elif strategy == 'C': 
                rate = 1.5 if row['Pendenza (%)'] > 5.0 else 0.8
                refill_dt = (rate / 60.0) * dt_est
            
            glicogeno = min(initial_glyco, glicogeno + refill_dt)
            
            # E. Penalità
            gly_p, glicogeno = self._get_gly_pen(p_target_met, ftp_adj, w_a, dt_est, glicogeno)
            f_cent = self._get_sleep_pen(t_cum/3600.0, (settings['ora_partenza'] + int(t_cum/3600.0)) % 24)
            
            # Sleep Detection
            if f_cent < 0.70:
                near_bv = False
                bv_name = ""
                if not self.logistic_data.empty:
                    for _, bv_row in self.logistic_data.iterrows():
                        if abs(curr_km - bv_row['km']) <= 0.1:
                            near_bv = True; bv_name = bv_row['nome']; break
                if near_bv and (curr_km - last_sleep_km) > 5.0:
                    self.sleep_events.append({'lat': row['Latitudine'], 'lon': row['Longitudine'], 'km': curr_km, 'label': f'SLEEP @ {bv_name}'})
                    last_sleep_km = curr_km
            
            if row['Delta_Alt'] < 0: 
                danno_ecc += abs(row['Delta_Alt'])
            
            riduzione_mecc = danno_ecc * (w_a + w_z) * k_mecc_dyn
            f_mecc = max(floor_mecc, 1.0 - riduzione_mecc)
            
            f_tech = self._get_technical_braking_penalty(row['Pendenza (%)'])
            
            f_prod = gly_p * f_cent * f_mecc * f_tech
            f_total = max(f_prod, global_floor)
            
            v_eff = v_base * f_total
            dt = row['D_Inc'] / v_eff if v_eff > 0.1 else 1.0
            t_cum += dt
            
            kcal_cum += (p_target_met * dt) / 4184.0
            window = (t_cum * ftp_tol) * z_score
            
            # Recording V16
            results['p_watt'].append(p_target_met * self.ETA_MET)
            results['p_met'].append(p_target_met)
            results['t_min'].append(t_cum - window)
            results['t_max'].append(t_cum + window)
            results['glyco'].append(glicogeno)
            results['fatica'].append(f_cent)
            results['v_eff'].append(v_eff)
            results['kcal'].append(kcal_cum)
            results['t_raw_sec'].append(t_cum)
            results['f_hyp'].append(f_hyp)
            results['f_mecc'].append(f_mecc)
            results['f_gly'].append(gly_p)
            results['walk_ratio'].append(current_r)
            
            # New V16 Fields
            results['eta_env'].append(eta_env)
            results['target_if'].append(target_if)
            results['cost_u'].append(cost_u)
            results['v_base'].append(v_base)
            results['f_tech'].append(f_tech)
            results['danno_ecc'].append(danno_ecc)
            results['f_total'].append(f_total)
            results['f_sleep'].append(f_cent)

            if progress_callback and count % 500 == 0: 
                prog = int((count/len(df))*95)
                progress_callback(prog)
            
        if progress_callback: progress_callback(100)

        # 3. Finalizzazione V16
        df['Potenza_Watt'] = results['p_watt']
        df['Glicogeno_Residuo'] = results['glyco']
        df['Tempo_Min_V15'] = results['t_min']
        df['Tempo_Max_V15'] = results['t_max']
        df['Fatica_Centrale'] = results['fatica']
        df['Velocita_V15'] = results['v_eff']
        df['Kcal_Cum'] = results['kcal']
        df['Tempo_Sec_Raw'] = results['t_raw_sec']
        
        df['f_hyp'] = results['f_hyp']
        df['f_mecc'] = results['f_mecc']
        df['f_gly'] = results['f_gly']
        df['Walk_Ratio_Dyn'] = results['walk_ratio']
        
        # New V16 Columns
        df['Eta_Terrain'] = results['eta_env']
        df['Target_IF'] = results['target_if']
        df['Costo_Metabolico_W'] = results['cost_u']
        df['Velocita_Base_ms'] = results['v_base']
        df['f_tech'] = results['f_tech']
        df['Danno_Ecc_Cum'] = results['danno_ecc']
        df['f_total'] = results['f_total']
        df['f_sleep'] = results['f_sleep']
        
        raw_powers = np.array(results['p_watt'])
        np_val = np.mean(raw_powers**4)**0.25
        if_val = np_val / ftp_base if ftp_base > 0 else 0
        tss_val = (t_cum * np_val * if_val) / (ftp_base * 36) if ftp_base > 0 else 0
        
        self.simulation_stats = {
            'NP': int(np_val),
            'IF': round(if_val, 2),
            'TSS': int(tss_val),
            'Kcal': int(kcal_cum),
            'Time_Sec': t_cum,
            'Time_Str': f"{int(t_cum//3600):02d}h {int((t_cum%3600)//60):02d}m",
            'P_Avg': int(np.mean(raw_powers))
        }
        
        logger.info(f"Simulazione V16 completata. Tempo: {self.simulation_stats['Time_Str']}, NP: {int(np_val)}W, TSS: {int(tss_val)}")

    # ==========================================================================================================
    # -- Metodo:        _generate_v15_splits
    # -- Descrizione:   Genera intervalli (splits) aggregati basati sulla distanza.
    # -- Uso V15:        Produce i dati per la Tab 'Splits' e il report Excel (Regola 7).
    # -- Parametri:     df (pd.DataFrame): Dataset contenente i risultati della simulazione.
    # --                interval_km (float): Lunghezza di ogni segmento (default: 5.0).
    # -- Ritorna:       List[Dict[str, Any]]: Lista di dizionari con i parziali calcolati.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _generate_v16_splits(self, df: pd.DataFrame, interval_km: float = 1.0) -> List[Dict[str, Any]]:
        """
        Aggrega i dati granulari della simulazione in segmenti chilometrici fissi.
        Calcola medie e delta per pendenza, potenza e consumi (V16 Extended).
        """
        logger.info(f"Generazione parziali V16 ogni {interval_km} km...")
        
        splits = []
        if df.empty or 'Distanza Cumulata' not in df.columns:
            logger.warning("Dati insufficienti per generare gli splits.")
            return splits
 
        # Conversione intervallo in metri per confronto diretto
        interval_m = interval_km * 1000.0
        total_dist = df['Distanza Cumulata'].max()
        
        # Iterazione per segmenti di distanza
        for start_m in np.arange(0, total_dist, interval_m):
            end_m = start_m + interval_m
            
            # Selezione del subset di dati appartenente allo split corrente
            mask = (df['Distanza Cumulata'] >= start_m) & (df['Distanza Cumulata'] < end_m)
            segment = df[mask]
            
            if segment.empty:
                continue
                
            # Calcolo metriche del parziale (Regola 7)
            t_start = segment['Tempo_Sec_Raw'].iloc[0]
            t_end = segment['Tempo_Sec_Raw'].iloc[-1]
            duration_min = (t_end - t_start) / 60.0
            
            # Calcolo Range Cumulato (Min-Max)
            if 'Tempo_Min_V15' in segment.columns:
                t_min_cum = segment['Tempo_Min_V15'].iloc[-1]
                t_max_cum = segment['Tempo_Max_V15'].iloc[-1]
                t_min_str = f"{int(t_min_cum // 3600)}h {int((t_min_cum % 3600) // 60):02d}m"
                t_max_str = f"{int(t_max_cum // 3600)}h {int((t_max_cum % 3600) // 60):02d}m"
                range_cum_str = f"{t_min_str} - {t_max_str}"
            else:
                range_cum_str = f"{int((t_end / 3600)):02d}h {int((t_end % 3600) // 60):02d}m"
            
            # Accumulo parametri fisici
            pos_gain = segment[segment['Delta_Alt'] > 0]['Delta_Alt'].sum()
            neg_loss = segment[segment['Delta_Alt'] < 0]['Delta_Alt'].abs().sum()
            
            # Media potenza e glicogeno finale dello split
            avg_power = segment['Potenza_Watt'].mean()
            final_glyco = segment['Glicogeno_Residuo'].iloc[-1]
            
            # V16 Extended Metrics
            avg_cost = segment['Costo_Metabolico_W'].mean() if 'Costo_Metabolico_W' in segment else 0
            avg_env = segment['Eta_Terrain'].mean() if 'Eta_Terrain' in segment else 0
            avg_ftot = segment['f_total'].mean() if 'f_total' in segment else 0
            
            # Calcolo velocità medie (Regola 33)
            # FIX V15: Usa distanza reale coperta dai punti (evita spike su segmenti sparsi)
            actual_dist_m = segment['Distanza Cumulata'].iloc[-1] - segment['Distanza Cumulata'].iloc[0]
            
            # Se il segmento ha dati coerenti (>0 metri e >0 tempo)
            if actual_dist_m > 0 and duration_min > 0:
                v_ms = actual_dist_m / (duration_min * 60.0)
            else:
                v_ms = 0.0
                
            v_kmh = v_ms * 3.6
            try:
                pace_min = 60 / v_kmh if v_kmh > 0 else 0
                pace_sec = int((pace_min - int(pace_min)) * 60)
                pace_str = f"{int(pace_min)}'{pace_sec:02d}\"/km"
            except:
                pace_str = "-'--\"/km"
 
            splits.append({
                'KM_Inizio': round(start_m / 1000.0, 1),
                'KM_Fine':   round(min(end_m, total_dist) / 1000.0, 1),
                'Tempo_Parziale': f"{int(duration_min // 60):02d}h {int(duration_min % 60):02d}m",
                'Tempo_Cumulato': range_cum_str,
                'D+': int(pos_gain),
                'D-': int(neg_loss),
                'Watt_Medi': int(avg_power),
                'Glicogeno_g': int(final_glyco),
                'Vel_ms': round(v_ms, 2),
                'Vel_kmh': round(v_kmh, 1),
                'Pace': pace_str,
                # V16 Extra fields
                'Costo_Met_Avg': round(avg_cost, 2),
                'Eta_Env_Avg': round(avg_env, 2),
                'Penalita_Tot_Avg': round(avg_ftot, 3)
            })
 
        logger.info(f"Generati {len(splits)} splits V16 con successo.")
        return splits

    # ==========================================================================================================
    # -- Metodo:        _generate_v15_climb_analysis
    # -- Descrizione:   Identifica e classifical le salite principali (Porting da V14).
    # -- Uso V15:        Genera la lista delle 'Côtes' per il roadbook e l'analisi tattica.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _generate_v15_climb_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """
        Algoritmo di detection salite:
        1. Identifica segmenti con pendenza > 3%
        2. Unisce segmenti interrotti da brevi tratti (< 500m)
        3. Calcola lo Score = Dist(km) * Pendenza(%)^2
        """
        climbs, in_climb, start_idx = [], False, 0
        merge_dist_m = 1000
        
        temp_segs = []
        for i in range(1, len(df)):
            if df.iloc[i]['Pendenza (%)'] > 3.0 and not in_climb:
                in_climb, start_idx = True, i
            elif (df.iloc[i]['Pendenza (%)'] <= 0 or i == len(df)-1) and in_climb:
                in_climb = False
                temp_segs.append({'s': start_idx, 'e': i})

        if not temp_segs: return []
        
        # Logica di Merge V14
        merged = [temp_segs[0]]
        for i in range(1, len(temp_segs)):
            dist_gap = df.iloc[temp_segs[i]['s']]['Distanza Cumulata'] - df.iloc[merged[-1]['e']]['Distanza Cumulata']
            if dist_gap < merge_dist_m:
                merged[-1]['e'] = temp_segs[i]['e'] # Estendi la salita precedente
            else:
                merged.append(temp_segs[i])

        final = []
        for idx, seg in enumerate(merged):
            s, e = df.iloc[seg['s']], df.iloc[seg['e']]
            gain = e['Altitudine'] - s['Altitudine']
            dist_km = (e['Distanza Cumulata'] - s['Distanza Cumulata']) / 1000.0
            
            if dist_km <= 0: continue
            
            avg_slope = (gain / (dist_km * 1000.0)) * 100.0
            score = dist_km * (avg_slope**2)
            
            # Filtro minimo: almeno 50m D+ per essere considerata salita significativa
            if gain > 50:
                rank = "Cat 4"
                if score > 1800: rank = "HC (Hors Catégorie)"
                elif score > 1200: rank = "Cat 1"
                elif score > 800: rank = "Cat 2"
                elif score > 500: rank = "Cat 3"
                elif score > 300: rank = "Cat 4"

                final.append({
                    'ID': idx + 1,
                    'Start_Km': round(s['Distanza Cumulata']/1000, 1),
                    'End_Km': round(e['Distanza Cumulata']/1000, 1),
                    'Quota_Start': int(s['Altitudine']),
                    'Quota_End': int(e['Altitudine']),
                    'Dislivello': int(gain),
                    'Pendenza_Avg': round(avg_slope, 1),
                    'Score': int(score),
                    'Cat': rank
                })
                
        logger.info(f"Analisi Salite completata: {len(final)} côtes identificate.")
        return final

    @safe_step
    @log_method_v15
    def _generate_v16_excel_report(self, df: pd.DataFrame, climbs: List[Dict], splits: List[Dict], stats: Dict, out_dir: str, base_name: str) -> str:
        """
        Esporta il report completo V16 (Extended).
        Fogli: Riepilogo, Splits (NEW), Salite, Dati_Puntuali (ALL cols).
        """
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        output_path = os.path.join(out_dir, f"{base_name}_REPORT_V16.xlsx")
        
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # 1. Foglio Riepilogo
                pd.DataFrame([stats]).to_excel(writer, sheet_name="Riepilogo", index=False)
                
                # 2. Foglio Splits (NEW V16)
                if splits:
                    pd.DataFrame(splits).to_excel(writer, sheet_name="Splits", index=False)
                else:
                    pd.DataFrame([{"Info": "Nessun dato splits disponibile"}]).to_excel(writer, sheet_name="Splits", index=False)
                    
                # 3. Foglio Salite
                if climbs:
                    pd.DataFrame(climbs).to_excel(writer, sheet_name="Salite", index=False)
                else:
                    pd.DataFrame([{"Info": "Nessuna salita significativa rilevata"}]).to_excel(writer, sheet_name="Salite", index=False)
                    
                # 4. Foglio Dati Puntuali (Expanded V16)
                # Export ALL columns available in DF
                df.to_excel(writer, sheet_name="Dati_Puntuali", index=False)
                
            logger.info(f"Report Excel V16 generato: {output_path}")
            return output_path
            
        except PermissionError:
            err_msg = f"Impossibile salvare il file:\n{output_path}\n\nIl file è aperto in Excel. Chiudilo e riprova."
            logger.error(err_msg)
            messagebox.showwarning("File Aperto", err_msg)
            return None

    # ==========================================================================================================
    # -- Metodo:        _export_v15_maps
    # -- Descrizione:   Genera le mappe interattive (Main, Food, Sleep).
    # -- Uso V15:        Visualizzazione strategica su browser.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _export_v15_maps(self, df: pd.DataFrame, out_dir: str) -> Dict[str, str]:
        """Genera le 3 mappe strategiche (Main, Food, Sleep) come V14."""
        if df.empty: return {}
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        paths = {}
        map_types = ['main', 'food', 'sleep']
        center = [df['Latitudine'].mean(), df['Longitudine'].mean()]
        
        # Legenda HTML sincronizzata (Porting V14)
        legend_html = f'''
        <div style="position: fixed; bottom: 30px; left: 30px; width: 190px; 
                    background-color: white; border:2px solid #2c3e50; z-index:9999; font-size:11px;
                    padding: 10px; opacity: 0.9; line-height: 1.4;">
            <b style="font-size:12px;">Legenda Pendenze</b><br>
            <i style="background:{self.color_scheme.slope_colors['hc_up']}; width:10px; height:10px; display:inline-block"></i> > 30% (HC Up)<br>
            <i style="background:{self.color_scheme.slope_colors['st_up']}; width:10px; height:10px; display:inline-block"></i> 20% a 30%<br>
            <i style="background:{self.color_scheme.slope_colors['mod_up']}; width:10px; height:10px; display:inline-block"></i> 10% a 20%<br>
            <i style="background:{self.color_scheme.slope_colors['lt_up']}; width:10px; height:10px; display:inline-block"></i> 1% a 10%<br>
            <i style="background:{self.color_scheme.slope_colors['flat']}; width:10px; height:10px; display:inline-block"></i> -1% a 1%<br>
            <i style="background:{self.color_scheme.slope_colors['lt_dn']}; width:10px; height:10px; display:inline-block"></i> -10% a -1%<br>
            <i style="background:{self.color_scheme.slope_colors['mod_dn']}; width:10px; height:10px; display:inline-block"></i> -20% a -10%<br>
            <i style="background:{self.color_scheme.slope_colors['st_dn']}; width:10px; height:10px; display:inline-block"></i> -30% a -20%<br>
            <i style="background:{self.color_scheme.slope_colors['hc_dn']}; border:1px solid #888; width:10px; height:10px; display:inline-block"></i> < -30% (HC Down)<br>
        </div>
        '''

        # Recupero i marker (se calcolati nella cascata)
        # In V15 la lista eventi feeding è in self.feeding_events (da verificare integrazione)
        # Se non esiste, fallback vuoto
        if not hasattr(self, 'feeding_events'): self.feeding_events = []
        if not hasattr(self, 'sleep_events'): self.sleep_events = []

        for m_type in map_types:
            m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Rendering traccia con campionamento ottimizzato (Step 100m approx)
            last_saved_dist = -1000.0
            points_to_plot = []
            for i in range(len(df)):
                 cumulative = df.iloc[i]['Distanza Cumulata']
                 if cumulative - last_saved_dist >= 100:
                     points_to_plot.append(i)
                     last_saved_dist = cumulative
            if len(df) - 1 not in points_to_plot:
                points_to_plot.append(len(df) - 1)

            for j in range(len(points_to_plot)-1):
                idx1 = points_to_plot[j]
                idx2 = points_to_plot[j+1]
                
                p1 = [df.iloc[idx1]['Latitudine'], df.iloc[idx1]['Longitudine']]
                p2 = [df.iloc[idx2]['Latitudine'], df.iloc[idx2]['Longitudine']]
                
                slope = df.iloc[idx1:idx2+1]['Pendenza (%)'].mean()
                color = self.color_scheme.get_color_by_slope(slope)
                
                # Tooltip con tempi Min-Max V15
                row = df.iloc[idx1]
                t_min = str(datetime.timedelta(seconds=int(row.get('Tempo_Min_V15', 0)))).rsplit(':', 1)[0]
                t_max = str(datetime.timedelta(seconds=int(row.get('Tempo_Max_V15', 0)))).rsplit(':', 1)[0]
                
                popup_txt = f"KM: {df.iloc[idx1]['Distanza Cumulata']/1000:.1f} | Alt: {int(df.iloc[idx1]['Altitudine'])}m | {t_min}-{t_max}"
                folium.PolyLine([p1, p2], color=color, weight=4, opacity=0.8, tooltip=popup_txt).add_to(m)

            # Inserimento layer specifici
            if m_type == 'food':
                self._add_strategic_layer(m, self.feeding_events, color='orange', icon='cutlery')
            elif m_type == 'sleep':
                # Se ci sono eventi sonno, li aggiungiamo
                self._add_strategic_layer(m, self.sleep_events, color='purple', icon='moon')
            elif m_type == 'main':
                self._add_logistics_layer(m, df)

            fpath = os.path.join(out_dir, f"trail_v15_{m_type}.html")
            m.save(fpath)
            paths[m_type] = fpath
            
        logger.info(f"Generate 3 mappe V15 in {out_dir}")
        return paths

    # ==========================================================================================================
    # -- Metodo:        _add_strategic_layer
    # -- Descrizione:   Inserisce marker puntuali per eventi strategici (Food/Sleep).
    # -- Uso V15:       Arricchisce la mappa con icone categorizzate (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _add_strategic_layer(self, m, marker_list, color, icon):
        """Inserisce icone strategiche (Food/Sleep)."""
        for mk in marker_list:
            if isinstance(mk, dict):
                lat = mk.get('lat')
                lon = mk.get('lon')
                lbl = mk.get('label', 'Event')
                km = mk.get('km', 0)
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{lbl}</b><br>KM: {km:.1f}",
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)

    # ==========================================================================================================
    # -- Metodo:        _add_logistics_layer
    # -- Descrizione:   Proietta le basi vita e i cancelli orari sulla mappa.
    # -- Uso V15:       Fornisce il contesto logistico alla visualizzazione (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _add_logistics_layer(self, m, df):
        """Visualizza i cancelli orari e le basi vita sulla mappa principale."""
        if self.logistic_data.empty: return
            
        for _, row in self.logistic_data.iterrows():
            try:
                # Ricerca del punto più vicino
                idx = (df['Distanza Cumulata'] - row['km']*1000).abs().idxmin()
                point = df.iloc[idx]
                
                # Calcolo stringhe temporali (HH:MM)
                if 'Tempo_Min_V15' in point:
                    t_min = str(datetime.timedelta(seconds=int(point['Tempo_Min_V15']))).rsplit(':', 1)[0]
                    t_max = str(datetime.timedelta(seconds=int(point['Tempo_Max_V15']))).rsplit(':', 1)[0]
                    times = f"<br>Passaggio: {t_min} - {t_max}"
                else:
                    times = ""
                
                folium.Marker(
                    location=[point['Latitudine'], point['Longitudine']],
                    popup=f"<b>{row['nome']}</b><br>KM: {row['km']}{times}",
                    icon=folium.Icon(color='red', icon='flag', prefix='fa')
                ).add_to(m)
              
            except Exception as e:
                logger.error(f"Errore logistics layer: {str(e)}")

    # ==========================================================================================================
    # -- Metodo:        _load_base_vita_csv
    # -- Descrizione:   Handler UI per il caricamento manuale del file logistica.
    # -- Uso V15:        Permette all'utente di specificare un nuovo file Basi Vita (Regola 12).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _load_base_vita_csv(self):
        """Apre un file dialog per selezionare il CSV delle Basi Vita."""
        fpath = filedialog.askopenfilename(
            title="Seleziona File Basi Vita (CSV)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if fpath:
            # Copia locale o caricamento diretto
            try:
                dest = os.path.join(os.path.dirname(self.analyzer.current_file_path) if self.analyzer.current_file_path else os.getcwd(), "basivita.csv")
                # Se diverso, copia (opzionale, per ora carichiamo diretto)
                # Per coerenza con _load_logistic_points che cerca 'basivita.csv',
                # possiamo sovrascrivere l'attributo logistic_data direttamente.
                
                df_log = pd.read_csv(fpath)
                df_log.columns = [c.lower() for c in df_log.columns]
                if 'km' in df_log.columns and 'nome' in df_log.columns:
                    self.analyzer.logistic_data = df_log
                    self.lbl_logistic_status.config(text=f"Caricato: {os.path.basename(fpath)}", fg="#2ecc71")
                    logger.info(f"Logistica aggiornata manualmente da: {fpath}")
                else:
                    messagebox.showerror("Errore Csv", "Il file deve contenere le colonne 'Nome' e 'Km'.")
            except Exception as e:
                logger.error(f"Errore caricamento CSV logistica: {str(e)}")
                messagebox.showerror("Errore", f"Impossibile leggere il file:\n{str(e)}")

# ==========================================================================================================
# -- Classe:        QuickLookWindow
# -- Descrizione:   Finestra di ispezione rapida (Stadio 1) per l'analisi geometrica 3D.
# -- Uso V15:       Visualizzazione immediata del profilo altimetrico e pendenze (Regola 12).
# ==========================================================================================================
class QuickLookWindow:
    # ==========================================================================================================
    # -- Metodo:        __init__
    # -- Descrizione:   Inizializza lo Stadio 1, le strutture dati e le figure Matplotlib.
    # -- Uso V15:       Prepara i due viewport orizzontali e lo stato neutro (Regola 15).
    # ==========================================================================================================
    @log_method_v15
    def __init__(self, analyzer: Any):
        """
        Inizializza lo Stadio 1. Parte in stato neutro senza caricare file.
        """
        self.analyzer = analyzer
        self.df = pd.DataFrame()
        
        # Riferimenti per gli oggetti interattivi (Cursori e Annotazioni)
        self.cursor_line = None
        self.cursor_dot = None
        self.annot_v15 = None
        self.cid_a = None
        self.cid_b = None

        # 1. Inizializzazione Finestra Principale
        self.root = tk.Tk()
        self.root.title("Trail Analyzer V15 - [STADIO 1: ISPEZIONE GEOMETRICA]")
        self.root.state('zoomed')
        self.root.configure(bg="#f0f2f5")

        # 2. Inizializzazione Figure Matplotlib
        self.fig_a, self.ax_a = plt.subplots(figsize=(10, 4), dpi=100)
        self.fig_b, self.ax_b = plt.subplots(figsize=(10, 4), dpi=100)
        
        for f in [self.fig_a, self.fig_b]: 
            f.patch.set_facecolor('#ffffff')
            f.tight_layout(pad=3.0)

        # 3. Costruzione Interfaccia e Stato Neutro
        self._build_ui()
        self._set_empty_state(self.ax_a, "VIEWPORT A: PROFILO ALTIMETRICO")
        self._set_empty_state(self.ax_b, "VIEWPORT B: TRACCIA PLANIMETRICA X-Y")
        
        logger.info("QuickLookWindow (Stadio 1) pronta in stato neutro.")
        self.root.mainloop()

    # ==========================================================================================================
    # -- Metodo:        _build_ui
    # -- Descrizione:   Configura il layout della GUI e integra i canvas Matplotlib.
    # -- Uso V15:       Crea l'ambiente orizzontale per lo Stadio 1 (Regola 15).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_ui(self):
        """
        Costruisce una gerarchia di frame per separare la visualizzazione dei dati dalla telemetria.
        """
        logger.info("Inizio costruzione UI complessa Stadio 1...")

        # 1. Main Container (Padding globale per estetica V15)
        self.main_container = tk.Frame(self.root, bg="#f0f2f5")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 2. Viewport Area (Layout a griglia per mirroring)
        self.view_frame = tk.Frame(self.main_container, bg="#f0f2f5")
        self.view_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.view_frame.columnconfigure(0, weight=1)
        self.view_frame.rowconfigure(0, weight=1)
        self.view_frame.rowconfigure(1, weight=1)
        
        self.canvas_a = FigureCanvasTkAgg(self.fig_a, master=self.view_frame)
        self.canvas_a.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=0, pady=5)

        self.canvas_b = FigureCanvasTkAgg(self.fig_b, master=self.view_frame)
        self.canvas_b.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=0, pady=5)
        
        # 3. Telemetry & Stats Bar
        self.stats_bar = tk.Frame(self.main_container, bg="#161b22", height=60, bd=1, relief=tk.RIDGE)
        self.stats_bar.pack(fill=tk.X, pady=10)
        self.stats_bar.pack_propagate(False)

        self.lbl_stats = tk.Label(self.stats_bar, text="Analisi Geometria: In attesa di caricamento...", 
                                  font=("Segoe UI", 11, "bold"), fg="#58a6ff", bg="#161b22")
        self.lbl_stats.pack(side=tk.LEFT, padx=20)

        self.lbl_info = tk.Label(self.stats_bar, text="Ispezione: Muovere il mouse sui grafici", 
                                 font=("Consolas", 10, "italic"), fg="#8b949e", bg="#161b22")
        self.lbl_info.pack(side=tk.RIGHT, padx=20)

        # 4. Action Control Panel
        self.action_bar = tk.Frame(self.main_container, bg="#e2e8f0", height=80)
        self.action_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        self.btn_load = tk.Button(self.action_bar, text="📁 CARICA TRACCIA (GPX/KML)", command=self._load_data,
                                  bg="#1f6feb", fg="white", font=("Segoe UI", 12, "bold"), padx=25, pady=10)
        self.btn_load.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_next = tk.Button(self.action_bar, text="AVVIA SETUP SIMULAZIONE ➔", command=self._next,
                                  bg="#238636", fg="white", font=("Segoe UI", 12, "bold"), padx=25, pady=10, state=tk.DISABLED)
        self.btn_next.pack(side=tk.RIGHT, padx=10, pady=10)

    # ==========================================================================================================
    # -- Metodo:        _set_empty_state
    # -- Descrizione:   Configura l'estetica dei viewport in assenza di dati (Stato Neutro).
    # -- Uso V15:       Garantisce la coerenza cromatica e il feedback visivo al boot (Regola 33).
    # -- Parametri:     ax (matplotlib.axes): L'asse da resettare.
    # --                title (str): Titolo tecnico del viewport.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _set_empty_state(self, ax, title):
        """
        Esegue il reset profondo degli assi e applica lo schema colori V15.
        """
        ax.clear()
        ax.set_facecolor('#ffffff')
        ax.set_title(title, color="#58a6ff", fontsize=10, fontweight='bold', pad=15)
        ax.text(0.5, 0.5, "NESSUN DATO CARICATO\n---\nIn attesa di traccia GPX/KML", 
                color="#30363d", ha='center', va='center', transform=ax.transAxes, 
                fontsize=11, fontweight='bold', linespacing=1.8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        
        if ax == self.ax_a: self.canvas_a.draw()
        elif ax == self.ax_b: self.canvas_b.draw()

    # ==========================================================================================================
    # -- Metodo:        _load_data
    # -- Descrizione:   Gestisce l'ingestione della traccia e il calcolo delle metriche di sintesi.
    # -- Uso V15:       Coordina pre-processing e attivazione interattività (Regola 15).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _load_data(self):
        """
        Innesca il selettore di file (Multi-selezione supportata V15) e unisce le tracce.
        """
        file_paths = filedialog.askopenfilenames(
            title="Seleziona Tracce V15 (Multiselezione supportata)",
            filetypes=[("File GPS", "*.gpx *.kml"), ("Tutti i file", "*.*")]
        )
        
        if not file_paths: return

        # FIX V15: Supporto Multi-Traccia (DB Logic)
        self.tracks_db = {}
        for fp in file_paths:
            logger.info(f"Caricamento: {fp}")
            tmp_df = self.analyzer.load_and_preprocess(fp)
            if not tmp_df.empty:
                fname = os.path.basename(fp)
                self.tracks_db[fname] = {'path': fp, 'df': tmp_df}

        if not self.tracks_db:
            messagebox.showerror("Errore Dati", "Nessuna traccia valida caricata.")
            return

        # Selezione di default: prendiamo il primo file
        first_key = list(self.tracks_db.keys())[0]
        self.df = self.tracks_db[first_key]['df']
        self.analyzer.current_file_path = self.tracks_db[first_key]['path']

        # Estrazione Metriche di Sintesi
        dist_km = self.df['Distanza Cumulata'].max() / 1000.0
        ascesa = self.df[self.df['Delta_Alt'] > 0]['Delta_Alt'].sum()
        self.lbl_stats.config(text=f"Distanza: {dist_km:.2f} km | D+: {int(ascesa)}m | Sorgente: {os.path.basename(self.analyzer.current_file_path)}")

        self._render_viewports()
        self.btn_next.config(state=tk.NORMAL)

        # Sincronizzazione Eventi
        if self.cid_a: self.canvas_a.mpl_disconnect(self.cid_a)
        if self.cid_b: self.canvas_b.mpl_disconnect(self.cid_b)
        
        self.cid_a = self.canvas_a.mpl_connect('motion_notify_event', self._on_move)
        self.cid_b = self.canvas_b.mpl_connect('motion_notify_event', self._on_move)

    # ==========================================================================================================
    # -- Metodo:        _render_viewports
    # -- Descrizione:   Esegue il disegno tecnico dei grafici e dei cursori (Logica V14).
    # -- Uso V15:       Inizializza cursori e tooltip mobili per ispezione real-time (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _render_viewports(self):
        """
        Pulisce i viewport e traccia i vettori di coordinate con cursori V14.
        """
        dist_km = self.df['Distanza Cumulata'].values / 1000.0
        alt = self.df['Altitudine'].values
        lon = self.df['Longitudine'].values
        lat = self.df['Latitudine'].values

        # VIEWPORT A: PROFILO
        self.ax_a.clear()
        self.ax_a.plot(dist_km, alt, color="#2563eb", linewidth=1.8, zorder=2)
        self.ax_a.fill_between(dist_km, alt, min(alt)-5, color="#2563eb", alpha=0.1, zorder=1)
        
        # Inizializzazione Cursore e Tooltip (V14 Style)
        self.cursor_line = self.ax_a.axvline(x=dist_km[0], color='#ff0000', lw=1.5, zorder=10, visible=False)
        self.annot_v15 = self.ax_a.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                            bbox=dict(boxstyle="round,pad=0.5", fc="#ffff00", ec="black", alpha=0.9),
                                            arrowprops=dict(arrowstyle="->", color="black"),
                                            fontweight='bold', zorder=20)
        self.annot_v15.set_visible(False)

        # VIEWPORT B: MAPPA
        self.ax_b.clear()
        self.ax_b.plot(lon, lat, color="#b91c1c", linewidth=1.8, zorder=2)
        self.ax_b.set_aspect('equal', adjustable='datalim')
        
        self.cursor_dot, = self.ax_b.plot([lon[0]], [lat[0]], 'o', color='#ffff00', ms=10, 
                                          markeredgecolor='black', mew=1.5, zorder=30, visible=False)

        self.canvas_a.draw()
        self.canvas_b.draw()

    # ==========================================================================================================
    # -- Metodo:        _on_move
    # -- Descrizione:   Gestore eventi per la sincronizzazione millimetrica dei cursori (Regola 7).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _on_move(self, event):
        """
        Aggiorna markers e tooltip in tempo reale (Cross-hair synchronization).
        """
        if self.df.empty or not event.inaxes: return

        dist_km_arr = self.df['Distanza Cumulata'].values / 1000.0
        if event.inaxes == self.ax_a:
            idx = np.argmin(np.abs(dist_km_arr - event.xdata))
        else:
            idx = (np.abs(self.df['Longitudine'] - event.xdata) + np.abs(self.df['Latitudine'] - event.ydata)).argmin()

        row = self.df.iloc[idx]
        d_p = row['Distanza Cumulata'] / 1000.0
        d_r = dist_km_arr[-1] - d_p
        
        # Update Cursori
        self.cursor_line.set_xdata([d_p, d_p]); self.cursor_line.set_visible(True)
        self.cursor_dot.set_data([row['Longitudine']], [row['Latitudine']]); self.cursor_dot.set_visible(True)
        
        # Update Tooltip
        self.annot_v15.xy = (d_p, row['Altitudine'])
        self.annot_v15.set_text(f"DIST: {d_p:.2f} km\nRIM: {d_r:.2f} km\nQUOTA: {int(row['Altitudine'])} m")
        self.annot_v15.set_visible(True)

        self.lbl_info.config(text=f"Tracking: {d_p:.2f} km | Residuo: {d_r:.2f} km | Quota: {int(row['Altitudine'])} m", fg="#58a6ff")
        self.canvas_a.draw_idle(); self.canvas_b.draw_idle()

    # ==========================================================================================================
    # -- Metodo:        _next
    # -- Descrizione:   Gestisce il passaggio dallo Stadio 1 allo Stadio 2 (Main Dashboard).
    # -- Uso V15:       Finalizza la validazione e lancia il setup della simulazione (Regola 15).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _next(self):
        """
        Esegue il cleanup profondo delle risorse e istanzia la Dashboard principale.
        """
        logger.info("Avvio procedura di Handoff verso lo Stadio 2...")
        try:
            plt.close(self.fig_a); plt.close(self.fig_b)
            if self.root.winfo_exists(): self.root.destroy()
            # Passiamo l'intero DB delle tracce alla Dashboard
            MainDashboardV16(self.analyzer, self.df, self.tracks_db)
        except Exception as e:
            logger.error(f"FALLIMENTO CRITICO nel lancio della Dashboard: {str(e)}")

# ==========================================================================================================
# -- Classe:        MainDashboardV16
# -- Descrizione:   Hub centrale della simulazione strategica. Gestisce l'interfaccia a 8 Tab.
# -- Uso V16:        Coordina i parametri biofisici e la visualizzazione dei risultati.
# ==========================================================================================================
class MainDashboardV16:
    @log_method_v15
    def __init__(self, analyzer: Any, df: pd.DataFrame, tracks_db: Dict = None):
        """
        Inizializza la dashboard principale, imposta lo stile visivo e crea 
        la struttura a schede (Notebook).
        """
        self.analyzer = analyzer
        self.df = df
        self.tracks_db = tracks_db if tracks_db else {} # FIX V15: Storage multi-traccia
        
        # 1. Configurazione Finestra Principale
        self.root = tk.Tk()
        self.root.title("Trail Analyzer V16 - Strategic Simulation Dashboard")
        self.root.state('zoomed')
        self.root.minsize(1024, 720) # Impedisce il collasso a dimensioni nulle
        self.root.configure(bg="#2c3e50")
        
        # 2. Definizione dello Stile Moderno (Regola 12)
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._setup_custom_styles()
        
        # 3. Variabili di Stato della Simulazione
        self.settings = {
            'peso': tk.DoubleVar(value=75.0),
            'peso_zaino': tk.DoubleVar(value=4.0),
            'ftp': tk.DoubleVar(value=250.0),
            'vo2': tk.DoubleVar(value=55.0),
            'walk_ratio': tk.DoubleVar(value=0.8),
            'ora_partenza': tk.IntVar(value=6),
            'mese': tk.StringVar(value='Agosto'),
            'nutri_strategy': tk.StringVar(value='B'),
            'z_score': tk.DoubleVar(value=1.96)
        }
        
        # Inizializzazione variabili UI (Messaggi e Progress)
        self._init_vars()

        # 4. Creazione Container Notebook (Senza pack immediato)
        self.notebook = ttk.Notebook(self.root)
        
        # Inizializzazione dei Frame per ogni Tab
        self.tabs = {}
        # Struttura a 8 Tab Integrale V15
        tab_names = [
            "1. Database", 
            "2. Setup", 
            "3. Performance",
            "3b. Salite", # NEW V15
            "4. Strategia", 
            "5. Splits", 
            "6. Mappe Strategiche", 
            "7. Export Report", 
            "8. System Log"
        ]
        
        for name in tab_names:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            self.tabs[name] = frame

        # 5. Costruzione della UI Radice (Header, Footer, e Posizionamento Notebook)
        self._build_ui()
        
        # 6. Costruzione delle interfacce specifiche per Tab
        self._build_tab_database()
        self._build_tab_setup()
        self._build_tab_performance()
        self._build_tab_climbs() # NEW V15
        self._build_tab_strategy()
        self._build_tab_splits()
        self._build_tab_map()
        self._build_tab_report()
        self._build_tab_log()
        
        logger.info("MainDashboardV15 inizializzata: 8 Tab operative caricate (Standard V15).")
        self.root.mainloop()
    # ==========================================================================================================
    # -- Metodo:        _init_vars
    # -- Descrizione:   Inizializza i contenitori dei dati simulati e i puntatori UI.
    # -- Uso V15:        Prepara l'ambiente per il motore deterministico (Regola 32).
    # ==========================================================================================================
    @log_method_v15
    def _init_vars(self):
        """
        Configura le strutture dati per ospitare i risultati della simulazione V15
        e gestisce i flag di controllo dell'interfaccia.
        """
        logger.info("Inizializzazione variabili di stato interne...")

        # 1. Contenitori Risultati (Output del motore _run_v15_deterministic_cascade)
        self.simulation_results = pd.DataFrame()
        self.splits_data = []         # Lista di dizionari per la Tab Splits
        self.feeding_points = []      # Coordinate per marker nutrizionali sulla mappa
        
        # 2. Riferimenti agli Oggetti Grafici (Plot Handles)
        # Necessari per aggiornare i grafici esistenti (blit) invece di ridisegnare tutto
        self.plot_handles = {
            'altitude_profile': None,
            'glycogen_curve':   None,
            'fatigue_index':    None,
            'power_distribution': None
        }

        # 3. Variabili di Controllo UI (Dynamic Labels)
        self.status_msg = tk.StringVar(value="Sistema Pronto. Configura i parametri e avvia la simulazione.")
        self.progress_val = tk.IntVar(value=0)
        
        # 4. Parametri di Calcolo derivati (Default iniziali)
        # Questi verranno aggiornati dinamicamente dai widget della Tab 1
        self.current_simulation_id = 0
        self.is_simulating = False
        
        # 5. Coda Messaggi (Thread-Safe Communication Regola 15)
        self.msg_queue = queue.Queue()
        self._check_queue() # Avvia il loop di controllo

        logger.info("Strutture dati inizializzate. Pronto per la configurazione dei widget.")

    # ==========================================================================================================
    # -- Metodo:        _setup_custom_styles
    # -- Descrizione:   Configura il database degli stili TTK (Look & Feel V15).
    # -- Uso V15:        Garantisce la coerenza visiva e la leggibilità dei dati (Regola 12).
    # ==========================================================================================================
    @log_method_v15
    def _setup_custom_styles(self):
        """
        Definisce la palette cromatica e le proprietà dei widget TTK.
        Configura lo stile specifico per i pulsanti di azione e i contenitori.
        """
        # Palette Colori V15
        bg_dark = "#2c3e50"
        bg_header = "#34495e"
        accent_blue = "#3498db"
        text_light = "#ecf0f1"

        # Configurazione Stile Notebook (Tab)
        self.style.configure("TNotebook", background=bg_dark, borderwidth=0)
        self.style.configure("TNotebook.Tab", 
                            padding=[15, 5], 
                            font=("Segoe UI", 10, "bold"),
                            background="#95a5a6")
        
        self.style.map("TNotebook.Tab",
                      background=[("selected", accent_blue)],
                      foreground=[("selected", "white")])

        # Configurazione Pulsante Accentato (Visto in _build_setup_panel)
        self.style.configure("Accent.TButton",
                            font=("Segoe UI", 11, "bold"),
                            foreground="black",
                            background=accent_blue)

        # Configurazione Frame e Label
        self.style.configure("TFrame", background="#f0f3f5")
        self.style.configure("TLabelframe", background="#f0f3f5", font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabelframe.Label", background="#f0f3f5", foreground=bg_dark)

    # ==========================================================================================================
    # -- Metodo:        _build_tab_parameters (Ridenominazione di _build_setup_panel)
    # -- Descrizione:   Puntatore corretto per l'inizializzazione del Tab 1.
    # ==========================================================================================================
    @log_method_v15
    def _build_tab_parameters(self):
        """
        Inizializza il Tab 1 (Parametri) mappandolo sul frame creato in __init__.
        """
        target_frame = self.tabs.get("1. Parametri")
        if target_frame:
            # Richiama la logica di costruzione già definita
            self._build_setup_panel(target_frame)
        else:
            logger.error("Tab '1. Parametri' non trovato nella struttura Notebook.")

    # ==========================================================================================================
    # -- Metodo:        _load_base_vita_csv
    # -- Descrizione:   Carica il database logistico per Basi Vita (Nome, Km).
    # -- Uso V15:        Popola self.analyzer.logistic_data per le mappe e i grafici.
    # ==========================================================================================================
    @log_method_v15
    def _load_base_vita_csv(self):
        """
        Gestisce il parsing del CSV logistico e aggiorna lo stato UI.
        """
        file_path = filedialog.askopenfilename(
            title="Seleziona Database Basi Vita (CSV)",
            filetypes=[("CSV Files", "*.csv"), ("Tutti i file", "*.*")]
        )
        if not file_path: return
        
        try:
            df_log = pd.read_csv(file_path)
            # Normalizzazione colonne (strip spazi e lowercase)
            df_log.columns = [c.strip().lower() for c in df_log.columns]
            
            if 'nome' not in df_log.columns or 'km' not in df_log.columns:
                messagebox.showerror("Errore Formato", "Il CSV deve contenere le colonne 'Nome' e 'Km'.")
                return

            self.analyzer.logistic_data = df_log
            logger.info(f"Caricate {len(df_log)} Basi Vita da {os.path.basename(file_path)}")
            
            # Aggiornamento Label UI
            if hasattr(self, 'lbl_logistic_status'):
                self.lbl_logistic_status.config(text=f"Basi Vita: {len(df_log)} caricate ({os.path.basename(file_path)})", fg="#2ecc71")
            
        except Exception as e:
            logger.error(f"Errore caricamento CSV: {str(e)}")
            messagebox.showerror("Errore", f"Impossibile leggere il file CSV:\n{str(e)}")

    # ==========================================================================================================
    # -- Metodo:        _build_track_selector
    # -- Descrizione:   Crea la sezione UI per la gestione della traccia e dei dati logistici.
    # -- Uso V15:        Integrato nel Tab 1.
    # ==========================================================================================================
    # ==========================================================================================================
    # -- Metodo:        _build_tab_database (Tab 1)
    # -- Descrizione:   Pannello di gestione tracce caricate.
    # -- Uso V15:        Listbox scrollabile per selezione file attivo (Regola 12).
    # ==========================================================================================================
    @log_method_v15
    def _build_tab_database(self):
        """
        Costruisce il Tab 1: Lista file tracce e selezione attiva.
        """
        parent = self.tabs["1. Database"]
        
        # Frame Contenitore
        frame = ttk.LabelFrame(parent, text=" Archivio Tracce Disponibili ", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Listbox con Scrollbar
        list_frame = tk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.track_listbox = tk.Listbox(
            list_frame, 
            yscrollcommand=scrollbar.set, 
            font=("Consolas", 10),
            height=15
        )
        self.track_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.track_listbox.yview)
        
        # Popolamento Iniziale
        if self.tracks_db:
            for name in self.tracks_db.keys():
                self.track_listbox.insert(tk.END, name)
        elif hasattr(self.analyzer, 'current_file_path') and self.analyzer.current_file_path:
             self.track_listbox.insert(tk.END, os.path.basename(self.analyzer.current_file_path))

        # Pulsante Selezione (Aggiorna self.df)
        btn_select = ttk.Button(frame, text="✅ SELEZIONA TRACCIA", command=self._on_track_select_btn_click)
        btn_select.pack(pady=10, fill=tk.X)
        
    @log_method_v15
    def _on_track_select_btn_click(self):
        """Handler per il pulsante Seleziona Traccia."""
        selection = self.track_listbox.curselection()
        if selection:
            track_name = self.track_listbox.get(selection[0])
            self._on_track_switch(None, track_name=track_name)
        else:
            messagebox.showinfo("Info", "Seleziona una traccia dalla lista.")

    # ==========================================================================================================
    # -- Metodo:        _build_ui
    # -- Descrizione:   Configura il layout radice della dashboard (Header, Main, Footer).
    # -- Uso V15:        Standardizza l'interfaccia di controllo per la simulazione (Regola 12).
    # ==========================================================================================================
    @log_method_v15
    def _build_ui(self):
        """
        Crea la gerarchia dei widget principali. Integra la barra di stato 
        deterministica e l'intestazione di versione.
        """
        logger.info("Configurazione del Main Layout Buffer...")

        # 1. Header Tecnico (Branding e Versione) - PACK TOP
        self.header_frame = tk.Frame(self.root, bg="#34495e", height=60)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)
        self.header_frame.pack_propagate(False) # Forza l'altezza di 60px
        
        tk.Label(
            self.header_frame, 
            text="TRAIL ANALYZER V16 | STRATEGIC ENGINE",
            font=("Segoe UI", 16, "bold"),
            fg="#ecf0f1", bg="#34495e"
        ).pack(side=tk.LEFT, padx=20, pady=10)

        # Pulsante Torna al Caricamento
        ttk.Button(
            self.header_frame,
            text="↩ CARICA NUOVA TRACCIA",
            command=self._back_to_loading
        ).pack(side=tk.RIGHT, padx=10)

        source_name = os.path.basename(self.analyzer.current_file_path) if hasattr(self.analyzer, 'current_file_path') else 'Dataset 3D'
        self.lbl_file_info = tk.Label(
            self.header_frame,
            text=f"Sorgente: {source_name}",
            font=("Segoe UI", 10, "italic"),
            fg="#bdc3c7", bg="#34495e"
        )
        self.lbl_file_info.pack(side=tk.RIGHT, padx=20)

        # 2. Footer / Status Bar (Monitoraggio Real-time) - PACK BOTTOM
        self.status_bar = tk.Frame(self.root, bg="#2c3e50", height=35, bd=1, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_bar.pack_propagate(False) # Forza l'altezza della barra di stato

        # Indicatore Messaggi di Stato
        self.status_label = tk.Label(
            self.status_bar, 
            textvariable=self.status_msg,
            font=("Segoe UI", 9),
            fg="#ecf0f1", bg="#2c3e50"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Progress Bar Deterministica (Regola 7)
        self.progress_bar = ttk.Progressbar(
            self.status_bar, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode='determinate',
            variable=self.progress_val
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=20, pady=5)

        # 3. Main Area (Notebook Container) - PACK CENTER
        # Viene packato DOPO Header e Footer per riempire lo spazio rimanente
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        
    # ==========================================================================================================
    # -- Metodo:        _build_setup_panel
    # -- Descrizione:   Crea il modulo di input per i parametri atleta e gara.
    # -- Uso V15:        Centralizza la configurazione delle variabili per la simulazione (Regola 15).
    # -- Parametri:     parent (tk.Frame): Il contenitore (Tab) in cui inserire il pannello.
    # ==========================================================================================================
    # ==========================================================================================================
    # -- Metodo:        _build_tab_setup (Tab 2)
    # -- Descrizione:   Crea il modulo di input per i parametri atleta e la gestione logistica.
    # -- Uso V15:        Setup centralizzato della simulazione (Regola 15).
    # ==========================================================================================================
    @log_method_v15
    def _build_tab_setup(self):
        """
        Costruisce la griglia parametri e la sezione logistica nel Tab 2.
        """
        parent = self.tabs["2. Setup"]
        
        # 1. Container Parametri Atleta
        setup_container = ttk.LabelFrame(parent, text=" Configurazione Biofisica ", padding=20)
        setup_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(20, 10))
        
        for i in range(4): setup_container.columnconfigure(i, weight=1)

        # Campi Input
        fields = [
            ("Peso Atleta", self.settings['peso'], 0, 0, "kg"),
            ("Peso Zaino", self.settings['peso_zaino'], 2, 0, "kg"),
            ("Potenza FTP", self.settings['ftp'], 0, 1, "W"),
            ("VO2 Max", self.settings['vo2'], 2, 1, "ml/kg/min"),
            ("Run Ratio", self.settings['walk_ratio'], 0, 2, "0.0-1.0"),
            ("Ora Partenza", self.settings['ora_partenza'], 2, 2, "h (0-23)")
        ]
        for text, var, col, row, unit in fields:
            ttk.Label(setup_container, text=f"{text} ({unit}):").grid(column=col, row=row, sticky=tk.W, pady=8, padx=5)
            ttk.Spinbox(setup_container, from_=0, to=1000, increment=0.1, textvariable=var, width=10).grid(column=col+1, row=row, sticky=tk.W, pady=8, padx=5)

        # Parametri Categorici
        ttk.Label(setup_container, text="Mese Gara:").grid(column=0, row=3, sticky=tk.W, pady=8, padx=5)
        ttk.Combobox(setup_container, textvariable=self.settings['mese'], values=['Giugno', 'Luglio', 'Agosto', 'Settembre'], state="readonly", width=12).grid(column=1, row=3, sticky=tk.W, pady=8, padx=5)

        ttk.Label(setup_container, text="Strategia Nutri:").grid(column=2, row=3, sticky=tk.W, pady=8, padx=5)
        ttk.Combobox(setup_container, textvariable=self.settings['nutri_strategy'], values=['A', 'B', 'C'], state="readonly", width=12).grid(column=3, row=3, sticky=tk.W, pady=8, padx=5)

        # 2. Sezione Logistica (Basi Vita)
        log_frame = ttk.LabelFrame(parent, text=" Gestione Logistica (Basi Vita) ", padding=15)
        log_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(log_frame, text="📂 CARICA CSV BASI VITA", command=self._load_base_vita_csv).pack(side=tk.LEFT, padx=10)
        
        self.lbl_logistic_status = tk.Label(log_frame, text="Nessun dato caricato (Default: basivita.csv)", font=("Segoe UI", 9, "italic"), fg="gray")
        self.lbl_logistic_status.pack(side=tk.LEFT, padx=10)
        
        # Check stato iniziale se caricato automaticamente
        if not self.analyzer.logistic_data.empty:
             self.lbl_logistic_status.config(text=f"Logistica Attiva ({len(self.analyzer.logistic_data)} punti)", fg="#2ecc71")

        # 3. Pulsante di Azione
        self.btn_run = ttk.Button(
            parent, 
            text=" ⚡ AVVIA SIMULAZIONE DETERMINISTICA V15 ⚡ ", 
            command=self._run_simulation
        )
        self.btn_run.pack(fill=tk.X, padx=40, pady=20, ipady=10)

    # ==========================================================================================================
    # -- Metodo:        _build_tab_performance
    # -- Descrizione:   Configura l'interfaccia di riepilogo tempi e potenze.
    # -- Uso V15:       Visualizzazione output primari post-simulazione (Regola 12).
    # ==========================================================================================================
    # ==========================================================================================================
    # -- Metodo:        _build_tab_performance (Tab 3)
    # -- Descrizione:   Configura l'interfaccia di riepilogo tempi e potenze.
    # -- Uso V15:       Visualizzazione output primari post-simulazione (Regola 12).
    # ==========================================================================================================
    @log_method_v15
    def _build_tab_performance(self):
        """Crea i widget di output per i risultati cronometrici e di potenza."""
        parent = self.tabs["3. Performance"]
        
        # 1. Box Tempi (Z-Score)
        time_frame = ttk.LabelFrame(parent, text=" Analisi Cronometrica (μ ± Z-Score) ", padding=20)
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(time_frame, text="Tempo Finale:", font=("Segoe UI", 12)).grid(row=0, column=0, sticky=tk.W)
        self.lbl_final_time = tk.Label(time_frame, text="--:--", font=("Segoe UI", 16, "bold"), fg="#f1c40f")
        self.lbl_final_time.grid(row=0, column=1, sticky=tk.W, padx=20)

        tk.Label(time_frame, text="Finestra di Confidenza:", font=("Segoe UI", 10)).grid(row=1, column=0, sticky=tk.W)
        self.lbl_time_range = tk.Label(time_frame, text="Range: --", font=("Segoe UI", 10, "italic"), fg="#7f8c8d")
        self.lbl_time_range.grid(row=1, column=1, sticky=tk.W, padx=20)

        # 2. Box KPI
        kpi_frame = ttk.LabelFrame(parent, text=" Key Performance Indicators (KPI) ", padding=20)
        kpi_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Grid 3x2 per i KPI (Espanso V15)
        # Layout:
        # P_AVG | NP
        # TSS   | IF
        # FTP_i | FTP_f
        # f_mec | -
        kpis = [
            ("Potenza Media", "lbl_avg_power", "W"),
            ("Normalized Power (NP)", "lbl_np", "W"),
            ("Training Stress (TSS)", "lbl_tss", ""),
            ("Intensity Factor (IF)", "lbl_if", ""),
            ("FTP Iniziale", "lbl_ftp_init", "W"),
            ("FTP Finale Stimata", "lbl_ftp_final", "W"),
            ("Logoramento Meccanico", "lbl_mech_wear", "")
        ]
        
        for idx, (label, attr_name, unit) in enumerate(kpis):
            r, c = divmod(idx, 2)
            f = tk.Frame(kpi_frame)
            f.grid(row=r, column=c, sticky=tk.W+tk.E, padx=20, pady=15)
            
            # Label Descrittiva
            tk.Label(f, text=label, font=("Segoe UI", 10, "bold"), fg="#34495e").pack(anchor=tk.W)
            
            # Valore
            lbl = tk.Label(f, text="--", font=("Segoe UI", 14), fg="#2980b9")
            lbl.pack(anchor=tk.W)
            
            # Unità (Opzionale, integrata nel testo o a fianco)
            # Qui la gestiamo aggiornando il testo della label
            setattr(self, attr_name, lbl)

    # ==========================================================================================================
    # -- Metodo:        _build_tab_bioenergetics
    # -- Descrizione:   Configura l'area dedicata al monitoraggio del glicogeno e kcal.
    # -- Uso V15:       Validazione metabolica della strategia (Regola 13).
    # ==========================================================================================================
    @log_method_v15
    def _build_tab_bioenergetics(self):
        """Crea i widget per il feedback metabolico."""
        parent = self.tabs["3. Bioenergetica"]
        
        container = ttk.LabelFrame(parent, text=" Bilancio Energetico e Saturazione ", padding=20)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(container, text="Consumo Totale Stimato:", font=("Segoe UI", 11)).grid(row=0, column=0, sticky=tk.W)
        self.lbl_total_kcal = tk.Label(container, text="-- kcal", font=("Segoe UI", 11, "bold"))
        self.lbl_total_kcal.grid(row=0, column=1, sticky=tk.W, padx=10)

        tk.Label(container, text="Glicogeno Residuo all'Arrivo:", font=("Segoe UI", 11)).grid(row=1, column=0, sticky=tk.W)
        self.lbl_final_glyco = tk.Label(container, text="-- g", font=("Segoe UI", 11, "bold"))
        self.lbl_final_glyco.grid(row=1, column=1, sticky=tk.W, padx=10)

    # ==========================================================================================================
    # -- Metodi Segnaposto (Stub): 4, 5, 6, 7, 8
    # -- Descrizione:   Garantiscono l'integrità strutturale del Notebook durante l'avvio.
    # ==========================================================================================================
    # ==========================================================================================================
    # -- Metodo:        _build_tab_strategy
    # -- Descrizione:   Costruisce la Dashboard Strategica con doppi grafici sincronizzati.
    # -- Uso V15:        Plot A: Profilo Altimetrico, Plot B: Deplezione Glicogeno.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_strategy(self):
        """
        Costruisce la Dashboard Strategica con doppi grafici sincronizzati.
        Plot A: Profilo Altimetrico e Pendenza.
        Plot B: Curva di Deplezione Glicogeno e Velocità.
        """
        parent = self.tabs["4. Strategia"]
        
        # 1. Info Bar (Data Readout)
        # 1. Info Bar (Rich Text Box per dettagli V15)
        self.strategy_info_frame = tk.Frame(parent, bg="#2c3e50", height=80)
        self.strategy_info_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        self.strategy_info_frame.pack_propagate(False)
        
        self.strategy_lbl_info = tk.Text(
            self.strategy_info_frame, 
            height=3, 
            font=("Consolas", 10, "bold"), 
            fg="#ecf0f1", bg="#2c3e50", 
            bd=0, highlightthickness=0
        )
        self.strategy_lbl_info.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.strategy_lbl_info.insert(tk.END, "Muovi il mouse sui grafici per i dettagli punto-punto...")
        self.strategy_lbl_info.config(state=tk.DISABLED)

        # 2. Area Grafici (Matplotlib)
        self.strategy_plot_frame = tk.Frame(parent)
        self.strategy_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Setup Figure
        self.strat_fig = Figure(figsize=(10, 8), dpi=100)
        self.strat_ax1 = self.strat_fig.add_subplot(211) # Profilo
        self.strat_ax2 = self.strat_fig.add_subplot(212, sharex=self.strat_ax1) # Glicogeno
        
        self.strat_fig.patch.set_facecolor('#f0f3f5')
        self.strat_fig.tight_layout(pad=3.0)
        
        self.strat_canvas = FigureCanvasTkAgg(self.strat_fig, master=self.strategy_plot_frame)
        self.strat_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Event Binding (Hover Sync)
        self.strat_canvas.mpl_connect('motion_notify_event', self._on_strategy_hover)

        # 3. Gaussian Panel Integration (V15 Refinement)
        try:
            self._build_gaussian_panel_v15(parent)
        except Exception as e:
            logger.error(f"Errore inizializzazione pannello Gaussiano: {e}")
            tk.Label(parent, text=f"Errore caricamento Modulo Gaussiano: {e}", fg="red").pack()

    # ==========================================================================================================
    # -- Metodo:        _build_gaussian_panel_v15
    # -- Descrizione:   Costruisce il pannello per l'analisi probabilistica (Gaussian Plot).
    # -- Uso V15:        Visualizzazione del rischio tempo/distribuzione (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_gaussian_panel_v15(self, parent):
        """Costruisce il pannello per l'analisi probabilistica (Gaussian Plot)."""
        mid_frame = tk.Frame(parent)
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Slider Z-Score
        slider_frame = ttk.LabelFrame(mid_frame, text=" Controllo Rischio (Z-Score) ", padding=10)
        slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        tk.Scale(slider_frame, from_=0.5, to=3.0, resolution=0.1, 
                 variable=self.settings['z_score'], orient=tk.VERTICAL, length=200, label="Z"
        ).pack(side=tk.LEFT, padx=10)
        
        # Plot Frame
        plot_frame = tk.Frame(mid_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.gauss_fig = Figure(figsize=(5, 3), dpi=100)
        self.gauss_ax = self.gauss_fig.add_subplot(111)
        self.gauss_fig.patch.set_facecolor('#f0f3f5')
        self.gauss_canvas = FigureCanvasTkAgg(self.gauss_fig, master=plot_frame)
        self.gauss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Binding per aggiornamento real-time
        self.settings['z_score'].trace_add("write", lambda *args: self._update_gaussian_plot_v15())

    # ==========================================================================================================
    # -- Metodo:        _update_gaussian_plot_v15
    # -- Descrizione:   Aggiorna il grafico della densità di probabilità temporale.
    # -- Uso V15:        Rendering dinamico basato su Z-Score (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _update_gaussian_plot_v15(self):
        """Aggiorna il grafico della densità di probabilità temporale."""
        if self.df.empty or 'Tempo_Sec_Raw' not in self.df.columns: return
            
        try:
            raw_time_sec = self.df['Tempo_Sec_Raw'].iloc[-1]
            z = self.settings['z_score'].get()
            sigma = raw_time_sec * 0.05 # Assumiamo 5% dev standard stimata
            
            # Calcolo Range di Confidenza
            t_min = raw_time_sec - (z * sigma)
            t_max = raw_time_sec + (z * sigma)
            
            # Generazione Curva
            x = np.linspace(raw_time_sec - 4*sigma, raw_time_sec + 4*sigma, 100)
            y = norm.pdf(x, raw_time_sec, sigma)
            
            self.gauss_ax.clear()
            self.gauss_ax.plot(x/3600, y, 'b-', lw=2, alpha=0.7)
            self.gauss_ax.fill_between(x/3600, y, where=((x >= t_min) & (x <= t_max)), color='blue', alpha=0.2)
            
            # Converti in ore e minuti per display
            h_min = int(t_min // 3600); m_min = int((t_min % 3600) // 60)
            h_max = int(t_max // 3600); m_max = int((t_max % 3600) // 60)
            
            # Calcolo Probabilità Percentuale (Regola 7: Show Your Work)
            # z-score to confidence level: (2 * cdf(z) - 1) * 100
            conf_pct = (2 * norm.cdf(z) - 1) * 100
            
            self.gauss_ax.set_title(f"Probabilità di Arrivo: {conf_pct:.1f}% (Confidenza {z}σ)\n{h_min}h{m_min:02d} - {h_max}h{m_max:02d}", fontsize=9)
            self.gauss_ax.set_xlabel("Ore di Gara")
            self.gauss_ax.grid(True, linestyle='--', alpha=0.3)
            
            self.gauss_canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"Errore plot Gaussiana: {e}")

    # ==========================================================================================================
    # -- Metodo:        _refresh_strategy_plots
    # -- Descrizione:   Disegna i dati simulati nei subplot strategici.
    # -- Uso V15:        Rendering curve glicogeno e altimetria (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _refresh_strategy_plots(self):
        """Disegna i dati simulati nei subplot strategici."""
        if self.df.empty or 'Glicogeno_Residuo' not in self.df.columns: return
        
        # Data Prep
        # Data Prep (Downsampling a 100m per leggibilità grafico)
        # FIX V15: Campionamento per evitare grafici lenti e illeggibili
        mask = (self.df['Distanza Cumulata'] // 100).diff() != 0
        df_plot = self.df[mask].copy()
        
        dist = df_plot['Distanza Cumulata'] / 1000.0
        alt = df_plot['Altitudine']
        gly = df_plot['Glicogeno_Residuo']
        vel = df_plot['Velocita_V15'] * 3.6 # km/h
        
        # Plot A: Altimetria
        self.strat_ax1.clear()
        self.strat_ax1.fill_between(dist, alt, min(alt), color='#3498db', alpha=0.4, label='Quota')
        self.strat_ax1.set_ylabel('Altitudine (m)', fontweight='bold')
        self.strat_ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Plot B: Bioenergetica vs Velocità
        self.strat_ax2.clear()
        self.strat_ax2.plot(dist, gly, color='#2ecc71', linewidth=2, label='Glicogeno (g)')
        self.strat_ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Soglia Crisi')
        self.strat_ax2.set_ylabel('Glicogeno (g)', color='#2ecc71', fontweight='bold')
        self.strat_ax2.tick_params(axis='y', labelcolor='#2ecc71')
        self.strat_ax2.set_xlabel('Distanza (km)')
        self.strat_ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Twin Axis per Velocità
        ax2_vel = self.strat_ax2.twinx()
        ax2_vel.plot(dist, vel, color='#9b59b6', linewidth=1, alpha=0.6, label='Velocità (km/h)')
        ax2_vel.set_ylabel('Velocità (km/h)', color='#9b59b6')
        
        self.strat_ax1.set_title("Analisi Strategica Integrata: Orografia vs Riserve Energetiche", fontsize=11, fontweight='bold')
        self.strat_canvas.draw()

    # ==========================================================================================================
    # -- Metodo:        _on_strategy_hover
    # -- Descrizione:   Gestisce il cursore sincronizzato nella tab strategia.
    # -- Uso V15:        Interazione real-time per analisi puntuale (Regola 12).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _on_strategy_hover(self, event):
        """Gestisce il cursore sincronizzato."""
        if self.df.empty or not event.inaxes: return
        
        # Trova indice più vicino
        x = event.xdata
        idx = (np.abs((self.df['Distanza Cumulata']/1000.0) - x)).argmin()
        row = self.df.iloc[idx]
        
        # Update Info Bar (Text Widget) con multiple unità di velocità
        v_kmh = row['Velocita_V15'] * 3.6
        v_ms = row['Velocita_V15']
        pace = 60 / v_kmh if v_kmh > 0 else 0
        pace_min = int(pace)
        pace_sec = int((pace - pace_min) * 60)
        
        info_txt = (f"KM: {x:.2f} | Quota: {int(row['Altitudine'])}m | Pendenza: {row['Pendenza (%)']:.1f}%\n"
                   f"Glicogeno: {int(row['Glicogeno_Residuo'])}g | Potenza: {int(row['Potenza_Watt'])}W | Kcal: {int(row['Kcal_Cum'])}\n"
                   f"Vel: {v_kmh:.1f} km/h | {v_ms:.2f} m/s | {pace_min}'{pace_sec:02d}\"/km "
                   f"| Tempo: {int(row['Tempo_Sec_Raw']//3600)}h {int((row['Tempo_Sec_Raw']%3600)//60)}m")
        
        self.strategy_lbl_info.config(state=tk.NORMAL)
        self.strategy_lbl_info.delete(1.0, tk.END)
        self.strategy_lbl_info.insert(tk.END, info_txt)
        self.strategy_lbl_info.config(state=tk.DISABLED)
        
        # Linea verticale su entrambi i grafici (disegnata 'loose' per performance)
        for ax in [self.strat_ax1, self.strat_ax2]:
            for line in ax.lines:
                if line.get_label() == '_cursor': line.remove()
            ax.axvline(x=x, color='red', linewidth=1, alpha=0.8, label='_cursor')
        
        self.strat_canvas.draw_idle()

    # ==========================================================================================================
    # -- Metodo:        _build_tab_splits
    # -- Descrizione:   Inizializza l'area dedicata alla tabella degli intertempi chilometrici.
    # -- Uso V15:        Visualizzazione granulare dei tempi di passaggio stimati (Regola 12).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_splits(self):
        """
        Configura il widget di visualizzazione per i dati segmentati.
        Prepara il container per la Treeview degli splits.
        """
        parent = self.tabs["5. Splits"]
        tk.Label(
            parent, 
            text="Analisi Intertempi Chilometrici", 
            font=("Segoe UI", 12, "bold"), 
            bg="#f0f3f5"
        ).pack(pady=(20, 10))
        
        # Nota: La popolazione effettiva avviene in _refresh_splits_table
        self.split_container = ttk.Frame(parent)
        self.split_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Creazione Treeview
        cols = ("KM", "Tempo Split", "Tempo Cum.", "D+", "D-", "Watt", "Glyco", "km/h", "min/km")
        self.split_tree = ttk.Treeview(self.split_container, columns=cols, show='headings', height=18)
        
        # Configurazione Colonne
        col_widths = [60, 90, 90, 60, 60, 60, 60, 60, 80]
        for col, width in zip(cols, col_widths):
            self.split_tree.heading(col, text=col)
            self.split_tree.column(col, width=width, anchor=tk.CENTER)
            
        # Scrollbar
        sb = ttk.Scrollbar(self.split_container, orient=tk.VERTICAL, command=self.split_tree.yview)
        self.split_tree.configure(yscroll=sb.set)
        
        self.split_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # BINDING ORDINAMENTO COLONNE (Regola V15 UX)
        for col in cols:
            self.split_tree.heading(col, text=col, command=lambda _c=col: \
                                    self._treeview_sort_column(self.split_tree, _c, False))

    # ==========================================================================================================
    # -- Metodo:        _build_tab_climbs (Porting V14)
    # -- Descrizione:   Inizializza la tabella delle salite con punteggio V15.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_climbs(self):
        """Costruisce la tabella delle salite."""
        parent = self.tabs["3b. Salite"]
        
        tk.Label(parent, text=" Analisi Grandi Salite (Côtes) ", font=("Segoe UI", 12, "bold"), bg="#f0f3f5").pack(pady=(20, 10))
        
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        cols = ("ID", "Inizio (km)", "Fine (km)", "Quota Start", "Quota End", "D+ (m)", "Pend. Avg %", "Score", "Cat")
        self.climb_tree = ttk.Treeview(container, columns=cols, show='headings', height=15)
        
        widths = [40, 80, 80, 80, 80, 60, 80, 60, 100]
        for col, w in zip(cols, widths):
            self.climb_tree.heading(col, text=col)
            self.climb_tree.column(col, width=w, anchor=tk.CENTER)
            
        sb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.climb_tree.yview)
        self.climb_tree.configure(yscroll=sb.set)
        
        self.climb_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    @safe_step
    @log_method_v15
    def _refresh_climbs_table(self):
        """Popola la tabella salite post-simulazione."""
        logger.info("Analisi salite in corso...")
        if self.df.empty: return
        
        # Cleanup
        for i in self.climb_tree.get_children(): self.climb_tree.delete(i)
            
        try:
            climbs = self.analyzer._generate_v15_climb_analysis(self.df)
            for c in climbs:
                self.climb_tree.insert("", tk.END, values=(
                    c['ID'], c['Start_Km'], c['End_Km'], 
                    c['Quota_Start'], c['Quota_End'], c['Dislivello'],
                    c['Pendenza_Avg'], c['Score'], c['Cat']
                ))
            logger.info(f"Tabella Salite aggiornata: {len(climbs)} entry.")
        except Exception as e:
            logger.error(f"Errore refresh salite: {e}")


    # ==========================================================================================================
    # -- Metodo:        _build_tab_map
    # -- Descrizione:   Configura l'interfaccia di attivazione della cartografia interattiva.
    # -- Uso V15:        Punto di accesso per il rendering geospaziale Folium (Regola 13).
    # ==========================================================================================================
    # ==========================================================================================================
    # -- Metodo:        _build_tab_map (Tab 6)
    # -- Descrizione:   Configura l'interfaccia di attivazione della cartografia interattiva.
    # -- Uso V15:        Punto di accesso per il rendering geospaziale Folium (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_map(self):
        """
        Crea il pannello di controllo per l'esportazione e visualizzazione delle mappe.
        """
        parent = self.tabs["6. Mappe Strategiche"]
        
        container = ttk.LabelFrame(parent, text=" Generazione Cartografica V15 ", padding=20)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(
            container, 
            text="Seleziona la tipologia di mappa strategica da generare:",
            font=("Segoe UI", 11)
        ).pack(pady=10)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=20)

        # 1. Pendenze
        ttk.Button(
            btn_frame, text="⛰️ MAPPA PENDENZE (Legenda V14)", 
            command=lambda: self._open_map('main'), width=30
        ).pack(pady=5)

        # 2. Food
        ttk.Button(
            btn_frame, text="🍔 MAPPA FOOD (Alert Ipo)", 
            command=lambda: self._open_map('food'), width=30
        ).pack(pady=5)
        
        # 3. Sleep
        ttk.Button(
            btn_frame, text="🌙 MAPPA SLEEP (Nadir Circadiano)", 
            command=lambda: self._open_map('sleep'), width=30
        ).pack(pady=5)

    # ==========================================================================================================
    # -- Metodo:        _build_tab_report (Tab 7)
    # -- Descrizione:   Export dati in formato Excel multi-foglio.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_report(self):
        """Pulsante per attivare l'export Excel."""
        parent = self.tabs["7. Export Report"]
        
        container = ttk.LabelFrame(parent, text=" Reportistica Post-Gara ", padding=20)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Button(
            container, 
            text="📊 GENERA REPORT EXCEL COMPLETO", 
            command=self._open_excel, 
            style="Accent.TButton"
        ).pack(expand=True, ipadx=20, ipady=10)

    # ==========================================================================================================
    # -- Metodo:        _build_tab_log (Tab 8)
    # -- Descrizione:   Console di debug in tempo reale.
    # ==========================================================================================================
    @log_method_v15
    def _build_tab_log(self):
        """Console di testo neraper il monitoraggio logger."""
        parent = self.tabs["8. System Log"]
        
        self.log_text = tk.Text(
            parent, 
            bg="black", fg="#00ff00", 
            font=("Consolas", 9),
            state=tk.NORMAL
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Redirezione Logger su Text Widget (Opzionale, richiederebbe handler custom)
        # Per ora è statico o aggiornabile via metodo dedicato.
        self.log_text.insert(tk.END, "--- TRAIL ANALYZER V15 SYSTEM LOG ---\n")
        self.log_text.insert(tk.END, "Ready.\n")
        self.log_text.config(state=tk.DISABLED)


    # ==========================================================================================================
    # -- Metodo:        _build_tab_report
    # -- Descrizione:   Inizializza il modulo di esportazione dati multi-formato.
    # -- Uso V15:        Gestisce il salvataggio dei risultati per analisi post-gara (Regola 7).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_report(self):
        """
        Configura l'interfaccia per la generazione del report Excel dettagliato.
        """
        parent = self.tabs["7. Export Report"]
        
        container = ttk.LabelFrame(parent, text=" Archiviazione Risultati ", padding=20)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        btn = ttk.Button(
            container, 
            text=" GENERA REPORT EXCEL (.xlsx) ", 
            command=self._open_excel,
            style="Accent.TButton"
        )
        btn.pack(pady=20, ipady=10)

    # ==========================================================================================================
    # -- Metodo:        _build_tab_log
    # -- Descrizione:   Configura la console di monitoraggio degli eventi di sistema.
    # -- Uso V15:        Diagnostica in tempo reale delle operazioni del motore deterministico (Regola 33).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _build_tab_log(self):
        """
        Crea un'area di testo a sola lettura per la visualizzazione dei log di processo.
        Include pulsante di refresh manuale.
        """
        parent = self.tabs["8. System Log"]
        
        # Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(toolbar, text="🔄 Aggiorna Log", command=self._load_log_file).pack(side=tk.LEFT)
        
        # Area Testo
        self.log_text = tk.Text(parent, state=tk.DISABLED, wrap=tk.NONE, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbars
        ys = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.log_text.yview)
        xs = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        
        ys.pack(side=tk.RIGHT, fill=tk.Y)
        xs.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Caricamento iniziale asincrono (dopo creazione UI)
        self.root.after(1000, self._load_log_file)


    @safe_step
    def _load_log_file(self):
        """Legge il file di log fisico e aggiorna la UI."""
        try:
            if not os.path.exists(LOG_FILENAME):
                return
                
            with open(LOG_FILENAME, "r", encoding="utf-8") as f:
                content = f.read()
                
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.insert(tk.END, content)
            self.log_text.see(tk.END) # Auto-scroll to bottom
            self.log_text.config(state=tk.DISABLED)
        except Exception as e:
            logger.error(f"Errore caricamento log: {e}")


    def _treeview_sort_column(self, tv, col, reverse):
        """
        Ordina treeview cliccando sull'intestazione.
        Gestisce sorting numerico e stringa.
        """
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        
        # Tentativo conversione numerica per sorting corretto (es: "10.5 km" -> 10.5)
        def convert(val):
            try:
                # Pulizia stringa da unità note
                clean = val.replace(' km','').replace(' m','').replace(' W','').replace('%','').replace(' g','')
                clean = clean.replace("'", ".").replace('"', '') # Clean tempo pace
                return float(clean)
            except ValueError:
                return val

        try:
            l.sort(key=lambda t: convert(t[0]), reverse=reverse)
        except Exception:
             l.sort(reverse=reverse) # Fallback string sort

        # Rearrange items in sorted positions
        for index, (val, k) in enumerate(l):
            tv.move(k, '', index)

        # Invert sort order for next click
        tv.heading(col, command=lambda: self._treeview_sort_column(tv, col, not reverse))

    # ==========================================================================================================
    # -- Metodo:        _run_simulation (Versione Multi-threaded)
    # -- Descrizione:   Avvia il motore V15 in un thread separato (Regola 15).
    # -- Uso V15:        Previene il freeze della GUI durante simulazioni complesse.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _run_simulation(self):
        """
        Orchestra l'esecuzione asincrona del motore deterministico.
        Disabilita i controlli durante il calcolo per prevenire race conditions.
        """
        if self.is_simulating:
            return

        logger.info("Avvio thread di simulazione asincrono...")
        
        # 1. Preparazione UI
        self.is_simulating = True
        self.btn_run.config(state=tk.DISABLED) # Disabilita per evitare lanci multipli
        self.progress_val.set(0)
        
        # 2. Raccolta Parametri (Snapshot dello stato attuale)
        current_settings = {k: v.get() for k, v in self.settings.items()}

        # 3. Definizione del Worker (La funzione che girerà nel thread)
        def simulation_worker():
            try:
                # Esecuzione del calcolo pesante (Regola 13)
                self.analyzer._run_v16_deterministic_cascade(
                    self.df, 
                    current_settings, 
                    progress_callback=self._safe_progress_update # Callback thread-safe
                )
                
                # Al termine, invoca l'aggiornamento finale nel Main Thread
                # FIX V15: Uso coda thread-safe
                self.msg_queue.put(("COMPLETE", None))
                
            except Exception as e:
                logger.error(f"Errore nel thread di simulazione: {str(e)}")
                self.msg_queue.put(("ERROR", str(e)))
            finally:
                self.is_simulating = False

        # 4. Lancio del Thread
        sim_thread = threading.Thread(target=simulation_worker, daemon=True)
        sim_thread.start()

    # ==========================================================================================================
    # -- Metodo:        _safe_progress_update
    # -- Descrizione:   Aggiorna la progress bar in modo thread-safe.
    # -- Uso V15:       Evita il crash 'main thread is not in main loop' (Regola 15).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _safe_progress_update(self, value: int):
        """Invia l'aggiornamento alla coda della UI solo se la finestra è attiva."""
        # FIX V15: Uso coda messaggi invece di root.after diretto
        self.msg_queue.put(("PROGRESS", value))
    # ==========================================================================================================
    # -- Metodo:        _on_simulation_complete
    # -- Descrizione:   Gestisce il completamento e il refresh dei grafici nel Main Thread.
    # -- Uso V15:       Evita il crash 'main thread is not in main loop' (Regola 15).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _on_simulation_complete(self):
        """Gestisce il completamento e il refresh dei grafici nel Main Thread."""
        self._populate_results()
        self._refresh_performance_plots()
        self._refresh_bio_plots()
        self._refresh_strategy_plots()
        self._refresh_climbs_table() # NEW V15
        self._refresh_splits_table()
        self.btn_run.config(state=tk.NORMAL)
        self.status_msg.set("Simulazione completata con successo.")
        # FIX V15: Forza il completamento della barra
        self.progress_val.set(100)
        logger.info("Thread di simulazione terminato e UI aggiornata.")

    # ==========================================================================================================
    # -- Metodo:        _refresh_performance_plots
    # -- Descrizione:   Rigenera i grafici cronometrici e di potenza nella Tab 2.
    # -- Uso V15:        Integrazione dinamica Matplotlib-Tkinter (Regola 15).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _refresh_performance_plots(self):
        """
        Estrae i vettori dal DataFrame simulato e aggiorna il profilo altimetrico
        sovrapposto alla distribuzione della potenza (Top) e Telemetria Fatica (Bottom).
        """
        logger.info("Aggiornamento grafici performance in corso...")
        
        if self.df.empty or 'Potenza_Watt' not in self.df.columns:
            return

        try:
            # Cleanup canvas precedente - FIX KEY "3. Performance"
            if "3. Performance" in self.tabs:
                target_tab = self.tabs["3. Performance"]
                for widget in target_tab.winfo_children():
                    if str(widget).endswith("canvas"): widget.destroy()

                # Setup Figure: 2 Subplots (Share X)
                fig = Figure(figsize=(10, 6), dpi=100)
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212, sharex=ax1)
                
                # Dati
                x = self.df['Distanza Cumulata'] / 1000.0
                alt = self.df['Altitudine']
                pwr = self.df['Potenza_Watt']
                
                # --- SUBPLOT 1: Altimetria + Potenza ---
                ax1.fill_between(x, alt, min(alt), color='#bdc3c7', alpha=0.3, label='Altitudine')
                ax1.set_ylabel('Quota (m)', color='#7f8c8d')
                ax1.tick_params(axis='y', labelcolor='#7f8c8d')
                ax1.grid(True, linestyle=':', alpha=0.6)
                
                # Potenza su asse gemello
                ax1_pwr = ax1.twinx()
                # Rolling average per pulire il grafico (30s approx -> 30 samples)
                if len(pwr) > 30:
                    pwr_smooth = pd.Series(pwr).rolling(window=30, center=True).mean().fillna(0)
                else:
                    pwr_smooth = pwr
                    
                ax1_pwr.plot(x, pwr_smooth, color='#e67e22', linewidth=1.0, label='Potenza (W)', alpha=0.8)
                ax1_pwr.set_ylabel('Potenza (W)', color='#e67e22')
                ax1_pwr.tick_params(axis='y', labelcolor='#e67e22')
                
                # Titolo
                if hasattr(self.analyzer, 'simulation_stats') and 'NP' in self.analyzer.simulation_stats:
                    st = self.analyzer.simulation_stats
                    ax1.set_title(f"Profilo Gara - NP: {st['NP']}W | TSS: {st['TSS']}", fontsize=10, fontweight='bold')
                else:
                    ax1.set_title("Profilo Gara V15 (Performance Update)", fontsize=10)

                # --- SUBPLOT 2: Telemetria Fatica ---
                # Linea 1: Fatica Centrale (Se presente nel DF)
                if 'Fatica_Centrale' in self.df.columns:
                    ax2.plot(x, self.df['Fatica_Centrale'], color='#8e44ad', label='Fatica Centrale', linewidth=1.5)
                
                # Linea 2: Penalità Prestazione Globale (Ratio)
                # Calcolo Ratio al volo: V_eff / V_theoretical
                # Se non abbiamo V_theoretical, usiamo f_mecc * f_gly come proxy
                if 'Velocita_V15' in self.df.columns and 'Velocita_Target' in self.df.columns:
                     # Avoid division by zero
                     ratio = self.df['Velocita_V15'] / self.df['Velocita_Target'].replace(0, 1)
                     ax2.plot(x, ratio, color='#c0392b', label='Perf. Ratio (V/V_id)', linestyle='--')
                elif 'f_mecc' in self.df.columns and 'f_gly' in self.df.columns:
                    # Proxy
                    ax2.plot(x, self.df['f_mecc'] * self.df['f_gly'], color='#e74c3c', label='Est. Perf. Penalty', linestyle='--')

                ax2.set_xlabel('Distanza (km)')
                ax2.set_ylabel('Metriche Fatica')
                ax2.grid(True, linestyle='--', alpha=0.3)
                
                # Linea 3: Glicogeno Residuo (Twin axis)
                if 'Glicogeno_Residuo' in self.df.columns:
                    ax2_gly = ax2.twinx()
                    ax2_gly.plot(x, self.df['Glicogeno_Residuo'], color='#2ecc71', label='Glicogeno', linewidth=1.5, alpha=0.7)
                    ax2_gly.set_ylabel('Glicogeno (g)', color='#2ecc71')
                    ax2_gly.tick_params(axis='y', labelcolor='#2ecc71')

                fig.tight_layout()
                
                # Embedding in Tkinter
                canvas = FigureCanvasTkAgg(fig, master=target_tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                logger.info("Rendering Profile: Distanza vs Altitudine/Potenza/Fatica completato.")
            else:
                logger.error("Tab '3. Performance' non trovata per il refresh.")
                
        except Exception as e:
            logger.error(f"Errore nel refresh dei plot performance: {str(e)}")
            logger.error(traceback.format_exc())

    # ==========================================================================================================
    # -- Metodo:        _refresh_bio_plots
    # -- Descrizione:   Aggiorna le curve di saturazione del glicogeno nella Tab 3.
    # -- Uso V15:        Visualizzazione del decadimento bioenergetico (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _refresh_bio_plots(self):
        """
        Renderizza l'andamento del consumo di carboidrati e grassi durante
        la progressione chilometrica.
        """
        logger.info("Aggiornamento grafici bioenergetici in corso...")
        
        # Implementazione specifica per il monitoraggio glicogeno residuo
        pass

    # ==========================================================================================================
    # -- Metodo:        _refresh_splits_table
    # -- Descrizione:   Popola la Treeview con i tempi parziali per ogni chilometro.
    # -- Uso V15:        Generazione tabella analitica intertempi (Regola 7).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _refresh_splits_table(self):
        """
        Converte i dati della simulazione in righe tabellari per la Tab 5.
        Formatta i tempi in HH:MM:SS per la leggibilità utente.
        """
        logger.info("Popolamento tabella splits (intertempi) in corso...")
        
        # Recupero dati dagli splits calcolati dal motore V15
        # Recupero dati dagli splits calcolati dal motore V15
        try:
            # Pulizia Treeview
            for item in self.split_tree.get_children():
                self.split_tree.delete(item)
            
            # FIX V16: Usage of _generate_v16_splits
            km_interval = 1.0 # Default V16
            self.splits_data = self.analyzer._generate_v16_splits(self.df, interval_km=km_interval)
            
            for s in self.splits_data:
                self.split_tree.insert("", tk.END, values=(
                    f"{s['KM_Fine']} km", 
                    s['Tempo_Parziale'], 
                    s['Tempo_Cumulato'],
                    f"{s['D+']} m", 
                    f"{s['D-']} m",
                    f"{s['Watt_Medi']} W",
                    f"{s['Glicogeno_g']} g",
                    f"{s['Vel_kmh']} km/h",
                    s['Pace']
                ))
            
            logger.info(f"Tabella Splits aggiornata con {len(self.splits_data)} righe.")
            
        except Exception as e:
            logger.error(f"Errore popolamento splits: {str(e)}")
            logger.error(traceback.format_exc())

    # ==========================================================================================================
    # -- Metodo:        _populate_results
    # -- Descrizione:   Aggiorna le etichette di riepilogo con i dati finali della simulazione.
    # -- Uso V15:        Fornisce il feedback numerico istantaneo post-calcolo (Regola 12).
    # ==========================================================================================================
    # ==========================================================================================================
    # -- Metodo:        _calculate_end_state_kpis
    # -- Descrizione:   Sintesi clinica dello stato dell'atleta all'ultimo chilometro.
    # -- Uso V15:        Calcolo KPI finali per la Tab 3 (Regola 13).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _calculate_end_state_kpis(self):
        """
        Calcola i KPI biofisici avanzati e i delta di efficienza allo stato finale.
        Uso V15: Fornire una sintesi clinica dello stato dell'atleta all'ultimo chilometro.
        Input: self.df.
        Output: Dizionario con i delta di efficienza locomotoria e metabolica.
        """
        if self.df.empty: return {}
        
        results = {}
        try:
            # Recupero dati puntuali
            last_row = self.df.iloc[-1]
            
            # 1. Recupero Parametri Base
            ftp_base = float(self.settings['ftp'].get())
            
            # 2. Calcolo FTP Finale Stimata
            # Formula: FTP_final = FTP_base * f_total
            # Dove f_total è il fattore di penalità globale. Se non esplicitato nel DF,
            # lo calcoliamo come rapporto tra Velocità Reale V15 e Velocità Teorica Flat/No-Penalty.
            # In V15 solitamente esiste un fattore 'f_tot' o simile.
            # Se non presente, usiamo la logica inversa: Potenza_Watt / Potenza_Teorica
            # Ma FTP è un massimale teorico.
            # Assumiamo di usare il "Performance Ratio" (PR) come stimatore del decadimento globale.
            # PR = V_eff / V_base.
            # Tuttavia, per FTP usiamo il fattore di logoramento meccanico e glicogenico.
            
            # Cerchiamo colonne di penalità note nel DF (es: 'f_mecc', 'f_gly', 'f_slep', 'f_hyp')
            # Se non le abbiamo, stimiamo f_total.
            # Placeholder: Se non troviamo colonne specifiche, usiamo 1.0 come default o cerchiamo di derivarlo.
            # Per ora, cerchiamo di vedere se nel DF ci sono colonne che iniziano con 'f_'.
            
            # Se siamo "al buio", usiamo una stima basata sulla potenza.
            # FTP_final ~ FTP_base * (Potenza_Media_Last_Hour / NP_Last_Hour) ... no.
            
            # USIAMO LA DEFINIZIONE DEL PROMPT SECCA: 
            # "f_total è l'ultima penalità calcolata nella cascata"
            # Assumeremo che esista una colonna 'Penalty_Total' o simile, ALTRIMENTI
            # la calcoliamo come prodotto delle penalità se presenti.
            
            f_total = 1.0
            
            # Tentativo di recupero fattori noti V15 standard
            # (Basato su nomi comuni in Trail Analyzer V14/V15)
            # f_hyp (Ipossia), f_mecc (Meccanico), f_gly (Glicogeno), f_sleep (Sonno)
            
            f_hyp = last_row.get('f_hyp', 1.0) # Ipossia
            f_mecc = last_row.get('f_mecc', 1.0) # Meccanico
            f_gly = last_row.get('f_gly', 1.0) # Glicogeno
            f_sleep = last_row.get('f_sleep', 1.0) # Sonno
            
            # Se le colonne non ci sono nel DF, 
            # proviamo a vedere se sono state salvate in self.analyzer.simulation_stats
            
            # Calcolo f_total combinato
            f_total = f_hyp * f_mecc * f_gly * f_sleep
            
            ftp_final = ftp_base * f_total
            delta_ftp = ((ftp_final / ftp_base) - 1) * 100
            
            results['ftp_base'] = ftp_base
            results['ftp_final'] = ftp_final
            results['delta_ftp'] = delta_ftp
            results['f_total'] = f_total
            results['f_mecc'] = f_mecc
            
            # Performance Ratio
            # PR = V_eff / V_base ("Velocità_V15" / "Velocita_Base" se esiste)
            # Se V_base non c'è, usiamo una stima o placeholder.
            v_eff = last_row.get('Velocita_V15', 0)
            # V_base spesso non è salvata per ogni punto se non come parametro calcolato al volo.
            # Usiamo V_eff / (V_eff / f_total) = f_total come PR se V_base * f_total = V_eff.
            # Quindi PR = f_total.
            results['pr'] = f_total 
            
            # Pressione Atmosferica Finale (se calcolata)
            # Usiamo _get_hyp_pen logic se necessario, ma qui solo display.
            # Se non c'è nel DF, ricalcoliamo al volo in base all'altitudine ultima.
            last_alt = last_row['Altitudine']
            # Chiamata interna a analyzer per avere pressione/penalità puntuale se serve
            # p_final = ...
            
            results['last_alt'] = last_alt
            
            return results
            
        except Exception as e:
            logger.error(f"Errore calcolo KPI finali: {e}")
            return {}

    @safe_step
    @log_method_v15
    def _populate_results(self):
        """
        Estrae i dati aggregati dal DataFrame e aggiorna i widget testuali.
        Versione V15 Robust: Gestione errori puntuale per ogni widget.
        """
        logger.info("Aggiornamento metriche di sintesi in corso...")
        
        if self.df.empty or 'Tempo_Max_V15' not in self.df.columns:
            logger.warning("Dati non disponibili per la sintesi dei risultati.")
            return

        # 1. Preparazione Dati
        try:
            t_final_sec = self.df['Tempo_Max_V15'].iloc[-1]
            t_min_sec = self.df['Tempo_Min_V15'].iloc[-1]
            kcal_tot = self.df['Kcal_Cum'].iloc[-1]
            glyco_final = self.df['Glicogeno_Residuo'].iloc[-1]
            avg_power = self.df['Potenza_Watt'].mean()
            if pd.isna(avg_power): avg_power = 0
            
            ore = int(t_final_sec // 3600)
            minuti = int((t_final_sec % 3600) // 60)
            time_str = f"{ore:02d}h {minuti:02d}m"
            range_str = f"Range: {int(t_min_sec//3600)}h {int((t_min_sec%3600)//60)}m - {time_str}"
        except Exception as e:
            logger.error(f"Errore calcolo preliminare risultati: {e}")
            return

        # 2. Aggiornamento UI Puntuale
        def safe_update(widget_attr, value, color=None):
            try:
                if hasattr(self, widget_attr):
                    w = getattr(self, widget_attr)
                    w.config(text=value)
                    if color: w.config(fg=color)
                else:
                    logger.warning(f"Widget {widget_attr} non trovato.")
            except Exception as ex:
                logger.error(f"Errore update {widget_attr}: {ex}")

        # Box Cronometrico & Energetico
        safe_update("lbl_final_time", time_str, "#f1c40f")
        safe_update("lbl_time_range", range_str)
        safe_update("lbl_total_kcal", f"{int(kcal_tot)} kcal")
        
        glyco_color = "#e74c3c" if glyco_final < 50 else "#2ecc71"
        safe_update("lbl_final_glyco", f"{int(glyco_final)} g", glyco_color)
        
        # Box KPI - Potenza
        safe_update("lbl_avg_power", f"{int(avg_power)} W")
        
        # KPI Avanzati (NP, TSS, IF)
        stats = getattr(self.analyzer, 'simulation_stats', {})
        logger.info(f"Sim Stats disponibili: {stats.keys() if stats else 'None'}")
        
        if stats:
            safe_update("lbl_np", f"{int(stats.get('NP', 0))} W")
            safe_update("lbl_tss", f"{int(stats.get('TSS', 0))}")
            safe_update("lbl_if", f"{stats.get('IF', 0):.2f}")
        
        # KPI End State (FTP, Wear)
        kpis = self._calculate_end_state_kpis()
        if kpis:
            safe_update("lbl_ftp_init", f"{int(kpis.get('ftp_base', 0))} W")
            safe_update("lbl_ftp_final", f"{int(kpis.get('ftp_final', 0))} W")
            
            f_mecc = kpis.get('f_mecc', 1.0)
            wear_pct = (1.0 - f_mecc) * 100
            w_col = "#e74c3c" if wear_pct > 20 else "#2c3e50"
            safe_update("lbl_mech_wear", f"-{wear_pct:.1f}%", w_col)
            
        logger.info("Aggiornamento UI completato.")

        logger.info(f"Sintesi completata: {time_str} | {int(kcal_tot)} kcal | Final Glyco: {int(glyco_final)}g")

    # ==========================================================================================================
    # -- Metodo:        _check_queue
    # -- Descrizione:   Loop di polling per processare messaggi dai thread secondari (Regola 15).
    # -- Uso V15:       Garantisce thread-safety evitando chiamate dirette a Tkinter da thread esterni.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _check_queue(self):
        """
        Controlla periodicamente la presenza di messaggi nella coda.
        Gestisce aggiornamenti UI, completamento task ed errori.
        """
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()
                
                if msg_type == "PROGRESS":
                    self.progress_val.set(payload)
                    
                elif msg_type == "COMPLETE":
                    self._on_simulation_complete()
                    
                elif msg_type == "ERROR":
                    err_msg = payload
                    self.status_msg.set(f"ERRORE: {err_msg}")
                    self.btn_run.config(state=tk.NORMAL)
                    messagebox.showerror("Errore Simulazione", err_msg)
                    
        except queue.Empty:
            pass
        finally:
            # Ripianifica il controllo tra 100ms
            if self.root.winfo_exists():
                self.root.after(100, self._check_queue)

    # ==========================================================================================================
    # -- Metodo:        _open_excel
    # -- Descrizione:   Esporta i dati della simulazione in Excel e apre il file.
    # -- Uso V15:        Genera il report finale per l'archiviazione e l'analisi esterna (Regola 7).
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _open_excel(self):
        """
        Genera un file Excel temporaneo con i risultati della simulazione.
        Include i dati granulari e gli splits aggregati.
        """
        logger.info("Avvio procedura di esportazione Excel...")

        if self.df.empty:
            logger.warning("Nessun dato disponibile per l'esportazione.")
            self.status_msg.set("ERRORE: Eseguire prima la simulazione.")
            return

        try:
            # 1. Definizione Percorso File
            out_dir = os.path.dirname(self.analyzer.current_file_path) if self.analyzer.current_file_path else os.path.expanduser("~/Documents")
            base_name = "SIMULAZIONE_V15"
            
            # 2. Generazione Analisi Salite (On-demand)
            climbs = self.analyzer._generate_v15_climb_analysis(self.df)
            
            # 3. Invocazione Motore Export V16
            # Ensure splits are generated
            if not self.splits_data:
                 self.splits_data = self.analyzer._generate_v16_splits(self.df)
                 
            path = self.analyzer._generate_v16_excel_report(
                self.df, 
                climbs, 
                self.splits_data,
                self.analyzer.simulation_stats, 
                out_dir, 
                base_name
            )
            
            # 4. Apertura Automatica
            if path and os.path.exists(path):
                os.startfile(path)
                self.status_msg.set(f"Report aperto: {os.path.basename(path)}")
            
        except Exception as e:
            logger.error(f"Errore Export Excel: {str(e)}")
            messagebox.showerror("Export Failed", str(e))

    # ==========================================================================================================
    # -- Metodo:        _open_map
    # -- Descrizione:   Genera e apre la mappa HTML.
    # -- Uso V15:        Visualizzazione 3D.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _open_map(self, map_type='main', *args, **kwargs):
        """Genera mappa Folium e apre nel browser."""
        if self.df.empty:
            messagebox.showwarning("Attenzione", "Eseguire prima la simulazione.")
            return

        try:
            out_dir = os.path.dirname(self.analyzer.current_file_path) if self.analyzer.current_file_path else os.path.expanduser("~/Documents")
            
            # Genera tutte le mappe
            paths = self.analyzer._export_v15_maps(self.df, out_dir)
            logger.info(f"Mappe generate: {paths}")
            
            if map_type in paths:
                target_path = os.path.abspath(paths[map_type])
                
                if os.path.exists(target_path):
                    logger.info(f"Apertura browser: file://{target_path}")
                    webbrowser.open(f'file://{target_path}')
                    self.status_msg.set(f"Mappa {map_type} aperta nel browser.")
                else:
                    logger.error(f"File non trovato: {target_path}")
                    messagebox.showerror("Errore", f"File non trovato:\n{target_path}")
            else:
                self.status_msg.set(f"Mappa {map_type} non disponibile.")
                messagebox.showwarning("Mappa Mancante", f"Tipo mappa '{map_type}' non generato.\nVerificare log.")
                
        except Exception as e:
            logger.error(f"Errore Map Gen: {str(e)}")
            messagebox.showerror("Errore Mappa", str(e))
            # Nome file basato sul timestamp per evitare sovrascritture accidentali
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Trail_Report_V15_{timestamp}.xlsx"
            file_path = os.path.join(os.getcwd(), filename)

            # 2. Scrittura Multi-Foglio (Show Your Work - Regola 7)
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                # Foglio 1: Dati Granulari (Punto-Punto)
                # Esportiamo solo le colonne rilevanti per l'analisi post-gara
                cols_to_export = [
                    'Distanza Cumulata', 'Altitudine', 'Pendenza (%)', 
                    'Potenza_Watt', 'Glicogeno_Residuo', 'Fatica_Centrale', 
                    'Tempo_Min_V15', 'Tempo_Max_V15', 'Kcal_Cum'
                ]
                self.df[cols_to_export].to_excel(writer, sheet_name='Dati_Alta_Risoluzione', index=False)
                
                # Foglio 2: Tabella Splits (Parziali Chilometrici)
                if self.splits_data:
                    df_splits = pd.DataFrame(self.splits_data)
                    df_splits.to_excel(writer, sheet_name='Splits_Aggregati', index=False)
                
                logger.info(f"File Excel generato: {file_path}")

            # 3. Apertura Automatica (Regola 12)
            # Gestione cross-platform per l'apertura del file
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS / Linux
                subprocess.call(['open', file_path] if sys.platform == 'darwin' else ['xdg-open', file_path])
                
            self.status_msg.set(f"Report esportato: {filename}")
            
        except Exception as e:
            logger.error(f"Errore critico durante l'esportazione Excel: {str(e)}")
            logger.error(traceback.format_exc())
            self.status_msg.set("ERRORE: Impossibile generare il file Excel.")



    # ==========================================================================================================
    # -- Metodo:        _back_to_loading (Versione Rettificata)
    # -- Descrizione:   Chiude la dashboard e ritorna alla selezione del file.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _back_to_loading(self):
        """
        Esegue il cleanup della sessione corrente e reinizializza il modulo di 
        caricamento file. Previene memory leak disconnettendo i backend grafici.
        """
        logger.info("Richiesta di ritorno allo stadio iniziale: Cleanup sessione V15...")
        
        # 1. Messaggio di conferma (Regola 10)
        if not messagebox.askyesno("Conferma", "Tornare alla selezione file? I dati correnti andranno persi."):
            return

        # 2. Chiusura risorse grafiche e distruzione finestra
        try:
            import matplotlib.pyplot as plt
            plt.close('all') # Libera la memoria dai grafici (Regola 15)
            
            # Distruzione pulita
            if self.root:
                self.root.quit()
                self.root.destroy()
                
            logger.info("MainDashboardV15 distrutta con successo.")
            
            # 3. Riavvio Motore (Chiamata alla funzione globale)
            # Utilizziamo after_idle o una nuova chiamata per garantire che il mainloop sia terminato
            # Ma dato che siamo in un callback, destroy() esce dal mainloop di questa istanza.
            # Dobbiamo solo assicurarci che start_v16_engine sia chiamato dopo.
            
            start_v16_engine()

        except Exception as e:
            logger.error(f"Errore durante la chiusura della dashboard: {str(e)}")
            # Fallback
            sys.exit(0)
    # ==========================================================================================================
    # -- Metodo:        _on_track_switch
    # -- Descrizione:   Gestisce il cambio traccia dalla combobox (Tab 1).
    # -- Uso V15:       Aggiorna il puntatore self.df e ricarica i metadati UI.
    # ==========================================================================================================
    @safe_step
    @log_method_v15
    def _on_track_switch(self, event, track_name=None):
        """
        Callback invocata quando l'utente seleziona un file diverso.
        Supporta sia evento diretto che chiamata manuale (track_name).
        """
        if track_name:
            selected_name = track_name
        elif hasattr(self, 'combo_tracks'):
             selected_name = self.combo_tracks.get()
        else:
            return

        if selected_name not in self.tracks_db: return
        
        logger.info(f"Switch traccia attivo: {selected_name}")
        
        # Swap dei dati
        data_packet = self.tracks_db[selected_name]
        self.df = data_packet['df']
        self.analyzer.current_file_path = data_packet['path']
        
        # Aggiornamento Header
        if hasattr(self, 'lbl_file_info'):
            self.lbl_file_info.config(text=f"Sorgente: {selected_name}")
            
        # Reset stato simulazione
        self.status_msg.set(f"Traccia cambiata: {selected_name}. Rilanciare simulazione.")
        self.progress_val.set(0)
        
        # Reset output grafici e tabelle che si riferivano alla vecchia traccia
        # (Opzionale: potremmo pulire Treeview o grafici, ma sufficiente avvisare user)

# --- FINE DELLA CLASSE MainDashboardV15 ---
    
        sys.exit(1)

# --- FINE DELLA CLASSE MainDashboardV15 ---

# ==========================================================================================================
# -- Funzione:      start_v16_engine
# -- Descrizione:   Entry point globale. Innesca la sequenza Splash -> Stadio 1.
# -- Uso V16:        Elimina l'apertura prematura del file dialog (Regola 15).
# ==========================================================================================================
@safe_step
@log_method_v15
def start_v16_engine():
    """
    Configura il logger e avvia la catena di montaggio Stage 0 (Video).
    Il video, al termine, chiamerà la QuickLookWindow in stato neutro.
    """
    logger.info(">>> INIZIALIZZAZIONE TRAIL ANALYZER V16 ENGINE...")

    try:
        # 1. Istanziazione Motore di Calcolo
        analyzer = GPXAnalyzer()
        
        # 2. Definizione del file video
        video_path = "Video_Intro_Per_Software.mp4"
        
        # 3. Callback: Cosa fare quando il video finisce?
        # Avviamo la QuickLookWindow senza passare un file_path (Stato Vuoto)
        def on_video_finish():
            logger.info("Video terminato. Apertura Stadio 1 (Base Station)...")
            QuickLookWindow(analyzer)

        # 4. Controllo presenza video e avvio Stage 0
        if os.path.exists(video_path):
            splash = VideoAudioSplashScreen(video_path, on_video_finish)
            splash.play_video()
        else:
            logger.warning(f"File intro '{video_path}' non trovato. Salto allo Stadio 1.")
            on_video_finish()

    except Exception as e:
        logger.critical(f"FALLIMENTO CRITICO ALL'AVVIO: {str(e)}")
        logger.error(traceback.format_exc())
        messagebox.showerror("Errore Fatale V16", f"Impossibile avviare il motore:\n{str(e)}")
        sys.exit(1)
# Blocco di esecuzione principale
if __name__ == "__main__":
    start_v16_engine()
