"""
Configuración global de Hadar Data Analytics: rutas, colores/temas y
constantes compartidas entre módulos. No depende de ningún otro módulo
propio de Hadar -- todos los demás pueden importar de acá sin riesgo de
importación circular.
"""
import os
import sys

import os
import re
import sys
import ast
import math
import json
import base64
import hashlib
import sqlite3
import operator
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import polars as pl
except ImportError:
    pl = None

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QThread, Signal, QObject, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QIcon, QPixmap, QPageSize, QTextCharFormat
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QTableView, QTabWidget, QScrollArea, QFrame, QLineEdit, QMessageBox,
    QFileDialog, QSizePolicy, QRadioButton, QButtonGroup, QHeaderView, QInputDialog,
    QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsTextItem, QGraphicsPixmapItem, QGraphicsItem, QMenu, QTabBar,
    QDialog, QDialogButtonBox, QTextEdit, QTextBrowser
)
from PySide6.QtPrintSupport import QPrinter

import pyqtgraph as pg

try:
    import requests
except ImportError:
    requests = None


# ----------------------------------------------------------------------------
# Paletas de colores
# ----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#000020",        # Azul profundo (secondary)
        "card": "#0A0A0A",      # Negro puro (dark)
        "text": "#FFFFFF",      # Blanco (primary)
        "border": "#1E293B",
        "muted": "#94A3B8",
    },
    "light": {
        "bg": "#FFFFFF",        # Blanco (primary)
        "card": "#F3F4F9",      # Blanco con un leve tinte del azul profundo
        "text": "#0A0A0A",      # Negro puro (dark)
        "border": "#E2E8F0",
        "muted": "#64748B",
    },
}
COLOR_ACCENT = "#6366F1"    # Índigo (accent_1) — acciones principales
COLOR_ACCENT_2 = "#06B6D4"  # Cian vibrante (accent_2) — acciones secundarias/actualizar
COLOR_ACCENT_3 = "#F59E0B"  # Ámbar dorado (accent_3) — estados activos/interruptores
COLOR_HOVER = "#4F46E5"
COLOR_DANGER = "#EF4444"

# ----------------------------------------------------------------------------
# Umbral para decidir el motor de carga: Polars para datasets masivos,
# Pandas para el resto (evita el "costo" de Polars en archivos chicos y
# mantiene el resto de la app -- filtros, tabla, gráficos -- sin cambios,
# porque siempre reciben un pandas.DataFrame al final).
# ----------------------------------------------------------------------------
FILAS_UMBRAL_MASIVO = 300_000        # filas: sobre esto, se prefiere Polars
TAMANO_UMBRAL_MASIVO_BYTES = 30 * 1024 * 1024  # 30 MB: heurística para CSV/Excel,
                                                 # donde no se puede saber el N° de
                                                 # filas sin leer el archivo completo

CHART_TYPES = ["Barras", "Líneas", "Dispersión", "Dona / Torta", "Histograma", "Mapa de Calor (Correlación)"]
DONUT_PALETTE = [
    "#6366F1", "#22D3EE", "#F472B6", "#FBBF24", "#34D399",
    "#F87171", "#A78BFA", "#60A5FA", "#FB923C", "#4ADE80",
]


def resolve_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        path = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(path):
            return path
    base_dir = os.path.dirname(
        os.path.abspath(sys.argv[0] if getattr(sys, "frozen", False) else __file__)
    )
    path = os.path.join(base_dir, relative_path)
    if os.path.exists(path):
        return path
    return os.path.abspath(relative_path)


ICON_PATH = resolve_path("logo.ico")
LOGO_PNG_PATH = resolve_path("logo.png")
