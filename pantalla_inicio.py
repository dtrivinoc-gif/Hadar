"""
Pantalla de inicio de Hadar: lo primero que se ve al abrir la app, antes
de la ventana grande de siempre -- igual que Word o Excel preguntan si
quieres un documento nuevo o seguir uno reciente.

Una sola pantalla con dos botones ("Nuevo Proyecto" / "Análisis Casual")
y, debajo, la lista de proyectos recientes siempre visible (con su peso
en disco y si tienen el aprendizaje adaptativo activado).

Uso típico, desde main.py:

    app = QApplication(sys.argv)
    pantalla = PantallaInicio()
    if pantalla.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)   # el usuario cerró la pantalla de inicio sin elegir nada

    ventana = HadarApp()
    if pantalla.modo == "continuar" and pantalla.ruta_proyecto:
        ventana.abrir_proyecto_desde_ruta(pantalla.ruta_proyecto)
    ventana.show()
    sys.exit(app.exec())

Esta pantalla no sabe nada de pandas, tablas ni de cómo se abre un
proyecto -- solo pregunta y devuelve una decisión (`self.modo` y
`self.ruta_proyecto`). Quien la usa (main.py) es quien de verdad abre el
proyecto con HadarApp.abrir_proyecto_desde_ruta().

self.modo puede ser:
    "nuevo"     -- proyecto nuevo, pensado para guardarse y usarse seguido
    "casual"    -- análisis rápido y desechable, nunca se guarda ni tiene ML
    "continuar" -- retomar un .hadarproy existente (self.ruta_proyecto)
"""
import os
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QWidget, QFileDialog, QSizePolicy,
)

from .config import THEMES, COLOR_ACCENT, LOGO_PNG_PATH
from .proyecto import listar_proyectos_recientes, EXTENSION, ruta_carpeta_proyectos


def _fecha_amigable(timestamp):
    """'hace 2 días', 'hoy', etc. -- sin librerías extra."""
    ahora = datetime.now()
    fecha = datetime.fromtimestamp(timestamp)
    dias = (ahora.date() - fecha.date()).days
    if dias <= 0:
        return f"hoy, {fecha.strftime('%H:%M')}"
    if dias == 1:
        return "ayer"
    if dias < 7:
        return f"hace {dias} días"
    if dias < 31:
        semanas = dias // 7
        return f"hace {semanas} semana{'s' if semanas > 1 else ''}"
    return fecha.strftime("%d-%m-%Y")


def _formato_tamano(num_bytes):
    """'300 KB', '7.4 MB', etc. -- sin librerías extra."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    kb = num_bytes / 1024
    if kb < 1024:
        return f"{kb:.0f} KB"
    return f"{kb / 1024:.1f} MB"


class _TarjetaProyecto(QFrame):
    """Una fila clickeable de la lista de proyectos recientes."""

    def __init__(self, info, colors, on_elegir):
        super().__init__()
        self.info = info
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("tarjetaProyecto")
        # OJO: QLabel hereda de QFrame en Qt -- si el selector del estilo
        # fuera "QFrame { ... }" se filtraría también a las etiquetas de
        # adentro. Por eso se apunta por nombre de objeto (#tarjetaProyecto),
        # no por tipo.
        self.setStyleSheet(
            f"#tarjetaProyecto {{ background-color: {colors['card']}; "
            f"border: 1px solid {colors['border']}; border-radius: 8px; }}"
            f"#tarjetaProyecto:hover {{ border: 1px solid {COLOR_ACCENT}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)

        columna_texto = QVBoxLayout()
        lbl_nombre = QLabel(info["nombre_archivo"])
        fuente = QFont()
        fuente.setBold(True)
        fuente.setPointSize(11)
        lbl_nombre.setFont(fuente)
        columna_texto.addWidget(lbl_nombre)

        detalle = f"{info['n_tablas']} tabla(s)"
        if info.get("tabla_activa"):
            detalle += f" · activa: {info['tabla_activa']}"
        lbl_detalle = QLabel(detalle)
        lbl_detalle.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        columna_texto.addWidget(lbl_detalle)
        layout.addLayout(columna_texto, stretch=1)

        lbl_peso = QLabel(_formato_tamano(info.get("peso_bytes", 0)))
        lbl_peso.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        lbl_peso.setMinimumWidth(60)
        layout.addWidget(lbl_peso)

        ml_activo = bool(info.get("ml_activado"))
        lbl_ml = QLabel("ML: Sí" if ml_activo else "ML: No")
        color_ml = COLOR_ACCENT if ml_activo else colors["muted"]
        lbl_ml.setStyleSheet(f"color: {color_ml}; font-size: 11px; font-weight: bold;")
        lbl_ml.setMinimumWidth(48)
        layout.addWidget(lbl_ml)

        lbl_fecha = QLabel(_fecha_amigable(info["modificado"]))
        lbl_fecha.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        layout.addWidget(lbl_fecha)

        self._on_elegir = on_elegir

    def mousePressEvent(self, event):
        self._on_elegir(self.info["ruta"])
        super().mousePressEvent(event)


class PantallaInicio(QDialog):
    """Ventana de bienvenida: Nuevo Proyecto / Análisis Casual, con la
    lista de proyectos recientes siempre visible debajo (ya no en una
    página aparte -- se ve todo de una, como en el boceto del usuario).

    Al cerrar con éxito (self.exec() == Accepted), revisa:
        self.modo           -- "nuevo", "casual" o "continuar"
        self.ruta_proyecto  -- ruta del .hadarproy elegido (solo si modo == "continuar")

    "Análisis Casual" es a propósito desechable: nunca llega a guardarse
    como proyecto, por eso nunca tiene ML ni aparece en esta lista.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.modo = None
        self.ruta_proyecto = None
        self.colors = THEMES["tactical"]

        self.setWindowTitle("Hadar Data Analytics")
        self.setMinimumSize(560, 560)
        self.setStyleSheet(
            f"QDialog {{ background-color: {self.colors['bg']}; }}"
            f"QLabel {{ color: {self.colors['text']}; }}"
        )

        layout_general = QVBoxLayout(self)
        layout_general.setContentsMargins(30, 30, 30, 30)
        layout_general.setSpacing(16)

        # --- encabezado (logo + título) ------------------------------
        encabezado = QHBoxLayout()
        if os.path.exists(LOGO_PNG_PATH):
            lbl_logo = QLabel()
            dpr = self.devicePixelRatioF()
            pixmap = QPixmap(LOGO_PNG_PATH).scaledToWidth(
                int(280 * dpr), Qt.TransformationMode.SmoothTransformation)
            pixmap.setDevicePixelRatio(dpr)
            lbl_logo.setPixmap(pixmap)
            encabezado.addWidget(lbl_logo)
        else:
            # Si falta el archivo del logo, se muestra el texto de siempre
            titulo = QLabel("HADAR ANALYTICS")
            fuente_titulo = QFont()
            fuente_titulo.setBold(True)
            fuente_titulo.setPointSize(16)
            titulo.setFont(fuente_titulo)
            encabezado.addWidget(titulo)
        encabezado.addStretch()
        layout_general.addLayout(encabezado)

        # --- los dos botones de entrada -------------------------------
        fila_botones = QHBoxLayout()
        fila_botones.setSpacing(16)
        fila_botones.addWidget(self._boton_grande(
            "Nuevo Proyecto", "Para un análisis que vas a seguir usando.",
            self._elegir_nuevo,
        ))
        fila_botones.addWidget(self._boton_grande(
            "Análisis Casual", "Revisión rápida y puntual. No se guarda.",
            self._elegir_casual,
        ))
        layout_general.addLayout(fila_botones)

        # --- lista de proyectos recientes, siempre visible -------------
        lbl_lista_titulo = QLabel("Tus proyectos recientes")
        fuente = QFont()
        fuente.setBold(True)
        fuente.setPointSize(12)
        lbl_lista_titulo.setFont(fuente)
        layout_general.addWidget(lbl_lista_titulo)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"QScrollArea {{ background-color: {self.colors['bg']}; border: none; }}")
        self._contenedor_lista = QWidget()
        self._contenedor_lista.setStyleSheet(f"background-color: {self.colors['bg']};")
        self._layout_lista = QVBoxLayout(self._contenedor_lista)
        self._layout_lista.setSpacing(6)
        self._layout_lista.addStretch()
        scroll.setWidget(self._contenedor_lista)
        layout_general.addWidget(scroll, stretch=1)

        btn_buscar = QPushButton("Buscar otro proyecto...")
        btn_buscar.clicked.connect(self._buscar_otro_proyecto)
        layout_general.addWidget(btn_buscar)

        self._refrescar_lista()

    def _boton_grande(self, titulo, subtitulo, on_click):
        boton = QPushButton()
        boton.setCursor(Qt.CursorShape.PointingHandCursor)
        boton.setMinimumHeight(90)
        boton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        boton.setStyleSheet(
            f"QPushButton {{ background-color: {self.colors['card']}; "
            f"border: 1px solid {self.colors['border']}; border-radius: 10px; text-align: left; "
            f"padding: 14px; }}"
            f"QPushButton:hover {{ border: 1px solid {COLOR_ACCENT}; }}"
        )
        contenido = QVBoxLayout(boton)
        lbl_t = QLabel(titulo)
        fuente = QFont()
        fuente.setBold(True)
        fuente.setPointSize(13)
        lbl_t.setFont(fuente)
        lbl_t.setStyleSheet(f"color: {COLOR_ACCENT};")
        contenido.addWidget(lbl_t)
        lbl_s = QLabel(subtitulo)
        lbl_s.setWordWrap(True)
        lbl_s.setStyleSheet(f"color: {self.colors['muted']};")
        contenido.addWidget(lbl_s)
        contenido.addStretch()
        boton.clicked.connect(on_click)
        return boton

    def _elegir_nuevo(self):
        self.modo = "nuevo"
        self.accept()

    def _elegir_casual(self):
        self.modo = "casual"
        self.ruta_proyecto = None
        self.accept()

    def _refrescar_lista(self):
        while self._layout_lista.count() > 1:  # deja el addStretch() final
            item = self._layout_lista.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        proyectos = listar_proyectos_recientes()
        if not proyectos:
            lbl_vacio = QLabel("Todavía no tienes proyectos guardados.")
            lbl_vacio.setStyleSheet(f"color: {self.colors['muted']};")
            self._layout_lista.insertWidget(0, lbl_vacio)
            return

        for info in proyectos:
            tarjeta = _TarjetaProyecto(info, self.colors, self._elegir_continuar)
            self._layout_lista.insertWidget(self._layout_lista.count() - 1, tarjeta)

    def _elegir_continuar(self, ruta):
        self.modo = "continuar"
        self.ruta_proyecto = ruta
        self.accept()

    def _buscar_otro_proyecto(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Buscar proyecto", ruta_carpeta_proyectos(), f"Proyecto Hadar (*{EXTENSION})"
        )
        if path:
            self._elegir_continuar(path)