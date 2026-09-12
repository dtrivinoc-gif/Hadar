"""
linea_tiempo_ui.py — Widget de la pestaña "Línea de Tiempo" de Hadar.

100% PySide6 + PyQtGraph (sin QtWebEngine, sin librerías nuevas), mismo
lenguaje visual que el resto de la app (self.colors, COLOR_ACCENT, etc.).

Ya no existe un "Modo Intervalo (Gantt)" separado: en su lugar, cualquier
par de fechas (automático, sugerido por sugerir_pares_temporales, o
manual, punto a punto) se dibuja como una LÍNEA entre los dos puntos ya
mostrados en Modo Hito. Un clic en la línea muestra cuánto tiempo pasó
entre ambos, en lenguaje natural.

Modelo de recálculo (decidido por el usuario): al arrastrar el slider de
rango, se recalculan en vivo Métricas/Indicadores/Frecuencias/Alarmas
(son baratos, ya es lo único que hace apply_table_filter()). Las
anomalías NO se recalculan solas -- para eso hay un botón explícito
"Actualizar anomalías en este rango", porque correr
SemanticAnomalyDetector en cada pixel de arrastre puede sentirse pegado
en equipos modestos con datasets grandes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer, QPoint, Signal
from PySide6.QtGui import QColor, QPen, QCursor, QPixmap, QPainter, QAction
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QGraphicsLineItem, QDialog, QDialogButtonBox, QMessageBox, QToolButton,
    QMenu,
)

from .config import COLOR_ACCENT, COLOR_DANGER, DONUT_PALETTE
from .anomalias import detectar_columnas_fecha, SemanticAnomalyDetector
from .linea_tiempo import (
    sugerir_pares_temporales, construir_serie_hitos, construir_intervalos,
    construir_histograma, anomalias_con_fecha, tiene_componente_horario,
    minutos_desde_medianoche,
    formatear_duracion, resumen_duraciones,
)

# Debounce del slider: recalcula Métricas/Indicadores como máximo cada
# tantos ms mientras se arrastra, en vez de en cada micro-movimiento del
# mouse. Sigue sintiéndose "en vivo" pero no satura la CPU.
_DEBOUNCE_SLIDER_MS = 120

# Color neutro para conexiones manuales (no confundirlas con los colores
# por columna ni por par automático).
_COLOR_CONEXION_MANUAL = "#C9C9D6"


def _epoch_a_timestamp(valor_epoch_seg):
    return pd.Timestamp(valor_epoch_seg, unit="s")


def _timestamp_a_epoch(ts):
    return ts.value / 1e9  # ns -> s, formato que espera pg (eje en segundos)


class _MenuMultiCheck(QMenu):
    """QMenu normal, pero un clic en una opción marcable NO cierra el menú
    -- así se pueden tildar 'fecha_envio', 'fecha_entrega' y una tercera
    sin tener que reabrir el botón cada vez. Es el truco estándar de Qt
    para esto (nada de PySide6 nuevo, solo interceptar el clic)."""

    def mouseReleaseEvent(self, event):
        accion = self.activeAction()
        if accion is not None and accion.isCheckable():
            accion.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _PopupInfoFila(QWidget):
    """Reemplaza el tooltip nativo para el clic sobre un punto de hito:
    visualmente es el mismo recuadro chico y oscuro, pero no depende de que
    el mouse se quede quieto -- un QToolTip se cierra solo con el primer
    micro-movimiento o timeout del sistema operativo, que era justo la queja
    ("desaparece rápido y a veces no la muestra"). Este se abre con el clic
    y se queda hasta el próximo clic (sobre otro punto, o fuera de todos)."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 7, 9, 7)
        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setStyleSheet("color: #F0F0F0; background: transparent;")
        layout.addWidget(self._label)
        self.setStyleSheet(
            "background-color: #2B2B36; border: 1px solid #46465A; border-radius: 6px;"
        )

    def mostrar(self, texto, posicion_global):
        self._label.setText(texto)
        self.adjustSize()
        self.move(posicion_global + QPoint(16, 16))
        self.show()

    def ocultar(self):
        self.hide()


class _PopupConexion(QWidget):
    """Mismo look que _PopupInfoFila, pero para el clic sobre una LÍNEA de
    conexión: muestra la duración entre los dos puntos, y si la conexión
    es manual, un botón para eliminarla (las automáticas no se pueden
    eliminar una por una -- se destildan desde 'Unir fechas')."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 7, 9, 7)
        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setStyleSheet("color: #F0F0F0; background: transparent;")
        layout.addWidget(self._label)
        self._btn_eliminar = QPushButton("Eliminar conexión")
        self._btn_eliminar.setStyleSheet(
            "background-color: #46465A; color: #F0F0F0; border: none; "
            "border-radius: 4px; padding: 4px 8px;"
        )
        layout.addWidget(self._btn_eliminar)
        self.setStyleSheet(
            "background-color: #2B2B36; border: 1px solid #46465A; border-radius: 6px;"
        )

    def mostrar(self, texto, posicion_global, on_eliminar=None):
        self._label.setText(texto)
        try:
            self._btn_eliminar.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        if on_eliminar is not None:
            self._btn_eliminar.setVisible(True)
            self._btn_eliminar.clicked.connect(on_eliminar)
        else:
            self._btn_eliminar.setVisible(False)
        self.adjustSize()
        self.move(posicion_global + QPoint(16, 16))
        self.show()

    def ocultar(self):
        self.hide()


class _TarjetaAnomalia(QDialog):
    """Diálogo simple con el detalle de una anomalía al hacer clic en su
    marcador, y el botón para mandarla al lienzo de Reporte."""

    def __init__(self, anomalia, colors, on_enviar_reporte, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detalle de la anomalía")
        self.resize(420, 260)
        layout = QVBoxLayout(self)

        descripcion = QLabel(anomalia.get("descripcion", ""))
        descripcion.setWordWrap(True)
        descripcion.setObjectName("title")
        layout.addWidget(descripcion)

        gravedad = QLabel(f"Gravedad: {anomalia.get('gravedad', 'media')}")
        gravedad.setObjectName("muted")
        layout.addWidget(gravedad)

        frase_memoria = anomalia.get("frase_memoria")
        if frase_memoria:
            lbl = QLabel(frase_memoria)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        frase_impacto = anomalia.get("frase_impacto_acumulado")
        if frase_impacto:
            lbl2 = QLabel(frase_impacto)
            lbl2.setWordWrap(True)
            layout.addWidget(lbl2)

        layout.addStretch()

        botones = QHBoxLayout()
        btn_reporte = QPushButton("Enviar a Reporte")
        btn_reporte.setObjectName("accentButton")
        btn_reporte.clicked.connect(lambda: (on_enviar_reporte(anomalia), self.accept()))
        botones.addWidget(btn_reporte)
        botones.addStretch()
        cerrar = QDialogButtonBox(QDialogButtonBox.Close)
        cerrar.rejected.connect(self.reject)
        cerrar.clicked.connect(self.reject)
        botones.addWidget(cerrar)
        layout.addLayout(botones)


class _DialogoMetricasTiempo(QDialog):
    """Promedio/mínimo/máximo de cada conexión activa (automática o el
    conjunto de las manuales). Se manda a Reporte como bloque de texto
    ('lista'), no como imagen -- son solo números, igual que ya hace
    Frecuencias."""

    def __init__(self, resumenes, on_enviar_reporte, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Métricas de tiempo")
        self.resize(440, 340)
        layout = QVBoxLayout(self)

        html_partes = []
        for etiqueta, resumen in resumenes:
            aviso_incoherentes = (
                f"<br>Atención: {resumen['n_incoherentes']} con orden invertido"
                if resumen["n_incoherentes"] else ""
            )
            bloque = QLabel(
                f"<b>{etiqueta}</b> ({resumen['n']} par(es))<br>"
                f"Promedio: {resumen['promedio']}<br>"
                f"Mínimo: {resumen['minimo']}<br>"
                f"Máximo: {resumen['maximo']}{aviso_incoherentes}"
            )
            bloque.setWordWrap(True)
            bloque.setObjectName("card")
            layout.addWidget(bloque)
            html_partes.append(
                f"<b>{etiqueta}</b> ({resumen['n']} par(es))<br>"
                f"Promedio: {resumen['promedio']} · Mínimo: {resumen['minimo']} · "
                f"Máximo: {resumen['maximo']}{aviso_incoherentes}"
            )

        layout.addStretch()

        botones = QHBoxLayout()
        btn_reporte = QPushButton("Enviar a Reporte")
        btn_reporte.setObjectName("accentButton")
        html_final = "<br><br>".join(html_partes)
        btn_reporte.clicked.connect(
            lambda: (on_enviar_reporte("Métricas de tiempo — Línea de Tiempo", html_final), self.accept())
        )
        botones.addWidget(btn_reporte)
        botones.addStretch()
        cerrar = QDialogButtonBox(QDialogButtonBox.Close)
        cerrar.rejected.connect(self.reject)
        cerrar.clicked.connect(self.reject)
        botones.addWidget(cerrar)
        layout.addLayout(botones)


class LineaTiempoPanel(QWidget):
    """
    host: la ventana principal (HadarApp). Se usa para leer host.df /
    host.filtered_df / host.colors / host.memoria / host.fingerprint_actual,
    y para avisarle el rango de tiempo activo (host.rango_tiempo) llamando
    a host.apply_table_filter() -- así Métricas/Indicadores/Frecuencias/
    Alarmas se refrescan solos, sin que este widget sepa nada de ellos.
    """

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.colors = host.colors

        self._columna_fecha = None
        self._columnas_mostrar = set()   # columnas de fecha tildadas en "Fechas a mostrar"
        self._color_por_columna = {}     # {columna: color}, estable mientras no cambien de dataset
        self._marcadores_anomalias = []  # [(scatter_item, anomalia_dict)]
        self._actualizando_desde_slider = False  # ver _aplicar_rango_activo

        # -- Conexiones entre puntos (reemplaza al extinto Modo Gantt) --
        self._pares_disponibles = {}  # {(col_inicio,col_fin): ParTemporal}
        self._pares_conectar = set()  # claves tildadas en "Unir fechas"
        self._color_por_par = {}      # {(col_inicio,col_fin): color}
        self._conexiones_manuales = []  # [{"desde":(indice,columna), "hasta":(indice,columna)}]
        self._modo_conectar_manual = False
        self._primer_punto_manual = None
        self._posicion_punto = {}     # {(indice,columna): (x,y)} -- para ubicar los extremos de una línea
        self._lineas_conexion = []    # [{"item":QGraphicsLineItem,"desde":...,"hasta":...,"duracion":...,"tipo":...,"etiqueta":...}]

        self._timer_debounce = QTimer(self)
        self._timer_debounce.setSingleShot(True)
        self._timer_debounce.setInterval(_DEBOUNCE_SLIDER_MS)
        self._timer_debounce.timeout.connect(self._aplicar_rango_activo)

        self._construir_ui()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _construir_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Filtrar por:"))
        self.combo_fecha = QComboBox()
        self.combo_fecha.setToolTip(
            "Esta es la columna que usa el slider de rango para recortar "
            "Métricas/Indicadores/Frecuencias/Alarmas."
        )
        self.combo_fecha.currentTextChanged.connect(self._on_columna_fecha_cambiada)
        toolbar.addWidget(self.combo_fecha)

        self.btn_columnas_mostrar = QToolButton()
        self.btn_columnas_mostrar.setText("Fechas a mostrar")
        self.btn_columnas_mostrar.setToolTip(
            "Si tu dataset tiene más de una fecha (ej. envío/entrega, "
            "última mantención/próxima mantención), marca las que quieras "
            "ver juntas en el gráfico -- cada una queda en un color distinto. "
            "Si hay otras tablas relacionadas con fechas propias, también "
            "aparecen acá, con un símbolo cuadrado en vez de circular."
        )
        self.btn_columnas_mostrar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_columnas_mostrar = _MenuMultiCheck(self.btn_columnas_mostrar)
        self.btn_columnas_mostrar.setMenu(self.menu_columnas_mostrar)
        toolbar.addWidget(self.btn_columnas_mostrar)

        self.btn_conectar = QToolButton()
        self.btn_conectar.setText("Unir fechas")
        self.btn_conectar.setToolTip(
            "Marca un par de fechas (ej. envío→entrega) para unir con una "
            "línea cada fila donde ambas existan. Los pares vienen de lo "
            "mismo que 'Configurar relaciones temporales' guarda (al final "
            "de este mismo menú)."
        )
        self.btn_conectar.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu_conectar = _MenuMultiCheck(self.btn_conectar)
        self.btn_conectar.setMenu(self.menu_conectar)
        toolbar.addWidget(self.btn_conectar)

        self.btn_conectar_manual = QPushButton("Conectar puntos")
        self.btn_conectar_manual.setCheckable(True)
        self.btn_conectar_manual.setToolTip(
            "Actívalo y haz clic en dos puntos cualquiera (de la misma fila "
            "o no) para unirlos con una línea propia. Vuelve a apretarlo "
            "para salir del modo."
        )
        self.btn_conectar_manual.toggled.connect(self._on_toggle_conectar_manual)
        toolbar.addWidget(self.btn_conectar_manual)

        toolbar.addStretch()

        self.btn_metricas_tiempo = QPushButton("Métricas de tiempo")
        self.btn_metricas_tiempo.clicked.connect(self._abrir_metricas_tiempo)
        toolbar.addWidget(self.btn_metricas_tiempo)

        btn_actualizar_vista = QPushButton("Actualizar")
        btn_actualizar_vista.setToolTip(
            "Actualizar la Línea de Tiempo con los filtros actuales de Datos "
            "(normalmente se actualiza sola; este botón es por si acaso)."
        )
        btn_actualizar_vista.clicked.connect(self.refrescar_vista)
        toolbar.addWidget(btn_actualizar_vista)

        layout.addLayout(toolbar)

        self.aviso = QLabel(
            "Arrastra los bordes del área sombreada para recortar un período: "
            "Métricas, Indicadores, Frecuencias y Alarmas se actualizan solas. "
            "Un clic en un punto muestra la fila; un clic en una línea "
            "muestra cuánto tiempo pasó entre sus dos puntos."
        )
        self.aviso.setObjectName("muted")
        self.aviso.setWordWrap(True)
        layout.addWidget(self.aviso)

        # Leyenda de colores por columna (solo se llena cuando hay 2+
        # columnas marcadas; ver _actualizar_leyenda).
        self.fila_leyenda = QHBoxLayout()
        layout.addLayout(self.fila_leyenda)

        # --- Histograma de densidad + slider de rango ---
        self.plot_histograma = pg.PlotWidget()
        self.plot_histograma.setMaximumHeight(140)
        self.plot_histograma.setBackground(self.colors["card"])
        self.plot_histograma.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_histograma.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_histograma.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_histograma.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_histograma.getAxis("bottom").setStyle(showValues=True)
        eje_fecha = pg.DateAxisItem(orientation="bottom")
        self.plot_histograma.setAxisItems({"bottom": eje_fecha})
        self.plot_histograma.showGrid(x=False, y=False)
        self._barra_histograma = None
        layout.addWidget(self.plot_histograma)

        self.region_rango = pg.LinearRegionItem(brush=pg.mkBrush(QColor(0, 0, 0, 0)))
        self.region_rango.setZValue(10)
        self.plot_histograma.addItem(self.region_rango)
        self.region_rango.sigRegionChanged.connect(self._on_region_cambiada)

        # --- Vista principal: puntos + líneas de conexión + anomalías ---
        self.plot_principal = pg.PlotWidget()
        self.plot_principal.setBackground(self.colors["card"])
        eje_fecha_2 = pg.DateAxisItem(orientation="bottom")
        self.plot_principal.setAxisItems({"bottom": eje_fecha_2})
        self.plot_principal.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_principal.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        self.plot_principal.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_principal.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        self.plot_principal.showGrid(x=False, y=False)
        # El eje Y se decide solo, al dibujar: "hora del día" si la columna
        # de fecha trae hora real, o "orden de aparición" si no (ver
        # _configurar_eje_y_hora/_configurar_eje_y_orden).
        layout.addWidget(self.plot_principal, stretch=1)

        self._scatters_hitos = {}  # {columna: pg.ScatterPlotItem}, se arma en cada redibujo
        self._hitos_xs_ordenados = np.array([])
        self._hitos_ys_ordenados = np.array([])
        self._hitos_indices_ordenados = []
        self._hitos_columna_ordenada = []

        self._scatter_anomalias = pg.ScatterPlotItem(
            size=14, symbol="t1", brush=pg.mkBrush(QColor(COLOR_DANGER))
        )
        self._scatter_anomalias.sigClicked.connect(self._on_click_marcador_anomalia)
        self.plot_principal.addItem(self._scatter_anomalias)

        self._popup_info_fila = _PopupInfoFila()
        self._popup_conexion = _PopupConexion()
        self.plot_principal.scene().sigMouseClicked.connect(self._on_click_principal)

        pie = QHBoxLayout()
        self.btn_actualizar_anomalias = QPushButton("Actualizar anomalías en este rango")
        self.btn_actualizar_anomalias.clicked.connect(self._actualizar_anomalias_en_rango)
        pie.addWidget(self.btn_actualizar_anomalias)
        self.lbl_estado = QLabel("")
        self.lbl_estado.setObjectName("muted")
        pie.addWidget(self.lbl_estado)
        pie.addStretch()
        layout.addLayout(pie)

    # ------------------------------------------------------------------
    # Carga / refresco de datos (llamado desde main_window al cambiar de
    # dataset o al abrir la pestaña -- ver integración en main_window.py)
    # ------------------------------------------------------------------
    def refrescar_datos(self):
        df = self.host.df
        self.combo_fecha.blockSignals(True)
        self.combo_fecha.clear()
        if df is None or df.empty:
            self.combo_fecha.blockSignals(False)
            self._limpiar_plots()
            return

        columnas_fecha = detectar_columnas_fecha(df)
        self.combo_fecha.addItems(columnas_fecha)
        self.combo_fecha.blockSignals(False)

        if not columnas_fecha:
            self.lbl_estado.setText("No se detectó ninguna columna de fecha en este dataset.")
            self._limpiar_plots()
            return

        self._columna_fecha = columnas_fecha[0]
        self.combo_fecha.setCurrentText(self._columna_fecha)

        # Registro de columnas de fecha disponibles para "Fechas a
        # mostrar": las de la tabla activa (clave (None, columna)) y, si
        # hay más tablas cargadas (base de datos relacional), también las
        # de esas otras tablas (clave (nombre_tabla, columna)) -- para
        # verlas juntas aunque no compartan fila ni esquema.
        self._fuentes_fecha = {(None, col): "local" for col in columnas_fecha}
        for nombre_tabla, tabla_df in self.host.tablas.items():
            if nombre_tabla == getattr(self.host, "nombre_tabla_activa", None):
                continue
            for col in detectar_columnas_fecha(tabla_df):
                self._fuentes_fecha[(nombre_tabla, col)] = "externa"

        # Color estable por columna (según su posición en el registro, no
        # según cuáles estén tildadas) -- así una misma columna no cambia
        # de color solo porque destildaste otra.
        self._color_por_columna = {
            clave: DONUT_PALETTE[i % len(DONUT_PALETTE)] for i, clave in enumerate(self._fuentes_fecha)
        }
        self._columnas_mostrar = {(None, self._columna_fecha)}
        self._poblar_menu_columnas_mostrar()

        # Nuevo dataset: las conexiones manuales de otro esquema no tienen
        # sentido acá (los índices de fila ni siquiera representan lo mismo).
        self._conexiones_manuales = []
        self._modo_conectar_manual = False
        self.btn_conectar_manual.setChecked(False)

        self._refrescar_menu_conexiones(columnas_fecha)
        self._redibujar_todo()

    def _poblar_menu_columnas_mostrar(self):
        self.menu_columnas_mostrar.clear()
        locales = [c for c in self._fuentes_fecha if c[0] is None]
        externas = [c for c in self._fuentes_fecha if c[0] is not None]
        for clave in locales:
            self._agregar_accion_columna_mostrar(clave, clave[1])
        if externas:
            self.menu_columnas_mostrar.addSeparator()
            for clave in externas:
                tabla, col = clave
                self._agregar_accion_columna_mostrar(clave, f"{tabla}.{col}  (otra tabla)")

    def _agregar_accion_columna_mostrar(self, clave, etiqueta):
        accion = QAction(etiqueta, self.menu_columnas_mostrar)
        accion.setCheckable(True)
        accion.setChecked(clave in self._columnas_mostrar)
        accion.toggled.connect(lambda marcado, clave=clave: self._on_columna_mostrar_toggled(clave, marcado))
        self.menu_columnas_mostrar.addAction(accion)

    def _on_columna_mostrar_toggled(self, columna, marcado):
        if marcado:
            self._columnas_mostrar.add(columna)
        else:
            self._columnas_mostrar.discard(columna)
        self._redibujar_todo()

    def _refrescar_menu_conexiones(self, columnas_fecha):
        pares_manuales = []
        if self.host.fingerprint_actual:
            pares_manuales = self.host.memoria.obtener_pares_temporales(self.host.fingerprint_actual)
        pares = sugerir_pares_temporales(columnas_fecha, pares_manuales)

        self._pares_disponibles = {(p.col_inicio, p.col_fin): p for p in pares}
        # +3 de desfase para no repetir tal cual los primeros colores que
        # ya usan las columnas sueltas en "Fechas a mostrar".
        self._color_por_par = {
            clave: DONUT_PALETTE[(i + 3) % len(DONUT_PALETTE)]
            for i, clave in enumerate(self._pares_disponibles)
        }
        self._pares_conectar = {c for c in self._pares_conectar if c in self._pares_disponibles}

        self.menu_conectar.clear()
        for clave, par in self._pares_disponibles.items():
            marca = "manual" if par.origen == "manual" else "auto"
            accion = QAction(f"{par.etiqueta}  ({marca})", self.menu_conectar)
            accion.setCheckable(True)
            accion.setChecked(clave in self._pares_conectar)
            accion.toggled.connect(lambda marcado, clave=clave: self._on_par_conectar_toggled(clave, marcado))
            self.menu_conectar.addAction(accion)
        self.menu_conectar.addSeparator()
        self.menu_conectar.addAction("Configurar relaciones temporales...", self._abrir_configuracion_pares)

    def _on_par_conectar_toggled(self, clave, marcado):
        if marcado:
            self._pares_conectar.add(clave)
        else:
            self._pares_conectar.discard(clave)
        self._redibujar_todo()

    def _abrir_configuracion_pares(self):
        # Reutiliza el diálogo que ya existe (ver Narrativa) -- esta
        # pestaña no define su propia forma de configurar pares, así los
        # pares quedan iguales se configuren desde donde se configuren.
        self.host._configurar_relaciones_temporales()
        if self.host.df is not None:
            self._refrescar_menu_conexiones(detectar_columnas_fecha(self.host.df))
            self._redibujar_todo()

    def aplicar_tema(self, colors):
        """Llamado desde apply_theme() de HadarApp cuando cambia claro/oscuro
        -- self.colors ahí se reemplaza por un diccionario nuevo cada vez,
        así que este panel necesita que se lo pasen de nuevo explícitamente."""
        self.colors = colors
        for plot in (self.plot_histograma, self.plot_principal):
            plot.setBackground(colors["card"])
            plot.getAxis("bottom").setPen(pg.mkPen(colors["muted"]))
            plot.getAxis("left").setPen(pg.mkPen(colors["muted"]))
            plot.getAxis("bottom").setTextPen(pg.mkPen(colors["text"]))
            plot.getAxis("left").setTextPen(pg.mkPen(colors["text"]))

    # ------------------------------------------------------------------
    # Redibujo
    # ------------------------------------------------------------------
    def _limpiar_plots(self):
        self._popup_info_fila.ocultar()
        self._popup_conexion.ocultar()
        for scatter in self._scatters_hitos.values():
            self.plot_principal.removeItem(scatter)
        self._scatters_hitos.clear()
        self._actualizar_leyenda([])
        self._hitos_xs_ordenados = np.array([])
        self._hitos_ys_ordenados = np.array([])
        self._hitos_indices_ordenados = []
        self._hitos_columna_ordenada = []
        self._posicion_punto = {}
        for conexion in self._lineas_conexion:
            self.plot_principal.removeItem(conexion["item"])
        self._lineas_conexion.clear()
        self._scatter_anomalias.setData([])
        self._marcadores_anomalias.clear()
        if self._barra_histograma is not None:
            self.plot_histograma.removeItem(self._barra_histograma)
            self._barra_histograma = None

    def _redibujar_todo(self):
        self._popup_info_fila.ocultar()
        self._popup_conexion.ocultar()
        # Base filtrada por TODO lo de Datos (buscador de filas puntuales,
        # filtro por columna, clic en un gráfico) menos el rango de este
        # mismo slider -- así el buscador de filas SÍ se refleja acá.
        df = self.host._construir_df_filtrado_base()
        if df is None or not self._columna_fecha:
            self._limpiar_plots()
            return

        serie_hitos = construir_serie_hitos(df, self._columna_fecha)
        self._dibujar_histograma(serie_hitos)

        self.plot_principal.enableAutoRange(axis="xy")
        self._dibujar_hitos_multi(df)
        self._redibujar_conexiones(df)

        # Bug real: self._columnas_mostrar guarda tuplas (tabla, columna)
        # desde que "Fechas a mostrar" se extendió a otras tablas -- un
        # join() directo sobre tuplas revienta con "expected str instance,
        # tuple found". Hay que pasar cada una por su etiqueta de texto
        # primero (columna sola si es local, "Tabla.columna" si es externa).
        columnas_mostradas = ", ".join(
            self._etiqueta_clave(c) for c in sorted(self._columnas_mostrar, key=lambda c: (c[0] or "", c[1]))
        ) or self._columna_fecha
        self.lbl_estado.setText(
            f"{len(serie_hitos):,} fecha(s) válida(s) en '{self._columna_fecha}' "
            f"(mostrando: {columnas_mostradas})."
        )

    def _dibujar_histograma(self, serie_hitos):
        if self._barra_histograma is not None:
            self.plot_histograma.removeItem(self._barra_histograma)
            self._barra_histograma = None
        if serie_hitos.empty:
            return
        bordes, conteos = construir_histograma(serie_hitos)
        if len(conteos) == 0:
            return
        centros = [
            (_timestamp_a_epoch(bordes[i]) + _timestamp_a_epoch(bordes[i + 1])) / 2
            for i in range(len(conteos))
        ]
        ancho = _timestamp_a_epoch(bordes[1]) - _timestamp_a_epoch(bordes[0])
        self._barra_histograma = pg.BarGraphItem(
            x=centros, height=list(conteos), width=ancho * 0.9,
            brush=COLOR_ACCENT,
        )
        self.plot_histograma.addItem(self._barra_histograma)

        t0, t1 = _timestamp_a_epoch(bordes[0]), _timestamp_a_epoch(bordes[-1])
        self.region_rango.blockSignals(True)
        self.region_rango.setBounds([t0, t1])
        if self.host.rango_tiempo is None:
            self.region_rango.setRegion([t0, t1])
        self.region_rango.blockSignals(False)

    def _dibujar_hitos_multi(self, df):
        for scatter in self._scatters_hitos.values():
            self.plot_principal.removeItem(scatter)
        self._scatters_hitos.clear()

        # "Fechas a mostrar" guarda claves (tabla_o_None, columna); los
        # pares de "Unir fechas" y las conexiones manuales de aquí para
        # abajo son siempre de la tabla activa, así que se normalizan a
        # esa misma forma. Una línea nunca puede apuntar a un punto
        # invisible, así que sus columnas también se agregan.
        columnas_deseadas = set(self._columnas_mostrar)
        for (col_a, col_b) in self._pares_conectar:
            columnas_deseadas.add((None, col_a))
            columnas_deseadas.add((None, col_b))
        for conexion in self._conexiones_manuales:
            columnas_deseadas.add(conexion["desde"][1])
            columnas_deseadas.add(conexion["hasta"][1])
        if not columnas_deseadas and self._columna_fecha:
            columnas_deseadas = {(None, self._columna_fecha)}

        # Cada clave trae su propia fuente: local = el df activo ya
        # filtrado que se está dibujando; externa = la tabla propia tal
        # cual está cargada, SIN los filtros de Datos de la tabla activa
        # (no correspondería aplicarle el filtro de una tabla a otra con
        # columnas distintas).
        series_por_clave = {}
        for clave in columnas_deseadas:
            tabla, columna = clave
            df_fuente = df if tabla is None else self.host.tablas.get(tabla)
            if df_fuente is None or columna not in df_fuente.columns:
                continue
            serie = construir_serie_hitos(df_fuente, columna)
            if not serie.empty:
                series_por_clave[clave] = serie

        self._posicion_punto = {}

        if not series_por_clave:
            self._hitos_xs_ordenados = np.array([])
            self._hitos_ys_ordenados = np.array([])
            self._hitos_indices_ordenados = []
            self._hitos_columna_ordenada = []
            self._actualizar_leyenda([])
            return

        usar_hora = any(tiene_componente_horario(s) for s in series_por_clave.values())
        if usar_hora:
            self._configurar_eje_y_hora()
        else:
            self._configurar_eje_y_orden()

        todos_x, todos_y, todos_idx, todos_col = [], [], [], []

        if usar_hora:
            # Y = hora del día, calculado por punto -- una columna sin hora
            # real simplemente cae en 00:00, que es honesto (no se sabe más
            # que eso), no hay que inventarle un ranking aparte.
            for clave, serie in series_por_clave.items():
                xs = [_timestamp_a_epoch(t) for t in serie]
                ys = minutos_desde_medianoche(serie).tolist()
                self._agregar_scatter_columna(clave, xs, ys)
                for indice, x, y in zip(serie.index, xs, ys):
                    self._posicion_punto[(indice, clave)] = (x, y)
                todos_x.extend(xs)
                todos_y.extend(ys)
                todos_idx.extend(serie.index)
                todos_col.extend([clave] * len(xs))
        else:
            # "Orden de aparición" es un ranking GLOBAL entre TODAS las
            # columnas mostradas juntas -- si cada columna tuviera su propio
            # ranking 0..n, dos columnas se pisarían en el mismo rango de Y
            # sin que signifique lo mismo.
            combinado = []
            for clave, serie in series_por_clave.items():
                for indice, ts in serie.items():
                    combinado.append((ts, indice, clave))
            combinado.sort(key=lambda t: t[0])

            puntos_por_columna = {clave: ([], []) for clave in series_por_clave}
            for orden, (ts, indice, clave) in enumerate(combinado):
                x = _timestamp_a_epoch(ts)
                puntos_por_columna[clave][0].append(x)
                puntos_por_columna[clave][1].append(orden)
                self._posicion_punto[(indice, clave)] = (x, orden)
                todos_x.append(x)
                todos_y.append(orden)
                todos_idx.append(indice)
                todos_col.append(clave)

            for clave, (xs, ys) in puntos_por_columna.items():
                self._agregar_scatter_columna(clave, xs, ys)

        # Para el clic: todos los puntos de todas las columnas, ordenados
        # juntos por X (búsqueda binaria sobre el conjunto combinado).
        orden_click = np.argsort(todos_x)
        self._hitos_xs_ordenados = np.array(todos_x)[orden_click]
        self._hitos_ys_ordenados = np.array(todos_y)[orden_click]
        self._hitos_indices_ordenados = [todos_idx[i] for i in orden_click]
        self._hitos_columna_ordenada = [todos_col[i] for i in orden_click]

        self._actualizar_leyenda(list(series_por_clave.keys()))

    def _agregar_scatter_columna(self, clave, xs, ys):
        # Círculo para columnas de la tabla activa, cuadrado para columnas
        # de otra tabla (base de datos relacional) -- el símbolo, no solo
        # el color, avisa que ese punto viene de otro origen.
        tabla, _ = clave
        simbolo = "o" if tabla is None else "s"
        color = self._color_por_columna.get(clave, COLOR_ACCENT)
        scatter = pg.ScatterPlotItem(x=xs, y=ys, size=8, symbol=simbolo, brush=pg.mkBrush(color))
        self.plot_principal.addItem(scatter)
        self._scatters_hitos[clave] = scatter

    @staticmethod
    def _etiqueta_clave(clave):
        """Texto legible de una clave (tabla_o_None, columna): solo el
        nombre si es de la tabla activa, "Tabla.columna" si es de otra."""
        tabla, col = clave
        return col if tabla is None else f"{tabla}.{col}"

    def _actualizar_leyenda(self, columnas_con_datos):
        # Leyenda propia (QLabel con un cuadradito de color) en vez de la
        # de PyQtGraph: con ScatterPlotItem sueltos, el ícono de la leyenda
        # nativa no siempre pinta el color correcto -- esto es más simple y
        # 100% predecible. Solo se muestra si hay 2+ columnas a la vista;
        # con una sola, el color no aporta nada que distinguir.
        while self.fila_leyenda.count():
            item = self.fila_leyenda.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if len(columnas_con_datos) <= 1:
            return
        for clave in columnas_con_datos:
            tabla, col = clave
            color = self._color_por_columna.get(clave, COLOR_ACCENT)
            cuadro = QLabel("■" if tabla is None else "▪")
            cuadro.setStyleSheet(f"color: {color}; font-size: 13px;")
            self.fila_leyenda.addWidget(cuadro)
            self.fila_leyenda.addWidget(QLabel(self._etiqueta_clave(clave)))
        self.fila_leyenda.addStretch()

    def _configurar_eje_y_hora(self):
        """La columna de fecha trae hora real: Y = hora del día (0-24h),
        X = fecha. Deja ver de un vistazo si algo se repite siempre a la
        misma hora (ej. fallas siempre de madrugada), algo que 'orden de
        aparición' nunca podría mostrar."""
        eje = self.plot_principal.getAxis("left")
        eje.setLabel("Hora del día")
        eje.setTicks([[(m, f"{m // 60:02d}:{m % 60:02d}") for m in range(0, 1441, 180)]])
        eje.setStyle(showValues=True)
        self.plot_principal.setYRange(0, 1440, padding=0.02)

    def _configurar_eje_y_orden(self):
        """La columna de fecha no trae hora real (todo a las 00:00:00) --
        mostrar 'hora del día' aquí mentiría (todo quedaría pegado en
        cero), así que se cae de vuelta a un eje sin unidad real, con eso
        explícito en la etiqueta."""
        eje = self.plot_principal.getAxis("left")
        eje.setLabel("Orden de aparición (sin agrupar)")
        eje.setTicks(None)
        eje.setStyle(showValues=False)
        self.plot_principal.enableAutoRange(axis="y")

    # ------------------------------------------------------------------
    # Conexiones entre puntos (automáticas por par, o manuales punto a punto)
    # ------------------------------------------------------------------
    def _redibujar_conexiones(self, df):
        for conexion in self._lineas_conexion:
            self.plot_principal.removeItem(conexion["item"])
        self._lineas_conexion.clear()

        # Automáticas: un par tildado en "Unir fechas" conecta ambas
        # columnas en cada fila donde las dos fechas son válidas. Se
        # reutiliza construir_intervalos(), que ya calculaba exactamente
        # esto (inicio/fin/duración/incoherente) para el extinto Gantt.
        for clave in self._pares_conectar:
            par = self._pares_disponibles.get(clave)
            if par is None:
                continue
            intervalos = construir_intervalos(df, par)
            color_normal = QColor(self._color_por_par.get(clave, COLOR_ACCENT))
            for indice, fila in intervalos.iterrows():
                p_ini = self._posicion_punto.get((indice, (None, par.col_inicio)))
                p_fin = self._posicion_punto.get((indice, (None, par.col_fin)))
                if p_ini is None or p_fin is None:
                    continue
                color = QColor(COLOR_DANGER) if fila["incoherente"] else color_normal
                item = self._crear_linea(p_ini, p_fin, color, Qt.PenStyle.SolidLine)
                self._lineas_conexion.append({
                    "item": item, "desde": (indice, par.col_inicio), "hasta": (indice, par.col_fin),
                    "duracion": fila["duracion"], "tipo": "auto", "etiqueta": par.etiqueta,
                })

        # Manuales: puntos elegidos a mano, de la misma fila o no.
        for conexion in self._conexiones_manuales:
            p_ini = self._posicion_punto.get(conexion["desde"])
            p_fin = self._posicion_punto.get(conexion["hasta"])
            if p_ini is None or p_fin is None:
                continue
            ts_ini = self._fecha_de_punto(df, conexion["desde"])
            ts_fin = self._fecha_de_punto(df, conexion["hasta"])
            duracion = (ts_fin - ts_ini) if (ts_ini is not None and ts_fin is not None) else None
            color = (
                QColor(COLOR_DANGER)
                if (duracion is not None and duracion.total_seconds() < 0)
                else QColor(_COLOR_CONEXION_MANUAL)
            )
            item = self._crear_linea(p_ini, p_fin, color, Qt.PenStyle.DashLine)
            self._lineas_conexion.append({
                "item": item, "desde": conexion["desde"], "hasta": conexion["hasta"],
                "duracion": duracion, "tipo": "manual", "etiqueta": "Conexión manual",
            })

    def _crear_linea(self, p_ini, p_fin, color, estilo):
        pluma = QPen(color, 2)
        # Cosmetic=True fija el grosor en PÍXELES de pantalla, sin importar
        # el zoom. Sin esto, un QGraphicsLineItem interpreta el "2" en
        # unidades de dato -- al acercar el slider de rango, esas mismas 2
        # unidades pasan a cubrir muchos más píxeles, y la línea se ve cada
        # vez más ancha hasta tapar la pantalla (justo lo que se veía).
        pluma.setCosmetic(True)
        pluma.setStyle(estilo)
        item = QGraphicsLineItem(p_ini[0], p_ini[1], p_fin[0], p_fin[1])
        item.setPen(pluma)
        item.setZValue(-1)  # detrás de los puntos, para no taparlos
        self.plot_principal.addItem(item)
        return item

    def _fecha_de_punto(self, df, referencia):
        indice, clave = referencia
        tabla, columna = clave
        df_fuente = df if tabla is None else self.host.tablas.get(tabla)
        if df_fuente is None or columna not in df_fuente.columns or indice not in df_fuente.index:
            return None
        valor = pd.to_datetime(df_fuente.loc[indice, columna], errors="coerce")
        return valor if pd.notna(valor) else None

    def _on_toggle_conectar_manual(self, activo):
        self._modo_conectar_manual = activo
        self._primer_punto_manual = None
        self._popup_info_fila.ocultar()
        self._popup_conexion.ocultar()
        if activo:
            self.lbl_estado.setText("Modo conectar activo: clic en el primer punto, luego en el segundo.")
        else:
            self._redibujar_todo()

    # ------------------------------------------------------------------
    # Slider de rango -> filtro global (solo Métricas/Indicadores/etc.)
    # ------------------------------------------------------------------
    def _on_region_cambiada(self):
        self._timer_debounce.start()

    def _aplicar_rango_activo(self):
        t0, t1 = self.region_rango.getRegion()
        self.host.rango_tiempo = (
            _epoch_a_timestamp(t0), _epoch_a_timestamp(t1), self._columna_fecha,
        )
        # apply_table_filter() no necesita redibujar ESTE panel por esto --
        # la base que usamos para dibujar (_construir_df_filtrado_base) ya
        # excluye el rango, así que nada de lo que se dibuja cambiaría.
        self._actualizando_desde_slider = True
        try:
            self.host.apply_table_filter()
        finally:
            self._actualizando_desde_slider = False

    def refrescar_vista(self):
        """Redibuja con los filtros de Datos actuales (buscador de filas,
        filtro por columna, clic en un gráfico), sin tocar los combos ni el
        menú de columnas -- eso es cosa de refrescar_datos(), pensado para
        cuando cambia el ESQUEMA, no los filtros. Se llama sola cada vez que
        cambia cualquier filtro de Datos (ver el hook en apply_table_filter),
        y también con el botón "Actualizar", por si acaso."""
        self._redibujar_todo()

    def limpiar_rango(self):
        self.host.rango_tiempo = None
        self.host.apply_table_filter()
        self._redibujar_todo()

    def _on_columna_fecha_cambiada(self, texto):
        if not texto:
            return
        self._columna_fecha = texto
        self.host.rango_tiempo = None
        self._redibujar_todo()

    # ------------------------------------------------------------------
    # Anomalías: bajo demanda, nunca en cada arrastre del slider
    # ------------------------------------------------------------------
    def _actualizar_anomalias_en_rango(self):
        df = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
        if df is None or df.empty or not self._columna_fecha:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        pares_personalizados = []
        if self.host.fingerprint_actual:
            pares_personalizados = self.host.memoria.obtener_pares_temporales(self.host.fingerprint_actual)

        detector = SemanticAnomalyDetector(df, pares_temporales_personalizados=pares_personalizados)
        anomalias = detector.detect_all()
        if self.host.fingerprint_actual:
            anomalias = self.host.memoria.enriquecer_con_memoria(self.host.fingerprint_actual, anomalias)

        cruces = anomalias_con_fecha(anomalias, df, self._columna_fecha)
        self._marcadores_anomalias = cruces
        if not cruces:
            self._scatter_anomalias.setData([])
            self.lbl_estado.setText("Sin anomalías detectadas en este rango.")
            return

        xs = [_timestamp_a_epoch(c["fecha"]) for c in cruces]
        y_base = float(self._hitos_ys_ordenados.max()) + 2 if len(self._hitos_ys_ordenados) else 1
        ys = [y_base] * len(xs)
        self._scatter_anomalias.setData(x=xs, y=ys)
        self.lbl_estado.setText(f"{len(cruces)} anomalía(s) marcada(s) en este rango.")

    # ------------------------------------------------------------------
    # Clics sobre la vista principal: punto (info de fila / conectar
    # manualmente) o línea (duración). Búsqueda de puntos siempre binaria
    # (np.searchsorted, O(log n)) -- segura con 100 mil filas.
    # ------------------------------------------------------------------
    def _punto_bajo_cursor(self, pos):
        if len(self._hitos_xs_ordenados) == 0:
            return None
        vb = self.plot_principal.getPlotItem().vb
        punto = vb.mapSceneToView(pos)
        (x0, x1), (y0, y1) = vb.viewRange()
        rect_vista = vb.boundingRect()
        datos_por_px_x = (x1 - x0) / max(rect_vista.width(), 1)
        datos_por_px_y = (y1 - y0) / max(rect_vista.height(), 1)
        tolerancia_x = datos_por_px_x * 14
        tolerancia_y = datos_por_px_y * 14

        idx = int(np.searchsorted(self._hitos_xs_ordenados, punto.x()))
        ventana = range(max(0, idx - 5), min(len(self._hitos_xs_ordenados), idx + 5))
        candidatos = [
            i for i in ventana
            if abs(self._hitos_xs_ordenados[i] - punto.x()) <= tolerancia_x
        ]
        if not candidatos:
            return None
        mejor = min(candidatos, key=lambda i: abs(self._hitos_ys_ordenados[i] - punto.y()))
        if abs(self._hitos_ys_ordenados[mejor] - punto.y()) > tolerancia_y:
            return None
        return self._hitos_indices_ordenados[mejor], self._hitos_columna_ordenada[mejor]

    def _conexion_bajo_cursor(self, pos):
        # QGraphicsLineItem con un pen de 2px ya tiene una zona de clic
        # razonable vía su shape() nativo -- se usa el picking normal de
        # Qt en vez de reinventar geometría de distancia punto-segmento.
        items_bajo_cursor = self.plot_principal.scene().items(pos)
        for conexion in self._lineas_conexion:
            if conexion["item"] in items_bajo_cursor:
                return conexion
        return None

    def _on_click_principal(self, evento):
        pos = evento.scenePos()
        if not self.plot_principal.sceneBoundingRect().contains(pos):
            self._popup_info_fila.ocultar()
            self._popup_conexion.ocultar()
            return

        if not self._modo_conectar_manual:
            conexion = self._conexion_bajo_cursor(pos)
            if conexion is not None:
                self._popup_info_fila.ocultar()
                self._mostrar_info_conexion(conexion)
                return

        resultado_punto = self._punto_bajo_cursor(pos)
        if resultado_punto is None:
            self._popup_info_fila.ocultar()
            self._popup_conexion.ocultar()
            return
        indice_real, columna_origen = resultado_punto

        if self._modo_conectar_manual:
            self._popup_conexion.ocultar()
            self._popup_info_fila.ocultar()
            if self._primer_punto_manual is None:
                self._primer_punto_manual = (indice_real, columna_origen)
                self.lbl_estado.setText(
                    f"Primer punto: fila {indice_real}, '{columna_origen}'. Clic en el segundo punto."
                )
            elif self._primer_punto_manual == (indice_real, columna_origen):
                self.lbl_estado.setText("Elige un segundo punto distinto del primero.")
            else:
                self._conexiones_manuales.append({
                    "desde": self._primer_punto_manual, "hasta": (indice_real, columna_origen),
                })
                self._primer_punto_manual = None
                self.lbl_estado.setText("Conexión creada. Puedes seguir conectando, o apagar el modo.")
                df = self.host._construir_df_filtrado_base()
                self._dibujar_hitos_multi(df)  # por si el 2do punto no estaba visible aún
                self._redibujar_conexiones(df)
            return

        self._popup_conexion.ocultar()
        tabla_origen, nombre_columna_origen = columna_origen
        df_para_fila = self.host.df if tabla_origen is None else self.host.tablas.get(tabla_origen)
        if df_para_fila is None or indice_real not in df_para_fila.index:
            self._popup_info_fila.ocultar()
            return
        fila = df_para_fila.loc[indice_real]
        texto = "<br>".join(f"<b>{col}:</b> {valor}" for col, valor in fila.items())
        if len(self._columnas_mostrar) > 1 or self._pares_conectar or self._conexiones_manuales:
            etiqueta_origen = self._etiqueta_clave(columna_origen)
            texto = f"<i>Fecha desde: {etiqueta_origen}</i><br>{texto}"
        self._popup_info_fila.mostrar(texto, QCursor.pos())

    def _mostrar_info_conexion(self, conexion):
        duracion = conexion["duracion"]
        if duracion is None or pd.isna(duracion):
            texto = f"<b>{conexion['etiqueta']}</b><br>No se pudo calcular la duración."
        else:
            texto = f"<b>{conexion['etiqueta']}</b><br>{formatear_duracion(duracion)}"
        on_eliminar = None
        if conexion["tipo"] == "manual":
            on_eliminar = lambda: self._eliminar_conexion_manual(conexion)
        self._popup_conexion.mostrar(texto, QCursor.pos(), on_eliminar=on_eliminar)

    def _eliminar_conexion_manual(self, conexion):
        self._conexiones_manuales = [
            c for c in self._conexiones_manuales
            if not (c["desde"] == conexion["desde"] and c["hasta"] == conexion["hasta"])
        ]
        self._popup_conexion.ocultar()
        self._redibujar_conexiones(self.host._construir_df_filtrado_base())

    def _on_click_marcador_anomalia(self, _scatter, puntos):
        if not puntos:
            return
        indice = puntos[0].index()
        if indice >= len(self._marcadores_anomalias):
            return
        anomalia = self._marcadores_anomalias[indice]["anomalia"]
        dialogo = _TarjetaAnomalia(anomalia, self.colors, self._enviar_anomalia_a_reporte, parent=self)
        dialogo.exec()

    def _enviar_anomalia_a_reporte(self, anomalia):
        # Reutiliza el mismo mecanismo con el que Narrativa ya manda
        # hallazgos al lienzo -- ver ReportBoxItem en main_window.py.
        self.host._reporte_agregar_anomalia_desde_linea_tiempo(anomalia)

    # ------------------------------------------------------------------
    # Métricas de tiempo (promedio/mínimo/máximo por conexión activa)
    # ------------------------------------------------------------------
    def _abrir_metricas_tiempo(self):
        df = self.host.filtered_df if self.host.filtered_df is not None else self.host.df
        if df is None or df.empty:
            QMessageBox.information(self, "Sin datos", "Carga datos primero.")
            return

        resumenes = []
        for clave in self._pares_conectar:
            par = self._pares_disponibles.get(clave)
            if par is None:
                continue
            intervalos = construir_intervalos(df, par)
            if intervalos.empty:
                continue
            resumen = resumen_duraciones(intervalos["duracion"])
            if resumen:
                resumenes.append((par.etiqueta, resumen))

        if self._conexiones_manuales:
            duraciones_manuales = []
            for conexion in self._conexiones_manuales:
                ts_ini = self._fecha_de_punto(df, conexion["desde"])
                ts_fin = self._fecha_de_punto(df, conexion["hasta"])
                if ts_ini is not None and ts_fin is not None:
                    duraciones_manuales.append(ts_fin - ts_ini)
            resumen_manual = resumen_duraciones(duraciones_manuales)
            if resumen_manual:
                resumenes.append(("Conexiones manuales", resumen_manual))

        if not resumenes:
            QMessageBox.information(
                self, "Sin conexiones",
                "No hay ninguna conexión activa todavía -- tilda un par en "
                "'Unir fechas' o conecta puntos a mano primero."
            )
            return

        dialogo = _DialogoMetricasTiempo(resumenes, self._enviar_metricas_a_reporte, parent=self)
        dialogo.exec()

    def _enviar_metricas_a_reporte(self, titulo, html):
        self.host._reporte_agregar_metricas_tiempo(titulo, html)

    # ------------------------------------------------------------------
    # Envío a la pestaña Reporte (como imagen, con botón de actualizar)
    # ------------------------------------------------------------------
    def snapshot(self) -> QPixmap:
        """Foto de la Línea de Tiempo (histograma + vista principal, con
        líneas de conexión incluidas) para la pestaña Reporte. Usa el
        exportador nativo de PyQtGraph -- igual que ChartPanel.snapshot()
        -- porque un .grab() de un widget que no está visible en pantalla
        en ese momento (ej. estás parado en Reporte, no en Línea de
        Tiempo) sale en blanco."""
        import pyqtgraph.exporters as pg_exporters

        def _exportar(plot_widget):
            try:
                exportador = pg_exporters.ImageExporter(plot_widget.getPlotItem())
                imagen = exportador.export(toBytes=True)
                pixmap = QPixmap.fromImage(imagen)
                if not pixmap.isNull():
                    return pixmap
            except Exception:
                pass
            return plot_widget.grab()

        pix_hist = _exportar(self.plot_histograma)
        pix_main = _exportar(self.plot_principal)
        ancho = max(pix_hist.width(), pix_main.width(), 1)
        alto = pix_hist.height() + pix_main.height()
        combinado = QPixmap(ancho, alto)
        combinado.fill(Qt.GlobalColor.white)
        pintor = QPainter(combinado)
        pintor.drawPixmap(0, 0, pix_hist)
        pintor.drawPixmap(0, pix_hist.height(), pix_main)
        pintor.end()
        return combinado

    def nombre(self) -> str:
        return "Línea de Tiempo"

    def etiqueta(self) -> str:
        cuantas_conexiones = len(self._pares_conectar) + len(self._conexiones_manuales)
        sufijo = f", {cuantas_conexiones} conexión(es)" if cuantas_conexiones else ""
        return f"Línea de Tiempo ({self._columna_fecha or 'sin columna'}{sufijo})"