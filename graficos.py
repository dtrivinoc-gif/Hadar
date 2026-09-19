"""
Widgets de visualización: gráfico de dona nativo (DonutChartWidget), el
panel de gráficos configurable (ChartPanel, usado en la pestaña
"Gráficos"), el worker de descarga de la UF (UfWorker) y utilidades de
theming/estilo de toda la app (build_stylesheet, create_stat_card,
render_boxplot).
"""
import math

import numpy as np
import pandas as pd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QObject, Signal, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QTableView, QTabWidget, QScrollArea, QFrame,
    QLineEdit, QRadioButton, QHeaderView, QCheckBox, QTabBar,
)

from .config import CHART_TYPES, DONUT_PALETTE, COLOR_ACCENT, COLOR_ACCENT_3, COLOR_HOVER, COLOR_DANGER
from .io_datos import fetch_uf_online

# ----------------------------------------------------------------------------
# Worker para consultar la UF online
# ----------------------------------------------------------------------------
class UfWorker(QObject):
    finished = Signal(object)

    def run(self):
        val = fetch_uf_online()
        self.finished.emit(val)


# ----------------------------------------------------------------------------
# Gráfico de Dona / Torta
# ----------------------------------------------------------------------------
class DonutChartWidget(QWidget):
    sliceClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels = []
        self.values = []
        self.text_color = QColor("#F8FAFC")
        self.setMinimumHeight(260)

    def set_data(self, labels, values, text_color="#F8FAFC"):
        self.labels = list(labels)
        self.values = list(values)
        self.text_color = QColor(text_color)
        self.update()

    def _geometry(self):
        side = min(self.width(), self.height()) - 40
        side = max(side, 10)
        cx = self.width() / 2 - side * 0.30
        cy = self.height() / 2
        rect_left = cx - side / 2
        rect_top = cy - side / 2
        return rect_left, rect_top, side

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        total = sum(self.values) if self.values else 0
        if total <= 0:
            painter.end()
            return

        rect_left, rect_top, side = self._geometry()
        inner_ratio = 0.55
        start_angle = 90 * 16
        for i, (label, val) in enumerate(zip(self.labels, self.values)):
            span = -int(round((val / total) * 360 * 16))
            color = QColor(DONUT_PALETTE[i % len(DONUT_PALETTE)])
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(self.palette().window().color()), 2))
            path = QPainterPath()
            path.moveTo(rect_left + side / 2, rect_top + side / 2)
            path.arcTo(rect_left, rect_top, side, side, start_angle / 16, span / 16)
            path.closeSubpath()
            painter.drawPath(path)
            start_angle += span

        hole = side * inner_ratio
        painter.setBrush(QBrush(self.palette().window().color()))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(rect_left + side / 2 - hole / 2), int(rect_top + side / 2 - hole / 2), int(hole), int(hole)
        )

        painter.setPen(QPen(self.text_color))
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        legend_x = int(rect_left + side + 25)
        legend_y = 20
        for i, (label, val) in enumerate(zip(self.labels, self.values)):
            color = QColor(DONUT_PALETTE[i % len(DONUT_PALETTE)])
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(legend_x, legend_y - 8, 10, 10)
            painter.setPen(QPen(self.text_color))
            pct = (val / total) * 100
            painter.drawText(legend_x + 16, legend_y + 1, f"{label} ({pct:.1f}%)")
            legend_y += 18
            if legend_y > self.height() - 10:
                break
        painter.end()

    def mousePressEvent(self, event):
        if not self.values:
            return
        rect_left, rect_top, side = self._geometry()
        cx = rect_left + side / 2
        cy = rect_top + side / 2
        pos = event.position() if hasattr(event, "position") else event.pos()
        dx = pos.x() - cx
        dy = pos.y() - cy
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > side / 2 or dist < side * 0.55 / 2:
            return

        angle = math.degrees(math.atan2(-dy, dx))
        angle = (90 - angle) % 360

        total = sum(self.values)
        cumulative = 0
        for i, val in enumerate(self.values):
            cumulative += (val / total) * 360
            if angle <= cumulative:
                self.sliceClicked.emit(i)
                return
        self.sliceClicked.emit(len(self.values) - 1)


# ----------------------------------------------------------------------------
# Panel de gráfico individual
# ----------------------------------------------------------------------------
class ChartPanel(QWidget):
    removeRequested = Signal(object)
    categoryClicked = Signal(str, object)

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.categories = []
        self.chart_x_col = None
        self.plot_widget = None
        self.donut_widget = None
        self._bar_xvals = []
        self.x_es_fecha = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame()
        self.card.setObjectName("card")
        card_layout = QVBoxLayout(self.card)
        outer.addWidget(self.card)

        controls_top = QHBoxLayout()
        card_layout.addLayout(controls_top)
        controls_bottom = QHBoxLayout()
        card_layout.addLayout(controls_bottom)

        controls_top.addWidget(QLabel("Nombre:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Sin título")
        self.name_edit.setMaximumWidth(160)
        self.name_edit.setToolTip(
            "Nombre opcional para identificar este gráfico y usarlo como título en el Reporte."
        )
        controls_top.addWidget(self.name_edit)

        controls_top.addWidget(QLabel("Tipo:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(CHART_TYPES)
        controls_top.addWidget(self.combo_type)
        controls_top.addStretch()

        self.lbl_eje_x = QLabel("Eje X:")
        controls_bottom.addWidget(self.lbl_eje_x)
        self.combo_x = QComboBox()
        # Sin esto, el combo se ensancha solo según el nombre de columna más
        # largo que le carguen (ver update_axis_choices) -- con columnas
        # reales tipo "fecha_contratacion" eso rompe la fila de controles.
        # AdjustToMinimumContentsLengthWithIcon + minimumContentsLength fija
        # un ancho visible razonable; el nombre completo se sigue viendo
        # completo en el desplegable, solo se trunca en el cuadro cerrado.
        self.combo_x.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_x.setMinimumContentsLength(14)
        self.combo_x.setMaximumWidth(180)
        controls_bottom.addWidget(self.combo_x)

        self.lbl_eje_y = QLabel("Eje Y:")
        controls_bottom.addWidget(self.lbl_eje_y)
        self.combo_y = QComboBox()
        self.combo_y.addItem("Conteo")
        self.combo_y.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_y.setMinimumContentsLength(14)
        self.combo_y.setMaximumWidth(180)
        controls_bottom.addWidget(self.combo_y)

        self.btn_generate = QPushButton("Generar")
        self.btn_generate.setObjectName("accentButton")
        controls_bottom.addWidget(self.btn_generate)

        self.chk_anotaciones = QCheckBox("Anotaciones")
        self.chk_anotaciones.setChecked(True)
        self.chk_anotaciones.setToolTip(
            "Muestra valores directamente sobre el gráfico y una línea de referencia con el promedio."
        )
        self.chk_anotaciones.toggled.connect(lambda: self.render_requested())
        controls_bottom.addWidget(self.chk_anotaciones)
        controls_bottom.addStretch()

        self.btn_delete = QPushButton("X")
        self.btn_delete.setObjectName("dangerButton")
        self.btn_delete.setFixedWidth(32)
        controls_bottom.addWidget(self.btn_delete)

        self.plot_container = QVBoxLayout()
        card_layout.addLayout(self.plot_container)

        self.btn_generate.clicked.connect(lambda: self.render_requested())
        self.btn_delete.clicked.connect(lambda: self.removeRequested.emit(self))
        self.combo_type.currentTextChanged.connect(self._on_tipo_changed)

        self._render_callback = None
        self._on_tipo_changed(self.combo_type.currentText())

    def set_render_callback(self, callback):
        self._render_callback = callback

    def _on_tipo_changed(self, tipo):
        es_heatmap = (tipo == "Mapa de Calor (Correlación)")
        self.lbl_eje_x.setVisible(not es_heatmap)
        self.combo_x.setVisible(not es_heatmap)
        self.lbl_eje_y.setVisible(not es_heatmap)
        self.combo_y.setVisible(not es_heatmap)

    def render_requested(self):
        if self._render_callback:
            self._render_callback(self)

    def update_axis_choices(self, all_cols, num_cols):
        current_x = self.combo_x.currentText()
        current_y = self.combo_y.currentText()

        self.combo_x.blockSignals(True)
        self.combo_x.clear()
        self.combo_x.addItems(all_cols)
        if current_x in all_cols:
            self.combo_x.setCurrentText(current_x)
        elif all_cols:
            self.combo_x.setCurrentIndex(0)
        self.combo_x.blockSignals(False)

        self.combo_y.blockSignals(True)
        self.combo_y.clear()
        self.combo_y.addItems(["Conteo"] + num_cols)
        if current_y in ["Conteo"] + num_cols:
            self.combo_y.setCurrentText(current_y)
        else:
            self.combo_y.setCurrentIndex(0)
        self.combo_y.blockSignals(False)

    def _clear_plot_area(self):
        while self.plot_container.count():
            item = self.plot_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.plot_widget = None
        self.donut_widget = None

    def apply_theme(self, colors):
        self.colors = colors

    @staticmethod
    def _parsear_fechas(serie_texto):
        """Convierte una Serie de texto a fechas probando primero formato
        ISO/mes-primero y, si rinde peor, DD/MM/AAAA (común en Chile). Evita
        el bug clásico de forzar dayfirst=True sobre fechas ISO como
        '2024-02-20', que quedarían mal parseadas o en NaT."""
        opcion_iso = pd.to_datetime(serie_texto, errors="coerce", dayfirst=False)
        opcion_dia_primero = pd.to_datetime(serie_texto, errors="coerce", dayfirst=True)
        if opcion_dia_primero.notna().sum() > opcion_iso.notna().sum():
            return opcion_dia_primero
        return opcion_iso

    @classmethod
    def _es_columna_fecha(cls, df, col):
        """Detecta si una columna es de tipo fecha: ya viene como datetime64,
        o su contenido (como texto) se puede parsear como fecha en la gran
        mayoría de los casos."""
        serie = df[col]
        if pd.api.types.is_datetime64_any_dtype(serie):
            return True
        if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_bool_dtype(serie):
            return False
        muestra = serie.dropna().astype(str).head(50)
        if muestra.empty:
            return False
        parseadas = cls._parsear_fechas(muestra)
        return parseadas.notna().mean() > 0.8

    def render(self, df: pd.DataFrame):
        chart_type = self.combo_type.currentText()
        x_col = self.combo_x.currentText()
        y_col = self.combo_y.currentText()

        if df is None or df.empty:
            return
        if chart_type != "Mapa de Calor (Correlación)" and (not x_col or x_col not in df.columns):
            return

        self._clear_plot_area()
        self.categories = []
        self._bar_xvals = []

        if chart_type == "Mapa de Calor (Correlación)":
            self.chart_x_col = None
            self.x_es_fecha = False
            self._render_heatmap_correlacion(df)
            return

        self.chart_x_col = x_col
        self.x_es_fecha = self._es_columna_fecha(df, x_col)

        if chart_type == "Dona / Torta":
            self._render_donut(df, x_col)
        else:
            self._render_pg_chart(df, chart_type, x_col, y_col)

    def _render_pg_chart(self, df, chart_type, x_col, y_col):
        usar_eje_fecha = self.x_es_fecha and chart_type in ("Barras", "Líneas")
        if usar_eje_fecha:
            date_axis = pg.DateAxisItem(orientation="bottom")
            pw = pg.PlotWidget(axisItems={"bottom": date_axis})
        else:
            pw = pg.PlotWidget()
        pw.setBackground(self.colors["card"])
        pw.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        pw.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        pw.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        pw.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        pw.showGrid(x=False, y=True, alpha=0.15)
        pw.setMinimumHeight(300)
        pw.setClipToView(True)
        self.plot_container.addWidget(pw)
        self.plot_widget = pw

        try:
            if chart_type == "Barras":
                if usar_eje_fecha:
                    labels, values, xvals = self._datos_ordenados_por_fecha(df, x_col, y_col)
                else:
                    if y_col == "Conteo":
                        counts = df[x_col].value_counts()
                        labels = counts.index.astype(str).tolist()
                        values = counts.values.tolist()
                    else:
                        labels = df[x_col].astype(str).tolist()
                        values = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()
                    xvals = list(range(len(labels)))

                self.categories = labels
                self._bar_xvals = xvals
                ancho = self._ancho_barra_para(xvals, usar_eje_fecha)
                bar = pg.BarGraphItem(x=xvals, height=values, width=ancho, brush=COLOR_ACCENT)
                pw.addItem(bar)
                if not usar_eje_fecha:
                    self._set_category_ticks(pw, labels)
                pw.scene().sigMouseClicked.connect(
                    lambda ev, pw=pw: self._on_bar_clicked(ev, pw)
                )
                self._agregar_hover_categoria(pw, xvals, labels, values)
                if self.chk_anotaciones.isChecked():
                    self._agregar_anotaciones_serie(pw, xvals, values)

            elif chart_type == "Líneas":
                if usar_eje_fecha:
                    labels, values, xvals = self._datos_ordenados_por_fecha(df, x_col, y_col)
                else:
                    if y_col == "Conteo":
                        counts = df[x_col].value_counts().sort_index()
                        labels = counts.index.astype(str).tolist()
                        values = counts.values.tolist()
                    else:
                        labels = df[x_col].astype(str).tolist()
                        values = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()
                    xvals = list(range(len(labels)))

                self.categories = labels
                self._bar_xvals = xvals
                # Con pocos puntos el símbolo ayuda a ubicar cada dato; con
                # muchos, dibujar un círculo por punto es lo que traba la UI
                # y además no se alcanza a ver (se pisan entre sí).
                usar_simbolos = len(values) <= 500
                curva = pw.plot(
                    xvals, values, pen=pg.mkPen(COLOR_ACCENT, width=2),
                    symbol="o" if usar_simbolos else None,
                    symbolBrush=COLOR_ACCENT, symbolSize=6,
                )
                if not usar_simbolos:
                    curva.setDownsampling(auto=True, method="peak")
                if not usar_eje_fecha:
                    self._set_category_ticks(pw, labels)
                self._agregar_hover_categoria(pw, xvals, labels, values)
                if self.chk_anotaciones.isChecked():
                    self._agregar_anotaciones_serie(pw, xvals, values)

            elif chart_type == "Dispersión":
                if y_col != "Conteo":
                    xs = pd.to_numeric(df[x_col], errors="coerce")
                    ys = pd.to_numeric(df[y_col], errors="coerce")
                    mask = xs.notna() & ys.notna()
                    n_puntos = int(mask.sum())
                    # Con demasiados puntos, dibujarlos uno a uno solo produce
                    # una mancha ilegible (y traba la UI). Mejor mostrar la
                    # densidad: dónde se concentran los datos, que es la
                    # pregunta real detrás de un scatter con este volumen.
                    if n_puntos > 20_000:
                        self._render_densidad(pw, xs[mask].to_numpy(), ys[mask].to_numpy())
                    else:
                        scatter = pg.ScatterPlotItem(
                            x=xs[mask].tolist(), y=ys[mask].tolist(),
                            brush=pg.mkBrush(COLOR_ACCENT + "B3"), pen=None, size=8
                        )
                        pw.addItem(scatter)
                    if self.chk_anotaciones.isChecked() and mask.any():
                        self._linea_referencia(
                            pw, float(ys[mask].mean()), angulo=0,
                            texto=f"Promedio Y: {self._formato_num(float(ys[mask].mean()))}"
                        )
                        self._linea_referencia(
                            pw, float(xs[mask].mean()), angulo=90,
                            texto=f"Promedio X: {self._formato_num(float(xs[mask].mean()))}"
                        )
                else:
                    pw.addItem(pg.TextItem("Selecciona una columna Y numérica", color=self.colors["muted"]))

            elif chart_type == "Histograma":
                data = pd.to_numeric(df[x_col], errors="coerce").dropna()
                if not data.empty:
                    y, x = np.histogram(data, bins=20)
                    bar = pg.BarGraphItem(
                        x=(x[:-1] + x[1:]) / 2, height=y, width=(x[1] - x[0]) * 0.9, brush=COLOR_ACCENT
                    )
                    pw.addItem(bar)
                    if self.chk_anotaciones.isChecked():
                        self._linea_referencia(
                            pw, float(data.mean()), angulo=90,
                            texto=f"Promedio: {self._formato_num(float(data.mean()))}"
                        )
        except Exception as e:
            pw.addItem(pg.TextItem(f"Error: {e}", color=COLOR_DANGER))

    def _set_category_ticks(self, pw, labels, max_ticks=9):
        """Menos etiquetas que categorías reales, a propósito: con muchas
        barras no caben todos los nombres sin que se amontonen, y forzar el
        texto no soluciona nada (Tufte: mejor pocas etiquetas legibles que
        muchas ilegibles). El nombre exacto de cada barra se ve completo
        al pasar el mouse por encima (ver _agregar_hover_categoria)."""
        n = len(labels)
        if n == 0:
            return
        step = max(1, math.ceil(n / max_ticks))
        largo_max = 12

        def _acortar(txt):
            txt = str(txt)
            return txt if len(txt) <= largo_max else txt[:largo_max - 1] + "…"

        ticks = [(i, _acortar(lbl)) for i, lbl in enumerate(labels) if i % step == 0]
        pw.getAxis("bottom").setTicks([ticks])

    @classmethod
    def _datos_ordenados_por_fecha(cls, df, x_col, y_col):
        """Para columnas de fecha: ordena cronológicamente y entrega
        posiciones X reales en segundos-epoch, para usar con DateAxisItem
        (así el espaciado entre barras/puntos refleja el tiempo real, no
        solo el orden de aparición en la tabla)."""
        fechas = cls._parsear_fechas(df[x_col])

        if y_col == "Conteo":
            validos = fechas.notna()
            conteo = df.loc[validos, x_col].value_counts()
            fechas_unicas = cls._parsear_fechas(pd.Series(conteo.index))
            tmp = pd.DataFrame({
                "_label": conteo.index.astype(str),
                "_valor": conteo.values,
                "_fecha": fechas_unicas,
            }).dropna(subset=["_fecha"]).sort_values("_fecha")
        else:
            tmp = pd.DataFrame({
                "_label": df[x_col].astype(str),
                "_valor": pd.to_numeric(df[y_col], errors="coerce"),
                "_fecha": fechas,
            }).dropna(subset=["_fecha", "_valor"]).sort_values("_fecha")

        labels = tmp["_label"].tolist()
        values = tmp["_valor"].tolist()
        xvals = (tmp["_fecha"].astype("int64") // 10**9).tolist()
        return labels, values, xvals

    @staticmethod
    def _ancho_barra_para(xvals, es_fecha):
        if not es_fecha or len(xvals) < 2:
            return 0.6
        diffs = sorted(b - a for a, b in zip(xvals, xvals[1:]) if b > a)
        if not diffs:
            return 0.6
        return diffs[len(diffs) // 2] * 0.7

    @staticmethod
    def _formato_num(v):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if float(v).is_integer():
            return f"{v:.0f}"
        return f"{v:.2f}"

    def _linea_referencia(self, pw, valor, angulo, texto):
        """Línea de referencia (ej. promedio) con su etiqueta pegada a la
        línea — contexto narrativo tipo Tufte en vez de un número suelto."""
        linea = pg.InfiniteLine(
            pos=valor, angle=angulo,
            pen=pg.mkPen(self.colors["muted"], width=1, style=Qt.DashLine),
            label=texto,
            labelOpts={"color": self.colors["muted"], "position": 0.95, "movable": False},
        )
        pw.addItem(linea)

    def _render_densidad(self, pw, xs, ys, bins=120):
        """Histograma 2D en vez de puntos individuales: con cientos de miles
        de filas la nube de puntos se satura y no se lee nada; la densidad sí
        muestra dónde se concentran realmente los datos."""
        hist, x_bordes, y_bordes = np.histogram2d(xs, ys, bins=bins)
        img = pg.ImageItem(hist)
        img.setRect(QRectF(x_bordes[0], y_bordes[0],
                            x_bordes[-1] - x_bordes[0], y_bordes[-1] - y_bordes[0]))
        colormap = pg.ColorMap(
            [0.0, 0.5, 1.0],
            [pg.mkColor(self.colors["card"]), pg.mkColor(COLOR_ACCENT), pg.mkColor("#f3f4f6")],
        )
        img.setLookupTable(colormap.getLookupTable(0.0, 1.0, 256))
        pw.addItem(img)
        nota = pg.TextItem(
            f"{len(xs):,} puntos agrupados por densidad".replace(",", "."),
            color=self.colors["muted"], anchor=(0, 0),
        )
        nota.setPos(x_bordes[0], y_bordes[-1])
        pw.addItem(nota)

    def _agregar_hover_categoria(self, pw, xvals, categories, values):
        """Al mover el mouse sobre el gráfico, muestra el nombre completo de
        la categoría y su valor exacto — así no hace falta que el eje X
        muestre todas las etiquetas a la vez para saber qué es cada barra."""
        if not xvals:
            return

        fondo = self.colors["card"] + "E6"  # semi-transparente
        etiqueta = pg.TextItem(anchor=(0, 1), color=self.colors["text"], fill=fondo,
                                border=pg.mkPen(self.colors["muted"], width=1))
        etiqueta.hide()
        pw.addItem(etiqueta, ignoreBounds=True)

        linea = pg.InfiniteLine(angle=90, movable=False,
                                 pen=pg.mkPen(self.colors["muted"], width=1, style=Qt.DashLine))
        linea.hide()
        pw.addItem(linea)

        paso_medio = (max(xvals) - min(xvals) + 1) / max(len(xvals), 1) if len(xvals) > 1 else 1

        def _mover(evento):
            escena_pos = evento[0]
            vb = pw.getPlotItem().getViewBox()
            if not pw.sceneBoundingRect().contains(escena_pos):
                etiqueta.hide()
                linea.hide()
                return
            punto = vb.mapSceneToView(escena_pos)
            idx = min(range(len(xvals)), key=lambda i: abs(xvals[i] - punto.x()))
            if abs(xvals[idx] - punto.x()) > paso_medio:
                etiqueta.hide()
                linea.hide()
                return
            etiqueta.setText(f"{categories[idx]}\n{self._formato_num(values[idx])}")
            etiqueta.setPos(xvals[idx], values[idx])
            linea.setPos(xvals[idx])
            etiqueta.show()
            linea.show()

        proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=30, slot=_mover)
        pw._hover_proxy = proxy  # referencia viva para que no la borre el GC

    def _agregar_anotaciones_serie(self, pw, xvals, values):
        """Línea de referencia con el promedio de la serie, y —si no hay
        demasiados puntos como para saturar el gráfico— el valor exacto
        escrito directamente sobre cada barra/punto (etiquetado directo en
        vez de obligar a leer los ejes)."""
        pares = [(x, v) for x, v in zip(xvals, values)
                 if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not pares:
            return
        valores_validos = [v for _, v in pares]
        promedio = float(np.mean(valores_validos))
        self._linea_referencia(pw, promedio, angulo=0, texto=f"Promedio: {self._formato_num(promedio)}")

        if len(pares) <= 25:
            for x, v in pares:
                etiqueta = pg.TextItem(self._formato_num(v), color=self.colors["text"], anchor=(0.5, 1.3))
                etiqueta.setPos(x, v)
                pw.addItem(etiqueta)

    def _on_bar_clicked(self, ev, pw):
        if not self._bar_xvals:
            return
        vb = pw.getPlotItem().getViewBox()
        point = vb.mapSceneToView(ev.scenePos())
        idx = min(range(len(self._bar_xvals)), key=lambda i: abs(self._bar_xvals[i] - point.x()))
        if 0 <= idx < len(self.categories):
            self.categoryClicked.emit(self.chart_x_col, self.categories[idx])

    def _render_donut(self, df, x_col):
        counts = df[x_col].value_counts()
        self.categories = counts.index.astype(str).tolist()
        donut = DonutChartWidget()
        donut.set_data(self.categories, counts.values.tolist(), text_color=self.colors["text"])
        donut.sliceClicked.connect(self._on_donut_clicked)
        self.plot_container.addWidget(donut)
        self.donut_widget = donut

    def _on_donut_clicked(self, idx):
        if 0 <= idx < len(self.categories):
            self.categoryClicked.emit(self.chart_x_col, self.categories[idx])

    def _render_heatmap_correlacion(self, df):
        """Mapa de calor de correlación entre todas las columnas numéricas,
        con el valor exacto anotado en cada celda (etiquetado directo) para
        no depender solo del color para leer el dato."""
        num_df = df.select_dtypes(include=[np.number])
        num_df = num_df.loc[:, num_df.nunique(dropna=True) > 1]  # descarta columnas constantes

        if num_df.shape[1] < 2:
            aviso = QLabel(
                "Se necesitan al menos 2 columnas numéricas (no constantes) para calcular correlaciones."
            )
            aviso.setObjectName("muted")
            aviso.setWordWrap(True)
            self.plot_container.addWidget(aviso)
            return

        corr = num_df.corr()
        cols = corr.columns.tolist()
        n = len(cols)
        matriz = corr.values

        pw = pg.PlotWidget()
        pw.setBackground(self.colors["card"])
        pw.getAxis("left").setPen(pg.mkPen(self.colors["muted"]))
        pw.getAxis("bottom").setPen(pg.mkPen(self.colors["muted"]))
        pw.getAxis("left").setTextPen(pg.mkPen(self.colors["text"]))
        pw.getAxis("bottom").setTextPen(pg.mkPen(self.colors["text"]))
        pw.showGrid(x=False, y=False)
        pw.setMinimumHeight(max(300, 40 * n))
        pw.getPlotItem().getViewBox().setAspectLocked(True)
        pw.getPlotItem().getViewBox().invertY(True)
        self.plot_container.addWidget(pw)
        self.plot_widget = pw

        # Mapa de color divergente: azul (-1) -> blanco (0) -> rojo (+1),
        # para leer el signo de la correlación de un vistazo.
        colormap = pg.ColorMap(
            [0.0, 0.5, 1.0],
            [pg.mkColor("#3b82f6"), pg.mkColor("#f3f4f6"), pg.mkColor("#ef4444")],
        )
        img = pg.ImageItem(matriz)
        img.setLookupTable(colormap.getLookupTable(0.0, 1.0, 256))
        img.setLevels((-1, 1))
        pw.addItem(img)

        ticks = [(i + 0.5, str(c)[:16]) for i, c in enumerate(cols)]
        pw.getAxis("bottom").setTicks([ticks])
        pw.getAxis("left").setTicks([ticks])

        # Valor exacto escrito en cada celda: la anotación es el dato, el
        # color es solo apoyo visual para escanear la matriz rápido.
        for i in range(n):
            for j in range(n):
                valor = matriz[i, j]
                color_texto = "#111827" if abs(valor) < 0.6 else "#ffffff"
                texto = pg.TextItem(f"{valor:.2f}", color=color_texto, anchor=(0.5, 0.5))
                texto.setPos(i + 0.5, j + 0.5)
                pw.addItem(texto)

        pw.setXRange(0, n, padding=0.02)
        pw.setYRange(0, n, padding=0.02)

        leyenda = QLabel("Correlación positiva  ·  Correlación negativa  ·  escala de -1 a 1")
        leyenda.setObjectName("muted")
        self.plot_container.addWidget(leyenda)

        self.categories = cols

    def snapshot(self) -> QPixmap:
        """Devuelve una foto (QPixmap) del gráfico tal como se ve ahora mismo,
        para usar en la pestaña Reporte. Usa el exportador nativo de
        pyqtgraph (renderiza directo desde el gráfico, no desde lo que haya
        pintado en pantalla) para que funcione aunque la pestaña Gráficos no
        esté visible en ese momento; si por algún motivo eso falla, cae de
        vuelta a una captura de pantalla normal del widget."""
        if self.plot_widget is not None:
            try:
                import pyqtgraph.exporters as pg_exporters
                exportador = pg_exporters.ImageExporter(self.plot_widget.getPlotItem())
                imagen = exportador.export(toBytes=True)
                pixmap = QPixmap.fromImage(imagen)
                if not pixmap.isNull():
                    return pixmap
            except Exception:
                pass
            return self.plot_widget.grab()
        if self.donut_widget is not None:
            return self.donut_widget.grab()
        return QPixmap()

    def nombre(self) -> str:
        return self.name_edit.text().strip()

    def etiqueta(self) -> str:
        tipo = self.combo_type.currentText()
        if tipo == "Mapa de Calor (Correlación)":
            base = tipo
        else:
            base = f"{tipo} — X: {self.combo_x.currentText()}, Y: {self.combo_y.currentText()}"
        nombre = self.nombre()
        return f"{nombre} ({base})" if nombre else base


# ----------------------------------------------------------------------------
# Boxplot horizontal simple
# ----------------------------------------------------------------------------
def render_boxplot(plot_widget: pg.PlotWidget, data: pd.Series, colors):
    plot_widget.clear()
    data = data.dropna()
    if data.empty:
        return

    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    y_center = 0
    half = 0.35
    pen = pg.mkPen(colors["text"], width=1.5)
    brush = pg.mkBrush(COLOR_ACCENT + "AA")

    box_x = [q1, q1, q3, q3, q1]
    box_y = [y_center - half, y_center + half, y_center + half, y_center - half, y_center - half]
    plot_widget.addItem(pg.PlotCurveItem(box_x, box_y, pen=pen, brush=brush, fillLevel=y_center - half))

    plot_widget.plot([median, median], [y_center - half, y_center + half], pen=pg.mkPen("white", width=2))
    plot_widget.plot([lower_whisker, q1], [y_center, y_center], pen=pen)
    plot_widget.plot([q3, upper_whisker], [y_center, y_center], pen=pen)
    plot_widget.plot([lower_whisker, lower_whisker], [y_center - half / 2, y_center + half / 2], pen=pen)
    plot_widget.plot([upper_whisker, upper_whisker], [y_center - half / 2, y_center + half / 2], pen=pen)

    plot_widget.setYRange(-1, 1)
    plot_widget.getAxis("left").setTicks([[(0, "")]])


# ----------------------------------------------------------------------------
# Hoja de estilos (QSS)
# ----------------------------------------------------------------------------
def build_stylesheet(colors):
    return f"""
    QMainWindow, QWidget {{
        background-color: {colors['bg']};
        color: {colors['text']};
        font-family: "Segoe UI";
    }}
    QFrame#sidebar {{
        background-color: {colors['card']};
        border-right: 1px solid {colors['border']};
    }}
    QFrame#sidebar[colapsado="true"] {{
        background-color: {colors['bg']};
        border-right: none;
    }}
    QPushButton#sidebarToggleBtn {{
        background-color: {COLOR_ACCENT};
        border: none;
        border-radius: 4px;
        color: white;
        font-size: 12px;
        font-weight: bold;
        padding: 0px;
    }}
    QPushButton#sidebarToggleBtn:hover {{
        background-color: {COLOR_HOVER};
    }}
    QLabel#sectionLabel {{
        color: {colors['muted']};
        font-size: 10px;
        font-weight: 600;
        padding-top: 4px;
    }}
    QFrame#subPanel {{
        background-color: {colors['bg']};
        border: none;
        border-left: 2px solid {COLOR_ACCENT};
        border-radius: 4px;
    }}
    QFrame#fileChip {{
        background-color: {colors['bg']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
    }}
    QFrame#card, QFrame#statCard {{
        background-color: {colors['card']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
    }}
    QFrame#statCard[alarma="true"] {{
        border: 2px solid {COLOR_DANGER};
    }}
    QLabel#title {{
        font-size: 18px;
        font-weight: bold;
    }}
    QLabel#header {{
        font-size: 22px;
        font-weight: bold;
    }}
    QLabel#muted {{
        color: {colors['muted']};
    }}
    QLabel#statValue {{
        font-size: 20px;
        font-weight: bold;
    }}
    QLabel#statTitle {{
        color: {colors['muted']};
        font-size: 11px;
    }}
    QPushButton {{
        background-color: {colors['border']};
        border: none;
        border-radius: 6px;
        padding: 8px 12px;
        color: {colors['text']};
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT};
        color: white;
    }}
    QPushButton:checkable:checked {{
        background-color: {COLOR_ACCENT_3};
        color: white;
    }}
    QPushButton#accentButton {{
        background-color: {COLOR_ACCENT};
        color: white;
        font-weight: bold;
    }}
    QPushButton#accentButton:hover {{
        background-color: {COLOR_HOVER};
    }}
    QPushButton#dangerButton {{
        background-color: transparent;
        color: {colors['muted']};
    }}
    QPushButton#dangerButton:hover {{
        background-color: {COLOR_DANGER};
        color: white;
    }}
    QComboBox, QLineEdit, QListWidget {{
        background-color: {colors['bg']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        padding: 4px 6px;
        color: {colors['text']};
    }}
    QTabWidget::pane {{
        border: none;
    }}
    QTabBar::tab {{
        background: {colors['card']};
        border: 1px solid {colors['border']};
        padding: 8px 16px;
        margin-right: 4px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        color: {colors['text']};
    }}
    QTabBar::tab:selected {{
        background: {COLOR_ACCENT};
        color: white;
    }}
    QTableView {{
        background-color: {colors['card']};
        alternate-background-color: {colors['bg']};
        gridline-color: {colors['border']};
        border: 1px solid {colors['border']};
        color: {colors['text']};
    }}
    QHeaderView::section {{
        background-color: {colors['bg']};
        color: {colors['text']};
        padding: 6px;
        border: none;
        border-bottom: 1px solid {colors['border']};
        font-weight: bold;
    }}
    QRadioButton, QCheckBox {{
        color: {colors['text']};
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px;
        height: 15px;
        border: 1px solid {colors['muted']};
        border-radius: 3px;
        background-color: {colors['bg']};
    }}
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT};
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border: 1px solid {COLOR_ACCENT};
    }}
    QScrollArea {{
        border: none;
    }}
    """


def create_stat_card(title):
    card = QFrame()
    card.setObjectName("statCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(15, 12, 15, 12)
    lbl_title = QLabel(title)
    lbl_title.setObjectName("statTitle")
    lbl_value = QLabel("-")
    lbl_value.setObjectName("statValue")
    layout.addWidget(lbl_title)
    layout.addWidget(lbl_value)
    return card, lbl_value


def set_card_alarma(card, en_alarma):
    """Marca (o desmarca) con marco rojo una tarjeta creada con
    create_stat_card, según si la regla de Alarma asociada está activa
    ahora mismo. Usa una propiedad dinámica de Qt (no un setStyleSheet
    directo) para no pisar el resto del tema."""
    card.setProperty("alarma", "true" if en_alarma else "false")
    card.style().unpolish(card)
    card.style().polish(card)