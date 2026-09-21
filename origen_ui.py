"""
Panel "Origen" de Narrativa: dibuja el grafo de origen_grafo.py como cajas
conectadas de izquierda a derecha (de dónde vinieron los datos -> qué se les
hizo -> columnas calculadas e indicadores), con:
  - leyenda de colores con conteos,
  - zoom con la rueda / botones y arrastrar para moverse ("Ajustar" lo encuadra),
  - clic en una caja: se resalta de dónde viene y a qué puede afectar (el resto
    se atenúa) y abajo se muestra su detalle en lenguaje llano.

Es solo lectura a propósito: muestra lo que pasó, no se edita nada desde acá.

OJO (mismo problema que en ontologia_ui.py): PySide6 no protege de la basura de
Python a los QGraphicsItem por el solo hecho de estar en una escena; por eso el
panel guarda referencias duras a cajas y flechas (self._cajas / self._aristas).
"""
import html

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPainterPath, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QGraphicsItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget,
)

from .config import COLOR_ACCENT, COLOR_ACCENT_3
from .origen_grafo import (
    NODO_ORIGEN, NODO_LIMPIEZA, NODO_MANOS, NODO_UNION, NODO_CALCULADA, NODO_INDICADOR,
)

ANCHO_CAJA = 156
ALTO_CAJA = 60
ALTO_CABECERA = 24
SEPARACION_X = 48      # largo de las flechas entre capas
SEPARACION_Y = 24
ZOOM_MINIMO_LEGIBLE = 0.4   # 'Ajustar' no encoge más que esto (debajo de eso el texto no se lee)

# Color de la cabecera de cada tipo de caja (mismos tres grupos que la leyenda).
COLOR_ORIGEN = "#64748B"
COLOR_PASOS = "#0E7490"
COLOR_CALCULADAS = COLOR_ACCENT
COLOR_OJO = COLOR_ACCENT_3
COLOR_OJO_TEXTO = "#D97706"   # ámbar un poco más oscuro para texto (se lee también sobre fondo claro)

_COLOR_POR_TIPO = {
    NODO_ORIGEN: COLOR_ORIGEN,
    NODO_LIMPIEZA: COLOR_PASOS,
    NODO_MANOS: COLOR_PASOS,
    NODO_UNION: COLOR_PASOS,
    NODO_CALCULADA: COLOR_CALCULADAS,
    NODO_INDICADOR: COLOR_CALCULADAS,
}


class _CajaOrigen(QGraphicsItem):
    """Una caja del diagrama: cabecera de color con el título y, debajo, una
    línea corta. Se dibuja entera en paint() (sin hijos) para no depender del
    recolector de basura."""

    def __init__(self, nodo, colors, al_hacer_clic):
        super().__init__()
        self.nodo = nodo
        self.colors = colors
        self._al_hacer_clic = al_hacer_clic
        self.seleccionada = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{nodo.titulo}\n{nodo.subtitulo}")
        self.setZValue(1)

    def boundingRect(self):
        return QRectF(-3, -3, ANCHO_CAJA + 6, ALTO_CAJA + 6)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        c = self.colors
        rect = QRectF(0, 0, ANCHO_CAJA, ALTO_CAJA)
        forma = QPainterPath()
        forma.addRoundedRect(rect, 8, 8)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(c["card"])))
        painter.drawPath(forma)

        painter.save()
        painter.setClipPath(forma)
        painter.fillRect(QRectF(0, 0, ANCHO_CAJA, ALTO_CABECERA),
                         QColor(_COLOR_POR_TIPO.get(self.nodo.tipo, COLOR_ORIGEN)))
        painter.restore()

        grosor = 2.2 if self.seleccionada else 1.0
        color_borde = c["text"] if self.seleccionada else c["border"]
        painter.setPen(QPen(QColor(color_borde), grosor))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(forma)

        # título (sobre la cabecera)
        fuente_titulo = QFont()
        fuente_titulo.setBold(True)
        fuente_titulo.setPointSizeF(9.5)
        painter.setFont(fuente_titulo)
        painter.setPen(QColor("#FFFFFF"))
        ancho_titulo = ANCHO_CAJA - 20 - (40 if self.nodo.ojo else 0)
        titulo = QFontMetrics(fuente_titulo).elidedText(self.nodo.titulo, Qt.ElideRight, ancho_titulo)
        painter.drawText(QRectF(10, 0, ancho_titulo, ALTO_CABECERA), Qt.AlignVCenter | Qt.AlignLeft, titulo)

        # subtítulo (sobre el cuerpo)
        fuente_sub = QFont()
        fuente_sub.setPointSizeF(8.5)
        painter.setFont(fuente_sub)
        painter.setPen(QColor(c["muted"]))
        sub = QFontMetrics(fuente_sub).elidedText(self.nodo.subtitulo, Qt.ElideRight, ANCHO_CAJA - 20)
        painter.drawText(QRectF(10, ALTO_CABECERA, ANCHO_CAJA - 20, ALTO_CAJA - ALTO_CABECERA),
                         Qt.AlignVCenter | Qt.AlignLeft, sub)

        # etiqueta "Ojo" (puede estar desactualizado)
        if self.nodo.ojo:
            etiqueta = QRectF(ANCHO_CAJA - 38, 4, 32, 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(COLOR_OJO))
            painter.drawRoundedRect(etiqueta, 4, 4)
            fuente_ojo = QFont()
            fuente_ojo.setBold(True)
            fuente_ojo.setPointSizeF(8)
            painter.setFont(fuente_ojo)
            painter.setPen(QColor("#1F2937"))
            painter.drawText(etiqueta, Qt.AlignCenter, "Ojo")

    def mousePressEvent(self, event):
        # hay que aceptar el press para recibir el release (y para que la vista
        # no empiece a arrastrar el lienzo cuando el clic es sobre una caja)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._al_hacer_clic(self.nodo.id)
        event.accept()


class _AristaOrigen(QGraphicsItem):
    """Flecha curva de una caja a la siguiente (sale por la derecha, entra por la izquierda)."""

    def __init__(self, desde: QPointF, hasta: QPointF, colors):
        super().__init__()
        self.colors = colors
        self.setZValue(0)
        self._ruta = QPainterPath(desde)
        fin = QPointF(hasta.x() - 8, hasta.y())
        dx = max(24.0, (fin.x() - desde.x()) * 0.5)
        self._ruta.cubicTo(QPointF(desde.x() + dx, desde.y()), QPointF(fin.x() - dx, fin.y()), fin)
        self._cabeza = QPolygonF([hasta, QPointF(hasta.x() - 8, hasta.y() - 4.5),
                                  QPointF(hasta.x() - 8, hasta.y() + 4.5)])

    def boundingRect(self):
        return self._ruta.boundingRect().united(self._cabeza.boundingRect()).adjusted(-3, -3, 3, 3)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(self.colors["muted"])
        painter.setPen(QPen(color, 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self._ruta)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(self._cabeza)


def _construir_escena(grafo, colors, al_hacer_clic, escena=None):
    """Arma la escena (cajas + flechas) de un grafo, en `escena` si se pasa una (el panel
    interactivo) o en una nueva (el render a imagen del informe). Devuelve
    (escena, {id: caja}, [(desde, hasta, flecha)]); quien la llame debe guardar esas
    referencias mientras use la escena."""
    if escena is None:
        escena = QGraphicsScene()
    cajas, aristas = {}, []
    for n in grafo.nodos:
        caja = _CajaOrigen(n, colors, al_hacer_clic)
        caja.setPos(n.capa * (ANCHO_CAJA + SEPARACION_X), n.fila * (ALTO_CAJA + SEPARACION_Y))
        escena.addItem(caja)
        cajas[n.id] = caja
    for a in grafo.aristas:
        ca, cb = cajas[a.desde], cajas[a.hasta]
        desde = ca.pos() + QPointF(ANCHO_CAJA, ALTO_CAJA / 2)
        hasta = cb.pos() + QPointF(0, ALTO_CAJA / 2)
        flecha = _AristaOrigen(desde, hasta, colors)
        escena.addItem(flecha)
        aristas.append((a.desde, a.hasta, flecha))
    escena.setSceneRect(escena.itemsBoundingRect().adjusted(-30, -30, 30, 30))
    return escena, cajas, aristas


def render_imagen_grafo(grafo, colors, ancho_max_px=1800):
    """Dibuja el grafo en una imagen (para el informe de Narrativa y su PDF), sin
    mostrar nada en pantalla. Devuelve (QImage, ancho_natural, alto_natural): el
    tamaño natural es el del diagrama al 100 %; la imagen se genera al doble (o
    menos, si es muy ancha) para que se vea nítida al achicarla. Devuelve None si
    el grafo está vacío."""
    if grafo is None or not grafo.nodos:
        return None
    escena, cajas, aristas = _construir_escena(grafo, colors, lambda _id: None)
    rect = escena.itemsBoundingRect().adjusted(-16, -16, 16, 16)
    escala = min(2.0, ancho_max_px / rect.width())
    imagen = QImage(max(1, int(rect.width() * escala)), max(1, int(rect.height() * escala)),
                    QImage.Format_ARGB32)
    imagen.fill(QColor(colors["bg"]))
    pintor = QPainter(imagen)
    pintor.setRenderHint(QPainter.Antialiasing)
    escena.render(pintor, QRectF(0, 0, imagen.width(), imagen.height()), rect)
    pintor.end()
    return imagen, rect.width(), rect.height()


class _VistaOrigen(QGraphicsView):
    """QGraphicsView con zoom por rueda y arrastre del lienzo."""

    def __init__(self, escena, al_hacer_zoom_manual):
        super().__init__(escena)
        self._al_hacer_zoom_manual = al_hacer_zoom_manual
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFrameShape(QGraphicsView.NoFrame)

    def wheelEvent(self, event):
        paso = event.angleDelta().y()
        if paso == 0:
            return
        self.zoom(1.15 if paso > 0 else 1 / 1.15)
        event.accept()

    def zoom(self, factor):
        actual = self.transform().m11()
        nuevo = actual * factor
        if nuevo < 0.25 or nuevo > 2.5:
            return
        self.scale(factor, factor)
        self._al_hacer_zoom_manual()


class PanelOrigen(QWidget):
    """Sub-pestaña 'Origen': leyenda + lienzo + panel de detalle."""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self.colors = colors
        self._grafo = None
        self._cajas = {}       # id -> _CajaOrigen  (referencias duras, ver nota arriba)
        self._aristas = []     # [(desde, hasta, _AristaOrigen)]
        self._seleccion = None
        self._auto_ajuste = True
        self._anomalias = []   # [{"etiqueta": str, "columnas": [str]}] de la última Narrativa

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        barra = QHBoxLayout()
        self.lbl_leyenda = QLabel()
        self.lbl_leyenda.setTextFormat(Qt.RichText)
        self.lbl_leyenda.setWordWrap(True)
        barra.addWidget(self.lbl_leyenda, stretch=1)
        self.btn_menos = QPushButton("−")
        self.btn_mas = QPushButton("+")
        self.btn_ajustar = QPushButton("Ajustar")
        for b, tip in ((self.btn_menos, "Alejar"), (self.btn_mas, "Acercar"),
                       (self.btn_ajustar, "Ver todo el diagrama")):
            b.setToolTip(tip)
            b.setFixedHeight(26)
            barra.addWidget(b)
        self.btn_menos.setFixedWidth(30)
        self.btn_mas.setFixedWidth(30)
        self.btn_menos.clicked.connect(lambda: self.vista.zoom(1 / 1.25))
        self.btn_mas.clicked.connect(lambda: self.vista.zoom(1.25))
        self.btn_ajustar.clicked.connect(self.ajustar)
        layout.addLayout(barra)

        # "Ver desde una anomalía": la pregunta inversa a tocar una caja -- una columna
        # está mal, ¿qué la pudo causar y qué más se afecta?
        fila_anomalia = QHBoxLayout()
        lbl_anomalia = QLabel("Ver desde una anomalía:")
        lbl_anomalia.setObjectName("muted")
        fila_anomalia.addWidget(lbl_anomalia)
        self.combo_anomalias = QComboBox()
        self.combo_anomalias.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_anomalias.setMinimumContentsLength(28)
        self.combo_anomalias.currentIndexChanged.connect(self._on_anomalia_elegida)
        fila_anomalia.addWidget(self.combo_anomalias, stretch=1)
        layout.addLayout(fila_anomalia)

        self.escena = QGraphicsScene()
        self.vista = _VistaOrigen(self.escena, self._zoom_manual)
        layout.addWidget(self.vista, stretch=1)

        self.detalle = QTextBrowser()
        self.detalle.setOpenLinks(False)
        self.detalle.setMaximumHeight(210)
        self.detalle.setMinimumHeight(130)
        layout.addWidget(self.detalle)

        self.mostrar(None, colors)

    # ------------------------------------------------------------------ armado
    def mostrar(self, grafo, colors=None):
        """Dibuja `grafo` (GrafoOrigen o None). Se puede llamar las veces que
        haga falta: reconstruye todo con los colores del tema actual."""
        if colors is not None:
            self.colors = colors
        self._grafo = grafo
        self._seleccion = None
        self._auto_ajuste = True
        self.escena.clear()
        self._cajas = {}
        self._aristas = []
        self._reiniciar_combo_anomalias()
        self.vista.setBackgroundBrush(QBrush(QColor(self.colors["bg"])))
        self.detalle.setStyleSheet(
            f"QTextBrowser {{ background: {self.colors['card']}; color: {self.colors['text']}; "
            f"border: 1px solid {self.colors['border']}; border-radius: 6px; padding: 6px; }}"
        )

        if grafo is None or not grafo.nodos:
            self.lbl_leyenda.setText("")
            self.detalle.setHtml(self._html_mensaje(
                "Todavía no hay nada que mostrar. Carga datos y aquí verás de dónde vinieron, "
                "qué se les hizo y qué columnas calculaste."
            ))
            return

        _e, self._cajas, self._aristas = _construir_escena(
            grafo, self.colors, self._seleccionar, escena=self.escena
        )

        c = grafo.cuenta_por_grupo()
        partes = [
            (COLOR_ORIGEN, f"Origen ({c['origen']})"),
            (COLOR_PASOS, f"Pasos sobre los datos ({c['pasos']})"),
            (COLOR_CALCULADAS, f"Calculadas e indicadores ({c['calculadas']})"),
        ]
        if c["ojo"]:
            partes.append((COLOR_OJO, f"Ojo: conviene revisar ({c['ojo']})"))
        self.lbl_leyenda.setText("&nbsp;&nbsp;".join(
            f"<span style='color:{col}'>&#9632;</span> <span style='color:{self.colors['muted']}'>{txt}</span>"
            for col, txt in partes
        ))
        self.detalle.setHtml(self._html_mensaje(
            "Toca una caja para ver de dónde viene y a qué puede afectar. "
            "Rueda del mouse para acercar, arrastra el fondo para moverte."
        ))
        QTimer.singleShot(0, self._ajuste_inicial)

    # -------------------------------------------------------------- selección
    def _seleccionar(self, nid):
        if self._grafo is None:
            return
        self.combo_anomalias.blockSignals(True)
        self.combo_anomalias.setCurrentIndex(0)      # tocar una caja sale del modo "desde una anomalía"
        self.combo_anomalias.blockSignals(False)
        self._seleccion = None if nid == self._seleccion else nid
        actual = self._seleccion
        if actual:
            arriba = self._grafo.ancestros(actual)                    # de dónde viene (flechas)
            abajo = self._grafo.alcance_hacia_adelante(actual)        # a qué puede afectar
            # pasos que la dejaron con Ojo (influyeron, aunque no haya flecha hacia ella)
            afectada_por = self._grafo.afectada_por(actual)
            en_juego = set(arriba) | set(abajo) | set(afectada_por) | {actual}
        else:
            arriba = abajo = afectada_por = []
            en_juego = None
        for k, caja in self._cajas.items():
            caja.seleccionada = (k == actual)
            caja.setOpacity(1.0 if en_juego is None or k in en_juego else 0.3)
            caja.update()
        for desde, hasta, flecha in self._aristas:
            visible = en_juego is None or (desde in en_juego and hasta in en_juego)
            flecha.setOpacity(1.0 if visible else 0.2)
        if actual is None:
            self.detalle.setHtml(self._html_mensaje(
                "Toca una caja para ver de dónde viene y a qué puede afectar."
            ))
        else:
            self.detalle.setHtml(self._html_detalle(self._grafo.nodo(actual), arriba, abajo, afectada_por))

    # ------------------------------------------------------------- anomalías
    def set_anomalias(self, entradas):
        """entradas: [{'etiqueta': texto, 'columnas': [nombres]}] (una por anomalía o grupo
        de anomalías de la última Narrativa). Solo cuentan las que tienen columnas."""
        self._anomalias = [e for e in (entradas or []) if e.get("columnas")]
        self._reiniciar_combo_anomalias()

    def _reiniciar_combo_anomalias(self):
        self.combo_anomalias.blockSignals(True)
        self.combo_anomalias.clear()
        if self._anomalias:
            self.combo_anomalias.addItem("Elige una anomalía…")
            for e in self._anomalias:
                self.combo_anomalias.addItem(e["etiqueta"])
            self.combo_anomalias.setEnabled(True)
        else:
            self.combo_anomalias.addItem("Genera Narrativa para ver aquí las anomalías")
            self.combo_anomalias.setEnabled(False)
        self.combo_anomalias.setCurrentIndex(0)
        self.combo_anomalias.blockSignals(False)

    def _on_anomalia_elegida(self, indice):
        if self._grafo is None:
            return
        if indice <= 0:
            self._limpiar_resaltado()
            self.detalle.setHtml(self._html_mensaje(
                "Toca una caja para ver de dónde viene y a qué puede afectar."
            ))
            return
        entrada = self._anomalias[indice - 1]
        causas, afectados = self._grafo.impacto_anomalia(entrada["columnas"])
        self._seleccion = None
        en_juego = set(causas) | set(afectados)
        for k, caja in self._cajas.items():
            caja.seleccionada = False
            caja.setOpacity(1.0 if k in en_juego else 0.3)
            caja.update()
        for desde, hasta, flecha in self._aristas:
            flecha.setOpacity(1.0 if (desde in en_juego and hasta in en_juego) else 0.2)
        self.detalle.setHtml(self._html_anomalia(entrada, causas, afectados))

    def _limpiar_resaltado(self):
        self._seleccion = None
        for caja in self._cajas.values():
            caja.seleccionada = False
            caja.setOpacity(1.0)
            caja.update()
        for _d, _h, flecha in self._aristas:
            flecha.setOpacity(1.0)

    def _html_anomalia(self, entrada, causas, afectados):
        c = self.colors
        esc = html.escape
        cols = ", ".join(entrada["columnas"])
        partes = [
            f"<p style='margin:0 0 4px 0'><b style='font-size:13px'>{esc(entrada['etiqueta'])}</b></p>",
            f"<p style='margin:2px 0;color:{c['muted']}'>Columnas: {esc(cols)}.</p>",
        ]
        if causas:
            detalle = "; ".join(
                f"{esc(self._grafo.nodo(i).titulo)} ({esc(self._grafo.nodo(i).subtitulo)})" for i in causas
            )
            partes.append(f"<p style='margin:2px 0'><span style='color:{COLOR_OJO_TEXTO}'>Posible causa:</span> "
                          f"pasos que ya tocaron esos datos: {detalle}.</p>")
        else:
            partes.append(f"<p style='margin:2px 0;color:{c['muted']}'>Ningún paso registrado en Hadar tocó "
                          f"esos datos: el problema probablemente viene tal cual del origen.</p>")
        if afectados:
            partes.append(f"<p style='margin:2px 0'><span style='color:{c['muted']}'>Puede afectar a:</span> "
                          f"{esc(self._titulos(afectados))}. Si corriges la columna, conviene volver a crear "
                          f"las calculadas.</p>")
        else:
            partes.append(f"<p style='margin:2px 0;color:{c['muted']}'>No hay columnas calculadas ni "
                          f"indicadores que salgan de esas columnas.</p>")
        return f"<div style='color:{c['text']}'>" + "".join(partes) + "</div>"

    def _anomalias_de(self, nodo):
        """Etiquetas de las anomalías detectadas en columnas relacionadas con `nodo`."""
        cols = set(nodo.columnas)
        return [e["etiqueta"] for e in self._anomalias if cols & set(e["columnas"])]

    def _titulos(self, ids):
        return ", ".join(self._grafo.nodo(i).titulo for i in ids)

    def _html_detalle(self, nodo, arriba, abajo, afectada_por=()):
        c = self.colors
        esc = html.escape
        partes = [f"<p style='margin:0 0 4px 0'><b style='font-size:13px'>{esc(nodo.titulo)}</b></p>"]
        for linea in nodo.detalle:
            if linea.startswith("Ojo:"):
                partes.append(
                    f"<p style='margin:2px 0'><b style='color:{COLOR_OJO_TEXTO}'>Ojo:</b> "
                    f"<span style='color:{c['text']}'>{esc(linea[4:].strip())}</span></p>"
                )
            else:
                partes.append(f"<p style='margin:2px 0;color:{c['muted']}'>{esc(linea)}</p>")
        relacionadas = self._anomalias_de(nodo)
        if relacionadas:
            extra = f" y {len(relacionadas) - 4} más" if len(relacionadas) > 4 else ""
            partes.append(
                f"<p style='margin:2px 0;color:{c['muted']}'>Anomalías detectadas en sus columnas: "
                f"{esc('; '.join(relacionadas[:4]))}{extra}.</p>"
            )
        viene = esc(self._titulos(arriba)) if arriba else "es el punto de partida"
        afecta = esc(self._titulos(abajo)) if abajo else "no alimenta a nada más"
        partes.append(
            f"<p style='margin:6px 0 0 0'><span style='color:{c['muted']}'>Viene de:</span> {viene}. "
            f"<span style='color:{c['muted']}'>Puede afectar a:</span> {afecta}.</p>"
        )
        if afectada_por:
            partes.append(
                f"<p style='margin:2px 0 0 0'><span style='color:{COLOR_OJO_TEXTO}'>Después la afectaron:</span> "
                f"{esc(self._titulos(afectada_por))}.</p>"
            )
        return f"<div style='color:{c['text']}'>" + "".join(partes) + "</div>"

    def _html_mensaje(self, texto):
        return f"<p style='color:{self.colors['muted']}'>{html.escape(texto)}</p>"

    # ------------------------------------------------------------------- zoom
    def _zoom_manual(self):
        self._auto_ajuste = False

    def _ajuste_inicial(self):
        if self._auto_ajuste:
            self._encuadrar()

    def ajustar(self):
        self._auto_ajuste = True
        self._encuadrar()

    def _encuadrar(self):
        if not self._cajas:
            return
        rect = self.escena.itemsBoundingRect().adjusted(-24, -24, 24, 24)
        self.vista.fitInView(rect, Qt.KeepAspectRatio)
        zoom = self.vista.transform().m11()
        if zoom > 1.0:
            # un grafo chico no se agranda de más: se muestra al 100 %, centrado
            self.vista.resetTransform()
            self.vista.centerOn(rect.center())
        elif zoom < ZOOM_MINIMO_LEGIBLE:
            # un grafo enorme no se encoge hasta volverse ilegible: se deja a un zoom
            # que todavía se lee, empezando por el origen (a la izquierda); el resto
            # se recorre arrastrando o con la rueda
            self.vista.resetTransform()
            self.vista.scale(ZOOM_MINIMO_LEGIBLE, ZOOM_MINIMO_LEGIBLE)
            self.vista.centerOn(QPointF(rect.left() + self.vista.viewport().width() / (2 * ZOOM_MINIMO_LEGIBLE),
                                        rect.top() + self.vista.viewport().height() / (2 * ZOOM_MINIMO_LEGIBLE)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_ajuste and self._cajas:
            QTimer.singleShot(0, self._ajuste_inicial)