"""
Sub-pestaña "Linaje" de Narrativa: dibuja linaje_grafo.py (el árbol de
contagio.explorar_linaje() convertido a cajas) en vez de una lista con
sangría, con el mismo espíritu visual que origen_ui.py:
  - clic en una caja: resalta de qué tablas sale y a cuáles alimenta,
    y el detalle se muestra abajo;
  - clic derecho en una caja: "Explorar desde aquí" -- vuelve a armar todo
    el diagrama anclado en ESE registro (mismo comportamiento que ya tenía
    Linaje en árbol, ahora desde una caja en vez de un ítem de lista);
  - zoom con la rueda / botones, arrastrar para moverse, "Ajustar" encuadra.

Es solo lectura salvo por el clic derecho, que dispara una NUEVA búsqueda
(vía el callback `al_explorar_desde` que recibe el panel) -- no edita nada.
"""
import html

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QBrush, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QMenu, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget,
)

ANCHO_CAJA = 168
ALTO_CAJA = 56
ALTO_CABECERA = 22
SEPARACION_X = 46
SEPARACION_Y = 20
ZOOM_MINIMO_LEGIBLE = 0.4

COLOR_CAJA = "#0E7490"          # cabecera de una caja normal
COLOR_ANOMALIA = "#B91C1C"      # punto que marca una fila con anomalía detectada
COLOR_TRUNCADO_TEXTO = "#B45309"
COLOR_RAIZ = "#F59E0B"          # franja a la izquierda de la caja desde la que se buscó


class _CajaLinaje(QGraphicsItem):
    """Una caja del diagrama: tabla (cabecera) + columna = valor (cuerpo).
    Un punto rojo marca una fila con anomalía detectada; una franja a la
    izquierda marca la caja desde la que se hizo la búsqueda. Se dibuja
    entera en paint() (sin hijos), igual que origen_ui._CajaOrigen."""

    def __init__(self, nodo, colors, al_hacer_clic, al_explorar_desde):
        super().__init__()
        self.nodo = nodo
        self.colors = colors
        self._al_hacer_clic = al_hacer_clic
        self._al_explorar_desde = al_explorar_desde
        self.seleccionada = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        tip = f"{nodo.tabla}\n{nodo.columna} = {nodo.valor}"
        if nodo.truncado:
            tip += "\n\nHay más registros conectados acá. Clic derecho para explorar desde esta caja."
        self.setToolTip(tip)
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
        painter.fillRect(QRectF(0, 0, ANCHO_CAJA, ALTO_CABECERA), QColor(COLOR_CAJA))
        if self.nodo.es_raiz:
            painter.fillRect(QRectF(0, 0, 4, ALTO_CAJA), QColor(COLOR_RAIZ))
        painter.restore()

        grosor = 2.2 if self.seleccionada else 1.0
        color_borde = c["text"] if self.seleccionada else c["border"]
        painter.setPen(QPen(QColor(color_borde), grosor))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(forma)

        fuente_titulo = QFont()
        fuente_titulo.setBold(True)
        fuente_titulo.setPointSizeF(9.5)
        painter.setFont(fuente_titulo)
        painter.setPen(QColor("#FFFFFF"))
        ancho_titulo = ANCHO_CAJA - 24 - (16 if self.nodo.truncado else 0)
        titulo = QFontMetrics(fuente_titulo).elidedText(self.nodo.tabla, Qt.ElideRight, ancho_titulo)
        painter.drawText(QRectF(12, 0, ancho_titulo, ALTO_CABECERA), Qt.AlignVCenter | Qt.AlignLeft, titulo)

        fuente_sub = QFont()
        fuente_sub.setPointSizeF(8.5)
        painter.setFont(fuente_sub)
        painter.setPen(QColor(c["muted"]))
        sub = f"{self.nodo.columna} = {self.nodo.valor}"
        sub = QFontMetrics(fuente_sub).elidedText(sub, Qt.ElideRight, ANCHO_CAJA - 22)
        painter.drawText(QRectF(12, ALTO_CABECERA, ANCHO_CAJA - 22, ALTO_CAJA - ALTO_CABECERA),
                         Qt.AlignVCenter | Qt.AlignLeft, sub)

        if self.nodo.es_anomalia:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(COLOR_ANOMALIA))
            painter.drawEllipse(QRectF(ANCHO_CAJA - 30, 6, 10, 10))
        if self.nodo.truncado:
            fuente_t = QFont()
            fuente_t.setBold(True)
            fuente_t.setPointSizeF(11)
            painter.setFont(fuente_t)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(QRectF(ANCHO_CAJA - 20, 0, 16, ALTO_CABECERA), Qt.AlignCenter, "…")

    def mousePressEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._al_hacer_clic(self.nodo.id)
        event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(event.widget())
        accion = menu.addAction(f'Explorar desde aquí ("{self.nodo.tabla}": {self.nodo.columna} = {self.nodo.valor})')
        elegida = menu.exec(event.screenPos())
        if elegida == accion:
            self._al_explorar_desde(self.nodo.tabla, self.nodo.columna, self.nodo.valor)
        event.accept()


class _AristaLinaje(QGraphicsItem):
    """Flecha curva de una caja a la siguiente (sale por la derecha, entra por la izquierda).
    Idéntica en trazo a origen_ui._AristaOrigen; se duplica en vez de importarse para que
    los dos diagramas puedan evolucionar por separado sin acoplarse."""

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


class _VistaLinaje(QGraphicsView):
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


class PanelLinaje(QWidget):
    """Sub-pestaña 'Linaje': cajas conectadas + panel de detalle. `al_explorar_desde`
    es una función (tabla, columna, valor) -> None que el panel llama cuando el
    usuario elige 'Explorar desde aquí' en una caja; quien la construye decide qué
    hacer con eso (re-buscar y volver a llamar a mostrar())."""

    def __init__(self, colors, al_explorar_desde, parent=None):
        super().__init__(parent)
        self.colors = colors
        self._al_explorar_desde = al_explorar_desde
        self._grafo = None
        self._cajas = {}
        self._aristas = []
        self._seleccion = None
        self._auto_ajuste = True

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

        self.escena = QGraphicsScene()
        self.vista = _VistaLinaje(self.escena, self._zoom_manual)
        layout.addWidget(self.vista, stretch=1)

        self.detalle = QTextBrowser()
        self.detalle.setOpenLinks(False)
        self.detalle.setMaximumHeight(150)
        self.detalle.setMinimumHeight(90)
        layout.addWidget(self.detalle)

        self.mostrar(None, colors)

    # ------------------------------------------------------------------ armado
    def mostrar(self, grafo, colors=None, mensaje_vacio=None):
        if colors is not None:
            self.colors = colors
        self._grafo = grafo
        self._seleccion = None
        self._auto_ajuste = True
        self.escena.clear()
        self._cajas = {}
        self._aristas = []
        self.vista.setBackgroundBrush(QBrush(QColor(self.colors["bg"])))
        self.detalle.setStyleSheet(
            f"QTextBrowser {{ background: {self.colors['card']}; color: {self.colors['text']}; "
            f"border: 1px solid {self.colors['border']}; border-radius: 6px; padding: 6px; }}"
        )

        if grafo is None or not grafo.nodos:
            self.lbl_leyenda.setText("")
            self.detalle.setHtml(self._html_mensaje(
                mensaje_vacio or "Elige una tabla y un ID, o doble clic en una anomalía, para ver "
                "aquí los registros conectados de otras tablas."
            ))
            return

        for n in grafo.nodos:
            caja = _CajaLinaje(n, self.colors, self._seleccionar, self._al_explorar_desde)
            caja.setPos(n.capa * (ANCHO_CAJA + SEPARACION_X), n.fila * (ALTO_CAJA + SEPARACION_Y))
            self.escena.addItem(caja)
            self._cajas[n.id] = caja
        for a in grafo.aristas:
            ca, cb = self._cajas[a.desde], self._cajas[a.hasta]
            desde = ca.pos() + QPointF(ANCHO_CAJA, ALTO_CAJA / 2)
            hasta = cb.pos() + QPointF(0, ALTO_CAJA / 2)
            flecha = _AristaLinaje(desde, hasta, self.colors)
            self.escena.addItem(flecha)
            self._aristas.append((a.desde, a.hasta, flecha))
        self.escena.setSceneRect(self.escena.itemsBoundingRect().adjusted(-30, -30, 30, 30))

        n_anom = sum(1 for n in grafo.nodos if n.es_anomalia)
        n_trunc = sum(1 for n in grafo.nodos if n.truncado)
        partes = [(COLOR_CAJA, f"Registro ({len(grafo.nodos)})"),
                  (COLOR_RAIZ, "Desde donde se buscó")]
        if n_anom:
            partes.append((COLOR_ANOMALIA, f"Con anomalía detectada ({n_anom})"))
        leyenda = "&nbsp;&nbsp;".join(
            f"<span style='color:{col}'>&#9632;</span> <span style='color:{self.colors['muted']}'>{txt}</span>"
            for col, txt in partes
        )
        if n_trunc:
            leyenda += (f"&nbsp;&nbsp;<span style='color:{self.colors['muted']}'>… = hay más conectados "
                       f"({n_trunc}); clic derecho para explorar desde ahí</span>")
        self.lbl_leyenda.setText(leyenda)
        self.detalle.setHtml(self._html_mensaje(
            "Toca una caja para ver con qué se conecta. Clic derecho: explorar desde ahí."
        ))
        QTimer.singleShot(0, self._ajuste_inicial)

    # -------------------------------------------------------------- selección
    def _seleccionar(self, nid):
        if self._grafo is None:
            return
        self._seleccion = None if nid == self._seleccion else nid
        actual = self._seleccion
        if actual:
            arriba = self._grafo.ancestros(actual)
            abajo = self._grafo.descendientes(actual)
            en_juego = set(arriba) | set(abajo) | {actual}
        else:
            arriba = abajo = []
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
                "Toca una caja para ver con qué se conecta. Clic derecho: explorar desde ahí."
            ))
        else:
            self.detalle.setHtml(self._html_detalle(self._grafo.nodo(actual), arriba, abajo))

    def _etiquetas(self, ids):
        return ", ".join(f"{self._grafo.nodo(i).tabla} ({self._grafo.nodo(i).columna} = "
                         f"{self._grafo.nodo(i).valor})" for i in ids)

    def _html_detalle(self, nodo, arriba, abajo):
        c = self.colors
        esc = html.escape
        partes = [f"<p style='margin:0 0 4px 0'><b style='font-size:13px'>{esc(nodo.tabla)}</b> · "
                  f"{esc(nodo.columna)} = {esc(nodo.valor)}</p>"]
        if nodo.es_raiz:
            partes.append(f"<p style='margin:2px 0;color:{COLOR_RAIZ}'>Este es el registro desde el que "
                          f"se hizo la búsqueda.</p>")
        if nodo.es_anomalia:
            partes.append(f"<p style='margin:2px 0;color:{COLOR_ANOMALIA}'>Esta fila tiene una anomalía "
                          f"detectada.</p>")
        if nodo.truncado:
            partes.append(f"<p style='margin:2px 0;color:{COLOR_TRUNCADO_TEXTO}'>Hay más registros "
                          f"conectados acá de los que se muestran. Clic derecho para explorar desde "
                          f"esta caja.</p>")
        viene = esc(self._etiquetas(arriba)) if arriba else "no depende de ninguna otra tabla mostrada"
        afecta = esc(self._etiquetas(abajo)) if abajo else "ninguna otra tabla mostrada depende de este"
        partes.append(
            f"<p style='margin:6px 0 0 0'><span style='color:{c['muted']}'>Sale de:</span> {viene}. "
            f"<span style='color:{c['muted']}'>Lo referencian:</span> {afecta}.</p>"
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
            self.vista.resetTransform()
            self.vista.centerOn(rect.center())
        elif zoom < ZOOM_MINIMO_LEGIBLE:
            self.vista.resetTransform()
            self.vista.scale(ZOOM_MINIMO_LEGIBLE, ZOOM_MINIMO_LEGIBLE)
            self.vista.centerOn(QPointF(rect.left() + self.vista.viewport().width() / (2 * ZOOM_MINIMO_LEGIBLE),
                                        rect.top() + self.vista.viewport().height() / (2 * ZOOM_MINIMO_LEGIBLE)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_ajuste and self._cajas:
            QTimer.singleShot(0, self._ajuste_inicial)
