"""
Sub-pestaña "Linaje" de Narrativa: dibuja linaje_grafo.py (el árbol de
contagio.explorar_linaje() convertido a cajas) en vez de una lista con
sangría, con el mismo espíritu visual que origen_ui.py:
  - una frase arriba, en español llano, resume qué se está viendo (de dónde
    sale la información y qué depende de ella) sin que haga falta entender
    qué es una "relación de esquema";
  - las cajas de la izquierda ("antes", lo que explica el registro) y las de
    la derecha ("depende de él") llevan colores distintos, y las cajas se
    acomodan centradas respecto de sus propias cajas conectadas, para que
    las líneas se lean de un vistazo en vez de salir todas torcidas hacia
    abajo (ver linaje_grafo.construir_grafo_linaje);
  - cuando 3 o más registros de la misma tabla cuelgan del mismo lugar y
    ninguno tiene una anomalía, se muestran juntos en una sola caja ("40
    registros") en vez de repetir la misma caja muchas veces;
  - clic en una caja: resalta todo lo que se conecta con ella (en las dos
    direcciones) y atenúa el resto -- funciona como un filtro: tocar una
    caja del extremo izquierdo dice, en la práctica, "esto es lo único que
    explica esta caja" porque normalmente no tiene nada más a su izquierda;
  - clic derecho en una caja: "Explorar desde aquí" -- vuelve a armar todo
    el diagrama anclado en ESE registro;
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

from .linaje_grafo import resumen_en_lenguaje_simple

ANCHO_CAJA = 190
ALTO_CAJA = 64
ALTO_CABECERA = 24
SEPARACION_X = 56          # separación horizontal entre columnas a la DERECHA (lo que depende del registro)
SEPARACION_X_ANTES = 260   # separación horizontal entre columnas a la IZQUIERDA (el origen): más ancha
                           # a propósito, para que la trayectoria de cada flecha hacia la derecha se
                           # note de un vistazo en vez de quedar apretada contra el centro.
SEPARACION_Y = 26
ZOOM_MINIMO_LEGIBLE = 0.4

COLOR_ANTES = "#4338CA"         # cabecera de una caja "antes" (a la izquierda: explica al registro)
COLOR_DESPUES = "#0E7490"       # cabecera de una caja "depende de él" (a la derecha)
COLOR_RAIZ = "#B45309"          # cabecera de la caja desde la que se buscó
COLOR_ANOMALIA = "#B91C1C"      # punto que marca una fila con anomalía detectada
COLOR_TRUNCADO_TEXTO = "#B45309"


def _color_de(nodo):
    if nodo.es_raiz:
        return COLOR_RAIZ
    return COLOR_ANTES if nodo.capa < 0 else COLOR_DESPUES


def _x_de_capa(capa):
    """Posición horizontal de una columna del diagrama. A la izquierda (capa < 0,
    el origen) el paso entre columnas es más ancho que a la derecha -- ver
    SEPARACION_X_ANTES."""
    if capa >= 0:
        return capa * (ANCHO_CAJA + SEPARACION_X)
    return capa * (ANCHO_CAJA + SEPARACION_X_ANTES)


class _CajaLinaje(QGraphicsItem):
    """Una caja del diagrama: tabla (cabecera) + nombre legible o columna = valor
    (cuerpo). Un punto rojo marca una fila con anomalía detectada. Una caja de
    grupo (varios registros parecidos juntos) se dibuja con un pequeño efecto de
    'pila' detrás, para que se note que no es un solo registro. Se dibuja entera
    en paint() (sin hijos), igual que origen_ui._CajaOrigen."""

    def __init__(self, nodo, colors, al_hacer_clic, al_explorar_desde):
        super().__init__()
        self.nodo = nodo
        self.colors = colors
        self._al_hacer_clic = al_hacer_clic
        self._al_explorar_desde = al_explorar_desde
        self.seleccionada = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        if nodo.es_grupo:
            tip = f"{nodo.tabla}\n{nodo.valor}"
        else:
            tip = f"{nodo.tabla}\n" + (f"{nodo.nombre_legible}\n" if nodo.nombre_legible else "") \
                + f"{nodo.columna} = {nodo.valor}"
        if nodo.truncado:
            tip += "\n\nHay más registros conectados acá. Clic derecho para explorar desde esta caja."
        self.setToolTip(tip)
        self.setZValue(1)

    def boundingRect(self):
        extra = 6 if self.nodo.es_grupo else 0   # la 'pila' detrás sobresale un poco
        return QRectF(-3 - extra, -3 - extra, ANCHO_CAJA + 6 + extra, ALTO_CAJA + 6 + extra)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        c = self.colors
        color_cabecera = _color_de(self.nodo)

        if self.nodo.es_grupo:
            # dos rectángulos apenas corridos detrás de la caja principal: sugiere
            # que hay varios registros juntos, sin dibujar cada uno por separado.
            for dx in (6, 3):
                atras = QPainterPath()
                atras.addRoundedRect(QRectF(dx, dx, ANCHO_CAJA, ALTO_CAJA), 8, 8)
                painter.setPen(QPen(QColor(c["border"]), 1.0))
                painter.setBrush(QBrush(QColor(c["card"]).lighter(106)))
                painter.drawPath(atras)

        rect = QRectF(0, 0, ANCHO_CAJA, ALTO_CAJA)
        forma = QPainterPath()
        forma.addRoundedRect(rect, 8, 8)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(c["card"])))
        painter.drawPath(forma)
        painter.save()
        painter.setClipPath(forma)
        painter.fillRect(QRectF(0, 0, ANCHO_CAJA, ALTO_CABECERA), QColor(color_cabecera))
        painter.restore()

        grosor = 2.4 if self.seleccionada else 1.0
        color_borde = c["text"] if self.seleccionada else c["border"]
        painter.setPen(QPen(QColor(color_borde), grosor))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(forma)

        fuente_titulo = QFont()
        fuente_titulo.setBold(True)
        fuente_titulo.setPointSizeF(10)
        painter.setFont(fuente_titulo)
        painter.setPen(QColor("#FFFFFF"))
        ancho_titulo = ANCHO_CAJA - 24 - (18 if self.nodo.truncado else 0)
        titulo = QFontMetrics(fuente_titulo).elidedText(self.nodo.tabla, Qt.ElideRight, ancho_titulo)
        painter.drawText(QRectF(12, 0, ancho_titulo, ALTO_CABECERA), Qt.AlignVCenter | Qt.AlignLeft, titulo)

        # Cuerpo: el nombre legible (o el conteo, si es un grupo) en letra normal,
        # y debajo -- más chico -- la columna = valor cruda, siempre disponible.
        principal = self.nodo.valor.split(" con ")[0] if self.nodo.es_grupo else (
            self.nodo.nombre_legible or f"{self.nodo.columna} = {self.nodo.valor}"
        )
        fuente_ppal = QFont()
        fuente_ppal.setPointSizeF(9.5)
        if self.nodo.nombre_legible or self.nodo.es_grupo:
            fuente_ppal.setBold(True)
        painter.setFont(fuente_ppal)
        painter.setPen(QColor(c["text"]))
        principal = QFontMetrics(fuente_ppal).elidedText(principal, Qt.ElideRight, ANCHO_CAJA - 22)
        y_cuerpo = ALTO_CABECERA + 3
        alto_linea = 18
        painter.drawText(QRectF(12, y_cuerpo, ANCHO_CAJA - 22, alto_linea),
                         Qt.AlignVCenter | Qt.AlignLeft, principal)

        if self.nodo.nombre_legible or self.nodo.es_grupo:
            fuente_sec = QFont()
            fuente_sec.setPointSizeF(8)
            painter.setFont(fuente_sec)
            painter.setPen(QColor(c["muted"]))
            secundaria = (f"{self.nodo.columna} = {self.nodo.valor}" if not self.nodo.es_grupo
                         else f"{self.nodo.columna} = {self.nodo.valor.split('= ')[-1]}")
            secundaria = QFontMetrics(fuente_sec).elidedText(secundaria, Qt.ElideRight, ANCHO_CAJA - 22)
            painter.drawText(QRectF(12, y_cuerpo + alto_linea, ANCHO_CAJA - 22, ALTO_CAJA - ALTO_CABECERA - alto_linea),
                             Qt.AlignVCenter | Qt.AlignLeft, secundaria)

        if self.nodo.es_anomalia:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(COLOR_ANOMALIA))
            painter.drawEllipse(QRectF(ANCHO_CAJA - 32, 6, 11, 11))
        if self.nodo.truncado:
            fuente_t = QFont()
            fuente_t.setBold(True)
            fuente_t.setPointSizeF(12)
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
        if self.nodo.es_grupo:
            accion_info = menu.addAction("Esta caja junta varios registros -- busca uno arriba, con su ID, para explorar desde ahí")
            accion_info.setEnabled(False)
            menu.exec(event.screenPos())
        else:
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
    """Sub-pestaña 'Linaje': frase resumen + cajas conectadas + panel de detalle.
    `al_explorar_desde` es una función (tabla, columna, valor) -> None que el panel
    llama cuando el usuario elige 'Explorar desde aquí' en una caja; quien la
    construye decide qué hacer con eso (re-buscar y volver a llamar a mostrar())."""

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

        self.lbl_resumen = QLabel()
        self.lbl_resumen.setWordWrap(True)
        self.lbl_resumen.setContentsMargins(0, 0, 0, 4)
        layout.addWidget(self.lbl_resumen)

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
        self.lbl_resumen.setStyleSheet(f"color: {self.colors['text']}; font-size: 13px;")

        if grafo is None or not grafo.nodos:
            self.lbl_resumen.setText("")
            self.lbl_leyenda.setText("")
            self.detalle.setHtml(self._html_mensaje(
                mensaje_vacio or "Elige una tabla y un ID, o doble clic en una anomalía, para ver "
                "aquí los registros conectados de otras tablas."
            ))
            return

        self.lbl_resumen.setText(resumen_en_lenguaje_simple(grafo))

        for n in grafo.nodos:
            caja = _CajaLinaje(n, self.colors, self._seleccionar, self._al_explorar_desde)
            caja.setPos(_x_de_capa(n.capa), n.fila * (ALTO_CAJA + SEPARACION_Y))
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
        n_grupo = sum(1 for n in grafo.nodos if n.es_grupo)
        partes = [
            (COLOR_RAIZ, "Desde aquí se buscó"),
            (COLOR_ANTES, "Antes (explica al registro)"),
            (COLOR_DESPUES, "Depende de él"),
        ]
        if n_anom:
            partes.append((COLOR_ANOMALIA, f"Con anomalía detectada ({n_anom})"))
        leyenda = "&nbsp;&nbsp;".join(
            f"<span style='color:{col}'>&#9632;</span> <span style='color:{self.colors['muted']}'>{txt}</span>"
            for col, txt in partes
        )
        extra = []
        if n_grupo:
            extra.append(f"varios registros parecidos se juntaron en {n_grupo} caja(s)")
        if n_trunc:
            extra.append(f"{n_trunc} caja(s) tienen más conectados de los que se muestran (clic derecho para explorar)")
        if extra:
            leyenda += f"&nbsp;&nbsp;<span style='color:{self.colors['muted']}'>· {'; '.join(extra)}</span>"
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

    def _etiqueta_nodo(self, nodo):
        if nodo.es_grupo:
            return f"{nodo.tabla} ({nodo.valor.split(' con ')[0]})"
        principal = nodo.nombre_legible or f"{nodo.columna} = {nodo.valor}"
        return f"{nodo.tabla} ({principal})"

    def _etiquetas(self, ids):
        return ", ".join(self._etiqueta_nodo(self._grafo.nodo(i)) for i in ids)

    def _html_detalle(self, nodo, arriba, abajo):
        c = self.colors
        esc = html.escape
        titulo = esc(nodo.valor) if nodo.es_grupo else esc(nodo.nombre_legible or f"{nodo.columna} = {nodo.valor}")
        partes = [f"<p style='margin:0 0 4px 0'><b style='font-size:13px'>{esc(nodo.tabla)}</b> · {titulo}</p>"]
        if not nodo.es_grupo and nodo.nombre_legible:
            partes.append(f"<p style='margin:2px 0;color:{c['muted']}'>{esc(nodo.columna)} = {esc(nodo.valor)}</p>")
        if nodo.es_grupo:
            listados = ", ".join(
                (nom or val) for val, nom in nodo.miembros[:8]
            )
            extra = f" y {len(nodo.miembros) - 8} más" if len(nodo.miembros) > 8 else ""
            partes.append(f"<p style='margin:2px 0;color:{c['muted']}'>Junta {len(nodo.miembros)} registros: "
                          f"{esc(listados)}{esc(extra)}.</p>")
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