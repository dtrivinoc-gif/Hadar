"""
ontologia_ui.py — Ventana del "Esquema sugerido" (diagrama de tablas y
relaciones) para Hadar.

Qué es esto, en simple:
  El usuario carga 2+ tablas (ej. Clientes y Pedidos). Esta ventana le
  muestra una cajita por tabla, con sus columnas adentro, y líneas que
  conectan las columnas que Hadar cree que están relacionadas (calculado
  por ontologia.inferir_relaciones). El usuario puede:
    - arrastrar las cajitas para acomodarlas a su gusto,
    - borrar una línea que esté mal (clic derecho -> Eliminar),
    - agregar una relación a mano si Hadar no la encontró sola.

  Nada de esto obliga al usuario a saber de bases de datos: si no toca
  nada, el esquema sugerido automático ya queda funcionando.

Este módulo no toca el DataFrame activo de la app (self.df en
HadarApp) ni ninguna otra pestaña — es autocontenido. La ventana
principal solo necesita: crear el diálogo pasándole las tablas y el
tema de colores, mostrarlo, y (más adelante, en el Paso C) leer
`dialogo.relaciones` para cruzar información en Narrativa.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem,
    QGraphicsTextItem, QGraphicsPathItem, QMenu, QMessageBox, QFrame,
    QDialogButtonBox, QToolTip,
)

from .config import COLOR_ACCENT, COLOR_ACCENT_2, COLOR_ACCENT_3, COLOR_DANGER
from .ontologia import inferir_relaciones, RelacionSugerida


# ----------------------------------------------------------------------
# Medidas del diagrama (en píxeles de la escena)
# ----------------------------------------------------------------------
ANCHO_CAJA = 220
ALTO_FILA_COLUMNA = 26
ALTO_TITULO = 40
ESPACIO_ENTRE_CAJAS_X = 90
ESPACIO_ENTRE_CAJAS_X_MAX_POR_FILA = 3   # cuántas cajas por fila antes de bajar
ESPACIO_ENTRE_CAJAS_Y = 60
MAX_COLUMNAS_VISIBLES = 14  # si una tabla tiene más columnas, se recorta la vista (con "+N más")


# ----------------------------------------------------------------------
# Cajita de una tabla
# ----------------------------------------------------------------------

class _CajaTabla(QGraphicsRectItem):
    """Representa una tabla en el diagrama: título + lista de columnas.
    Se puede arrastrar; cuando se mueve, avisa a las líneas conectadas
    para que se redibujen."""

    def __init__(self, nombre_tabla: str, columnas: list[str], colors: dict):
        columnas_visibles = columnas[:MAX_COLUMNAS_VISIBLES]
        self._hay_columnas_ocultas = len(columnas) > MAX_COLUMNAS_VISIBLES
        n_filas = len(columnas_visibles) + (1 if self._hay_columnas_ocultas else 0)
        alto = ALTO_TITULO + n_filas * ALTO_FILA_COLUMNA + 10

        super().__init__(0, 0, ANCHO_CAJA, alto)
        self.nombre_tabla = nombre_tabla
        self.columnas = columnas
        self.colors = colors
        self.lineas_conectadas: list["_LineaRelacion"] = []
        self._filas_columna: dict[str, QRectF] = {}

        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setZValue(1)
        self.setPen(QPen(QColor(colors["border"]), 1.5))
        self.setBrush(QBrush(QColor(colors["card"])))
        self.setCursor(Qt.OpenHandCursor)

        # OJO: PySide6 no protege automáticamente de la basura (garbage
        # collector) de Python a los QGraphicsItem hijos solo por tener un
        # "padre gráfico" — hay que guardar una referencia dura acá, o los
        # textos/puntos desaparecen del diagrama sin avisar ni dar error.
        self._items_hijos: list = []

        # --- título ---
        titulo = QGraphicsTextItem(nombre_tabla, self)
        self._items_hijos.append(titulo)
        fuente_titulo = QFont()
        fuente_titulo.setBold(True)
        fuente_titulo.setPointSize(11)
        titulo.setFont(fuente_titulo)
        titulo.setDefaultTextColor(QColor(COLOR_ACCENT))
        titulo.setPos(10, 8)

        subtitulo = QGraphicsTextItem(f"{len(columnas)} columna(s)", self)
        self._items_hijos.append(subtitulo)
        fuente_sub = QFont()
        fuente_sub.setPointSize(8)
        subtitulo.setFont(fuente_sub)
        subtitulo.setDefaultTextColor(QColor(colors["muted"]))
        subtitulo.setPos(10, 24)

        # línea separadora bajo el título
        separador = QGraphicsRectItem(0, ALTO_TITULO, ANCHO_CAJA, 1, self)
        separador.setPen(QPen(QColor(colors["border"])))
        separador.setBrush(QBrush(QColor(colors["border"])))
        self._items_hijos.append(separador)

        # --- filas de columnas ---
        y = ALTO_TITULO + 5
        for col in columnas_visibles:
            texto = QGraphicsTextItem(str(col), self)
            self._items_hijos.append(texto)
            fuente_col = QFont()
            fuente_col.setPointSize(9)
            texto.setFont(fuente_col)
            texto.setDefaultTextColor(QColor(colors["text"]))
            texto.setPos(18, y)
            # recorta nombres muy largos para que no se salgan de la caja
            metrica = texto.font()
            if texto.boundingRect().width() > ANCHO_CAJA - 30:
                texto.setPlainText(str(col)[:22] + "…")
            self._filas_columna[col] = QRectF(0, y, ANCHO_CAJA, ALTO_FILA_COLUMNA)
            y += ALTO_FILA_COLUMNA

        if self._hay_columnas_ocultas:
            n_ocultas = len(columnas) - len(columnas_visibles)
            extra = QGraphicsTextItem(f"… +{n_ocultas} columna(s) más", self)
            self._items_hijos.append(extra)
            fuente_extra = QFont()
            fuente_extra.setPointSize(8)
            fuente_extra.setItalic(True)
            extra.setFont(fuente_extra)
            extra.setDefaultTextColor(QColor(colors["muted"]))
            extra.setPos(18, y)

    # -- puntos de anclaje para las líneas --------------------------------

    def tiene_columna_visible(self, nombre_columna: str) -> bool:
        return nombre_columna in self._filas_columna

    def punto_ancla(self, nombre_columna: str, lado: str) -> QPointF:
        """
        Punto (en coordenadas de ESCENA) donde debe llegar/salir una línea
        para esta columna. Si la columna no está entre las visibles (tabla
        con muchas columnas, recortada), ancla al título de la caja.
        lado: 'izquierda' o 'derecha'.
        """
        fila = self._filas_columna.get(nombre_columna)
        if fila is None:
            y_local = ALTO_TITULO / 2
        else:
            y_local = fila.top() + fila.height() / 2
        x_local = 0 if lado == "izquierda" else ANCHO_CAJA
        return self.mapToScene(QPointF(x_local, y_local))

    def resaltar_columna(self, nombre_columna: str, color: QColor):
        """Dibuja un puntito de color junto a una columna involucrada en
        una relación (pista visual rápida, sin tener que seguir la línea)."""
        fila = self._filas_columna.get(nombre_columna)
        if fila is None:
            return
        punto = QGraphicsRectItem(6, fila.top() + fila.height() / 2 - 3, 6, 6, self)
        punto.setBrush(QBrush(color))
        punto.setPen(QPen(Qt.NoPen))
        self._items_hijos.append(punto)

    # -- mover la caja actualiza las líneas --------------------------------

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for linea in self.lineas_conectadas:
                linea.actualizar_geometria()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)


# ----------------------------------------------------------------------
# Línea que conecta dos columnas
# ----------------------------------------------------------------------

class _LineaRelacion(QGraphicsPathItem):
    """Curva entre la columna de una tabla y la de otra. Se puede eliminar
    con clic derecho. El grosor/color reflejan la confianza."""

    def __init__(self, relacion: RelacionSugerida, caja_origen: _CajaTabla,
                 caja_destino: _CajaTabla, colors: dict, on_eliminar, on_confirmar_direccion=None):
        super().__init__()
        self.relacion = relacion
        self.caja_origen = caja_origen
        self.caja_destino = caja_destino
        self.colors = colors
        self._on_eliminar = on_eliminar
        self._on_confirmar_direccion = on_confirmar_direccion

        self.setZValue(0)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)

        caja_origen.lineas_conectadas.append(self)
        caja_destino.lineas_conectadas.append(self)
        self.actualizar_geometria()

    def _color_por_confianza(self) -> QColor:
        if getattr(self.relacion, "manual", False):
            return QColor(COLOR_ACCENT_3)   # relación agregada a mano por el usuario
        if getattr(self.relacion, "certeza_direccion", "alta") == "sin_definir":
            return QColor(COLOR_DANGER)     # dirección padre/hijo sin confirmar: salta a la vista
        c = self.relacion.confianza
        if c >= 0.75:
            return QColor(COLOR_ACCENT)
        if c >= 0.5:
            return QColor(COLOR_ACCENT_2)
        return QColor(self.colors["muted"])

    def actualizar_geometria(self):
        r = self.relacion
        p1 = self.caja_origen.punto_ancla(r.columna_origen, "derecha")
        p2 = self.caja_destino.punto_ancla(r.columna_destino, "izquierda")

        # si la caja destino terminó quedando a la izquierda de la origen,
        # usamos los bordes contrarios para que la línea no cruce por
        # ENCIMA de las cajas de forma antiestética
        if p2.x() < p1.x():
            p1 = self.caja_origen.punto_ancla(r.columna_origen, "izquierda")
            p2 = self.caja_destino.punto_ancla(r.columna_destino, "derecha")

        camino = QPainterPath(p1)
        dx = max(abs(p2.x() - p1.x()) * 0.5, 40)
        ctrl1 = QPointF(p1.x() + dx if p2.x() >= p1.x() else p1.x() - dx, p1.y())
        ctrl2 = QPointF(p2.x() - dx if p2.x() >= p1.x() else p2.x() + dx, p2.y())
        camino.cubicTo(ctrl1, ctrl2, p2)
        self.setPath(camino)

        color = self._color_por_confianza()
        grosor = 1.0 + 2.5 * r.confianza
        pen = QPen(color, grosor)
        pen.setCapStyle(Qt.RoundCap)
        if getattr(r, "manual", False):
            pen.setStyle(Qt.DashLine)
        elif getattr(r, "certeza_direccion", "alta") == "sin_definir":
            pen.setStyle(Qt.DotLine)
        self.setPen(pen)

    def hoverEnterEvent(self, event):
        etiqueta = getattr(self.relacion, "manual", False)
        certeza_direccion = getattr(self.relacion, "certeza_direccion", "alta")
        if etiqueta:
            linea_direccion = "Agregada manualmente"
        else:
            linea_direccion = f"{int(self.relacion.confianza*100)}% de confianza"
        aviso_direccion = ""
        if not etiqueta and certeza_direccion == "sin_definir":
            aviso_direccion = "\n⚠ No está claro cuál tabla es la principal — clic derecho para elegirla"
        self._tooltip_texto = (
            f"{self.relacion.tabla_origen}.{self.relacion.columna_origen} <-> "
            f"{self.relacion.tabla_destino}.{self.relacion.columna_destino}\n"
            f"{linea_direccion}\n"
            f"{self.relacion.razon}{aviso_direccion}\n(clic derecho para más opciones)"
        )
        QToolTip.showText(event.screenPos(), self._tooltip_texto)
        pen = self.pen()
        pen.setWidthF(pen.widthF() + 1.5)
        self.setPen(pen)
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        # QToolTip.showText() se esconde solo a los pocos segundos y no se
        # reinicia con el mouse quieto encima -- por eso, mientras el mouse
        # se siga moviendo (aunque sea un poco) dentro de la línea, se
        # vuelve a pedir que se muestre, lo que le renueva el tiempo. Así
        # el tooltip dura mientras el usuario lo esté leyendo de verdad.
        texto = getattr(self, "_tooltip_texto", None)
        if texto:
            QToolTip.showText(event.screenPos(), texto)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.actualizar_geometria()
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event):
        r = self.relacion
        menu = QMenu()
        accion_eliminar = menu.addAction("Eliminar esta relación")

        accion_principal_origen = None
        accion_principal_destino = None
        if not getattr(r, "manual", False) and self._on_confirmar_direccion is not None:
            menu.addSeparator()
            marca_o = "  ✓" if r.tabla_principal == r.tabla_origen else ""
            marca_d = "  ✓" if r.tabla_principal == r.tabla_destino else ""
            accion_principal_origen = menu.addAction(f'"{r.tabla_origen}" es la tabla principal{marca_o}')
            accion_principal_destino = menu.addAction(f'"{r.tabla_destino}" es la tabla principal{marca_d}')

        elegida = menu.exec(event.screenPos())
        if elegida == accion_eliminar:
            self._on_eliminar(self)
        elif elegida is not None and elegida == accion_principal_origen:
            self._on_confirmar_direccion(self, r.tabla_origen, r.columna_origen)
        elif elegida is not None and elegida == accion_principal_destino:
            self._on_confirmar_direccion(self, r.tabla_destino, r.columna_destino)


# ----------------------------------------------------------------------
# Diálogo principal
# ----------------------------------------------------------------------

class DialogoEsquemaOntologia(QDialog):
    """
    Ventana del esquema sugerido. Uso típico desde HadarApp:

        dialogo = DialogoEsquemaOntologia(self.tablas, self.colors, self)
        dialogo.exec()
        # más adelante (Paso C): dialogo.relaciones tiene la lista final
        # (con lo que el usuario haya agregado/borrado a mano).
    """

    def __init__(self, tablas: dict, colors: dict, parent=None, relaciones_iniciales=None):
        super().__init__(parent)
        self.tablas = tablas
        self.colors = colors
        self.setWindowTitle("Esquema sugerido — Hadar")
        self.resize(1000, 640)
        self.setModal(True)

        # Si ya había un esquema confirmado antes (relaciones_iniciales), se
        # parte de ahí -- así reabrir este diálogo no borra las relaciones
        # que el usuario agregó, eliminó o corrigió la última vez. Solo se
        # vuelve a detectar todo desde cero la primerísima vez que se abre.
        if relaciones_iniciales is not None:
            self.relaciones: list[RelacionSugerida] = list(relaciones_iniciales)
        else:
            self.relaciones: list[RelacionSugerida] = inferir_relaciones(tablas)
        self._cajas: dict[str, _CajaTabla] = {}
        self._lineas: list[_LineaRelacion] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- encabezado ---
        titulo = QLabel("Esquema sugerido entre tus tablas")
        titulo.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {colors['text']};")
        layout.addWidget(titulo)

        ayuda = QLabel(
            "Hadar detectó estas conexiones solo. Arrastra las tablas para acomodarlas. "
            "Clic derecho sobre una línea para borrarla si está mal. "
            "Si falta una relación, agrégala con el botón de abajo."
        )
        ayuda.setWordWrap(True)
        ayuda.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        layout.addWidget(ayuda)

        # --- barra de herramientas ---
        barra = QHBoxLayout()
        self.btn_agregar = QPushButton("+ Agregar relación manual")
        self.btn_agregar.clicked.connect(self._abrir_dialogo_agregar_relacion)
        barra.addWidget(self.btn_agregar)

        self.btn_redetectar = QPushButton("Detectar automáticamente de nuevo")
        self.btn_redetectar.setToolTip(
            "Vuelve a analizar las tablas. No borra las relaciones que agregaste a mano."
        )
        self.btn_redetectar.clicked.connect(self._redetectar)
        barra.addWidget(self.btn_redetectar)
        barra.addStretch()

        self.lbl_resumen = QLabel("")
        self.lbl_resumen.setStyleSheet(f"color: {colors['muted']}; font-size: 11px;")
        barra.addWidget(self.lbl_resumen)
        layout.addLayout(barra)

        # --- lienzo ---
        self.escena = QGraphicsScene(self)
        self.escena.setBackgroundBrush(QBrush(QColor(colors["bg"])))
        self.vista = QGraphicsView(self.escena)
        self.vista.setRenderHint(QPainter.Antialiasing)
        self.vista.setDragMode(QGraphicsView.RubberBandDrag)
        self.vista.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        layout.addWidget(self.vista, stretch=1)

        # --- leyenda ---
        leyenda = QLabel(
            f"<span style='color:{COLOR_ACCENT}'>●</span> confianza alta&nbsp;&nbsp;"
            f"<span style='color:{COLOR_ACCENT_2}'>●</span> confianza media&nbsp;&nbsp;"
            f"<span style='color:{colors['muted']}'>●</span> confianza baja&nbsp;&nbsp;"
            f"<span style='color:{COLOR_ACCENT_3}'>- - -</span> agregada por ti&nbsp;&nbsp;"
            f"<span style='color:{COLOR_DANGER}'>···</span> dirección sin confirmar (clic derecho)"
        )
        leyenda.setStyleSheet("font-size: 11px;")
        layout.addWidget(leyenda)

        # --- botones de cierre ---
        botones = QDialogButtonBox(QDialogButtonBox.Close)
        botones.rejected.connect(self.reject)
        botones.accepted.connect(self.accept)
        botones.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        layout.addWidget(botones)

        self._dibujar_todo()

    # ------------------------------------------------------------------

    def _dibujar_todo(self):
        self.escena.clear()
        self._cajas.clear()
        self._lineas.clear()
        self._colocar_cajas()
        self._dibujar_relaciones()
        self._actualizar_resumen()
        self.vista.setSceneRect(self.escena.itemsBoundingRect().adjusted(-40, -40, 40, 40))

    def _colocar_cajas(self):
        x, y = 0, 0
        max_alto_fila = 0
        for i, (nombre_tabla, df) in enumerate(self.tablas.items()):
            caja = _CajaTabla(nombre_tabla, list(df.columns), self.colors)
            caja.setPos(x, y)
            self.escena.addItem(caja)
            self._cajas[nombre_tabla] = caja

            max_alto_fila = max(max_alto_fila, caja.rect().height())
            x += ANCHO_CAJA + ESPACIO_ENTRE_CAJAS_X
            if (i + 1) % ESPACIO_ENTRE_CAJAS_X_MAX_POR_FILA == 0:
                x = 0
                y += max_alto_fila + ESPACIO_ENTRE_CAJAS_Y
                max_alto_fila = 0

    def _dibujar_relaciones(self):
        for r in self.relaciones:
            caja_o = self._cajas.get(r.tabla_origen)
            caja_d = self._cajas.get(r.tabla_destino)
            if not caja_o or not caja_d:
                continue
            linea = _LineaRelacion(r, caja_o, caja_d, self.colors, self._eliminar_relacion, self._confirmar_direccion)
            self.escena.addItem(linea)
            self._lineas.append(linea)
            caja_o.resaltar_columna(r.columna_origen, linea._color_por_confianza())
            caja_d.resaltar_columna(r.columna_destino, linea._color_por_confianza())

    def _actualizar_resumen(self):
        n = len(self.relaciones)
        manuales = sum(1 for r in self.relaciones if getattr(r, "manual", False))
        if n == 0:
            self.lbl_resumen.setText("No se detectaron relaciones todavía.")
        else:
            extra = f" ({manuales} agregada(s) por ti)" if manuales else ""
            self.lbl_resumen.setText(f"{n} relación(es) sugerida(s){extra}")

    # ------------------------------------------------------------------
    # Acciones del usuario
    # ------------------------------------------------------------------

    def _eliminar_relacion(self, linea: _LineaRelacion):
        respuesta = QMessageBox.question(
            self, "Eliminar relación",
            f"¿Eliminar la relación entre \"{linea.relacion.tabla_origen}."
            f"{linea.relacion.columna_origen}\" y \"{linea.relacion.tabla_destino}."
            f"{linea.relacion.columna_destino}\"?",
        )
        if respuesta != QMessageBox.Yes:
            return
        if linea.relacion in self.relaciones:
            self.relaciones.remove(linea.relacion)
        self._dibujar_todo()

    def _confirmar_direccion(self, linea: _LineaRelacion, tabla_elegida: str, columna_elegida: str):
        """El usuario eligió a mano, desde el clic derecho, cuál tabla es
        la principal de esta relación (para corregir un 'sin_definir', o
        simplemente porque el motor adivinó mal)."""
        r = linea.relacion
        r.tabla_principal = tabla_elegida
        r.columna_principal = columna_elegida
        r.certeza_direccion = "alta"
        r.direccion_confirmada = True
        self._dibujar_todo()

    def _redetectar(self):
        """Vuelve a correr el motor automático, conservando las relaciones
        que el usuario agregó a mano (y sin resucitar las que borró, salvo
        que efectivamente sigan siendo válidas y el usuario no las haya
        tocado — se identifican por tabla+columna en ambos extremos), y
        respetando cualquier dirección padre/hijo que el usuario haya
        confirmado a mano en una relación detectada automáticamente."""
        manuales = [r for r in self.relaciones if getattr(r, "manual", False)]
        direcciones_confirmadas = {
            (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino): r
            for r in self.relaciones
            if getattr(r, "direccion_confirmada", False) and not getattr(r, "manual", False)
        }
        nuevas_auto = inferir_relaciones(self.tablas)

        claves_manuales = {
            (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) for r in manuales
        }
        nuevas_auto = [
            r for r in nuevas_auto
            if (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) not in claves_manuales
        ]

        for r in nuevas_auto:
            clave = (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino)
            vieja = direcciones_confirmadas.get(clave)
            if vieja is not None:
                r.tabla_principal = vieja.tabla_principal
                r.columna_principal = vieja.columna_principal
                r.certeza_direccion = "alta"
                r.direccion_confirmada = True

        self.relaciones = nuevas_auto + manuales
        self._dibujar_todo()

    def _abrir_dialogo_agregar_relacion(self):
        dialogo = _DialogoAgregarRelacion(self.tablas, self)
        if dialogo.exec() == QDialog.DialogCode.Accepted:
            t1, c1, t2, c2 = dialogo.seleccion()
            if t1 == t2:
                QMessageBox.warning(self, "Relación inválida", "Elige dos tablas distintas.")
                return
            ya_existe = any(
                (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) == (t1, c1, t2, c2)
                or (r.tabla_origen, r.columna_origen, r.tabla_destino, r.columna_destino) == (t2, c2, t1, c1)
                for r in self.relaciones
            )
            if ya_existe:
                QMessageBox.information(self, "Ya existe", "Esa relación ya está en el esquema.")
                return
            nueva = RelacionSugerida(
                tabla_origen=t1, columna_origen=c1,
                tabla_destino=t2, columna_destino=c2,
                confianza=1.0, razon="Agregada manualmente por el usuario",
            )
            nueva.manual = True
            self.relaciones.append(nueva)
            self._dibujar_todo()

    # ------------------------------------------------------------------

    def obtener_relaciones_confirmadas(self) -> list[RelacionSugerida]:
        """Para usar más adelante (Paso C): el estado final del esquema,
        tal como quedó después de que el usuario lo revisó."""
        return list(self.relaciones)

    def render_a_pixmap(self) -> QPixmap:
        """Imagen (QPixmap) del esquema tal como está dibujado ahora mismo
        -- para llevarlo a la pestaña Reporte sin necesidad de abrir esta
        ventana ni pedirle nada al usuario."""
        rect = self.escena.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        ancho = max(1, int(rect.width()))
        alto = max(1, int(rect.height()))
        pixmap = QPixmap(ancho, alto)
        pixmap.fill(QColor(self.colors["bg"]))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        self.escena.render(painter, target=pixmap.rect(), source=rect)
        painter.end()
        return pixmap


class _DialogoAgregarRelacion(QDialog):
    """Formulario chico: elegir tabla+columna de un lado y del otro."""

    def __init__(self, tablas: dict, parent=None):
        super().__init__(parent)
        self.tablas = tablas
        self.setWindowTitle("Agregar relación manual")

        layout = QVBoxLayout(self)

        fila1 = QHBoxLayout()
        self.combo_tabla1 = QComboBox()
        self.combo_tabla1.addItems(list(tablas.keys()))
        self.combo_col1 = QComboBox()
        fila1.addWidget(QLabel("Tabla:"))
        fila1.addWidget(self.combo_tabla1)
        fila1.addWidget(QLabel("Columna:"))
        fila1.addWidget(self.combo_col1)
        layout.addLayout(fila1)

        layout.addWidget(QLabel("se relaciona con"))

        fila2 = QHBoxLayout()
        self.combo_tabla2 = QComboBox()
        self.combo_tabla2.addItems(list(tablas.keys()))
        if len(tablas) > 1:
            self.combo_tabla2.setCurrentIndex(1)
        self.combo_col2 = QComboBox()
        fila2.addWidget(QLabel("Tabla:"))
        fila2.addWidget(self.combo_tabla2)
        fila2.addWidget(QLabel("Columna:"))
        fila2.addWidget(self.combo_col2)
        layout.addLayout(fila2)

        self.combo_tabla1.currentTextChanged.connect(self._refrescar_col1)
        self.combo_tabla2.currentTextChanged.connect(self._refrescar_col2)
        self._refrescar_col1(self.combo_tabla1.currentText())
        self._refrescar_col2(self.combo_tabla2.currentText())

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _refrescar_col1(self, nombre_tabla):
        self.combo_col1.clear()
        if nombre_tabla in self.tablas:
            self.combo_col1.addItems([str(c) for c in self.tablas[nombre_tabla].columns])

    def _refrescar_col2(self, nombre_tabla):
        self.combo_col2.clear()
        if nombre_tabla in self.tablas:
            self.combo_col2.addItems([str(c) for c in self.tablas[nombre_tabla].columns])

    def seleccion(self):
        return (
            self.combo_tabla1.currentText(), self.combo_col1.currentText(),
            self.combo_tabla2.currentText(), self.combo_col2.currentText(),
        )