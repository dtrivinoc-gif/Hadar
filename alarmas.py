"""
Función "Alarma": el usuario define un umbral sobre un Indicador, una
Métrica (media/mediana/moda/desviación estándar de una columna numérica)
o una Frecuencia (conteo de la categoría más repetida de una columna), y
Hadar avisa visualmente cuando el valor actual lo supera.

Este módulo tiene tres partes:
- El motor puro (AlarmaHadar, calcular_valor_actual, evaluar_regla): no
  sabe nada de Qt, solo guarda reglas en SQLite y las evalúa contra un
  DataFrame. Mismo patrón de persistencia que memoria.py.
- AlarmaCard: la tarjeta de una regla en la pestaña "Alarma".
- DialogoAlarma: crear/editar una regla.

La MARCA visual (marco rojo) sobre las tarjetas de Indicadores/Métricas/
Frecuencias la aplica main_window.py directamente sobre esos widgets --
este módulo solo dice "esta regla está en alarma sí/no".
"""
import operator
import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QComboBox, QPushButton, QFrame, QMessageBox, QCheckBox,
)

from .memoria import ruta_directorio_datos_usuario

OPERADORES = {
    "Mayor que (>)": operator.gt,
    "Mayor o igual (≥)": operator.ge,
    "Menor que (<)": operator.lt,
    "Menor o igual (≤)": operator.le,
}

ESTADISTICOS_METRICA = {
    "Media": lambda serie: serie.mean(),
    "Mediana": lambda serie: serie.median(),
    "Moda": lambda serie: serie.mode().iloc[0] if not serie.mode().empty else None,
    "Desviación Std": lambda serie: serie.std(),
}

TIPOS_ALARMA = ["Indicador", "Métrica", "Frecuencia"]


def ruta_alarmas_db():
    return os.path.join(ruta_directorio_datos_usuario(), "hadar_alarmas.db")


@contextmanager
def _conexion(ruta_db):
    con = sqlite3.connect(ruta_db)
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ----------------------------------------------------------------------------
# Motor: guardar y evaluar reglas
# ----------------------------------------------------------------------------

class AlarmaHadar:
    """Guarda y evalúa las reglas de alarma configuradas por el usuario.
    Persisten en SQLite (misma carpeta que la memoria de anomalías) para
    que sobrevivan a cerrar y volver a abrir Hadar."""

    def __init__(self, ruta_db=None):
        self.ruta_db = ruta_db or ruta_alarmas_db()
        self._crear_tabla()

    def _crear_tabla(self):
        with _conexion(self.ruta_db) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS reglas_alarma (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT NOT NULL,
                    tipo TEXT NOT NULL,
                    objetivo TEXT NOT NULL,
                    estadistico TEXT,
                    operador TEXT NOT NULL,
                    umbral REAL NOT NULL,
                    mensaje TEXT,
                    activa INTEGER NOT NULL DEFAULT 1,
                    notificar_correo INTEGER NOT NULL DEFAULT 0,
                    correo_destino TEXT
                )
            """)
            # Migración suave: si la tabla ya existía de antes de agregar
            # el aviso por correo, le sumamos las columnas que falten sin
            # perder las reglas ya guardadas.
            columnas_existentes = {
                fila[1] for fila in con.execute("PRAGMA table_info(reglas_alarma)")
            }
            if "notificar_correo" not in columnas_existentes:
                con.execute(
                    "ALTER TABLE reglas_alarma ADD COLUMN notificar_correo INTEGER NOT NULL DEFAULT 0"
                )
            if "correo_destino" not in columnas_existentes:
                con.execute("ALTER TABLE reglas_alarma ADD COLUMN correo_destino TEXT")

    def listar_reglas(self):
        columnas = ["id", "nombre", "tipo", "objetivo", "estadistico",
                    "operador", "umbral", "mensaje", "activa",
                    "notificar_correo", "correo_destino"]
        with _conexion(self.ruta_db) as con:
            filas = con.execute(
                f"SELECT {', '.join(columnas)} FROM reglas_alarma ORDER BY id"
            ).fetchall()
        return [dict(zip(columnas, f)) for f in filas]

    def agregar_regla(self, nombre, tipo, objetivo, operador, umbral,
                       estadistico=None, mensaje="", notificar_correo=0, correo_destino=None):
        with _conexion(self.ruta_db) as con:
            cur = con.execute(
                "INSERT INTO reglas_alarma "
                "(nombre, tipo, objetivo, estadistico, operador, umbral, mensaje, activa, "
                "notificar_correo, correo_destino) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (nombre, tipo, objetivo, estadistico, operador, umbral, mensaje,
                 notificar_correo, correo_destino),
            )
            return cur.lastrowid

    def actualizar_regla(self, regla_id, **campos):
        if not campos:
            return
        sets = ", ".join(f"{k} = ?" for k in campos)
        valores = list(campos.values()) + [regla_id]
        with _conexion(self.ruta_db) as con:
            con.execute(f"UPDATE reglas_alarma SET {sets} WHERE id = ?", valores)

    def eliminar_regla(self, regla_id):
        with _conexion(self.ruta_db) as con:
            con.execute("DELETE FROM reglas_alarma WHERE id = ?", (regla_id,))


def calcular_valor_actual(regla, df, indicadores):
    """Valor actual de lo que la regla vigila, o None si no se puede
    calcular ahora mismo (ej. la columna ya no existe en los datos
    cargados, o el indicador fue borrado)."""
    if df is None or df.empty:
        return None

    if regla["tipo"] == "Indicador":
        indicador = next((i for i in indicadores if i.nombre == regla["objetivo"]), None)
        if indicador is None:
            return None
        try:
            return indicador.calcular(df)
        except Exception:
            return None

    if regla["tipo"] == "Métrica":
        columna = regla["objetivo"]
        if columna not in df.columns:
            return None
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()
        if serie.empty:
            return None
        funcion = ESTADISTICOS_METRICA.get(regla["estadistico"])
        if funcion is None:
            return None
        try:
            valor = funcion(serie)
            return float(valor) if valor is not None else None
        except (TypeError, ValueError):
            return None

    if regla["tipo"] == "Frecuencia":
        columna = regla["objetivo"]
        if columna not in df.columns:
            return None
        serie = df[columna].dropna().astype(str)
        if serie.empty:
            return None
        return int(serie.value_counts().max())

    return None


def evaluar_regla(regla, valor_actual):
    """True si el valor actual dispara la alarma."""
    if valor_actual is None or not regla.get("activa", True):
        return False
    funcion_operador = OPERADORES.get(regla["operador"])
    if funcion_operador is None:
        return False
    try:
        return bool(funcion_operador(valor_actual, regla["umbral"]))
    except TypeError:
        return False


# ----------------------------------------------------------------------------
# UI: tarjeta de una regla + diálogo para crearla/editarla
# ----------------------------------------------------------------------------

class AlarmaCard(QFrame):
    """Tarjeta de una regla de alarma: nombre, condición en palabras,
    estado actual (🟢/🔴), valor actual, y botones editar/eliminar."""

    def __init__(self, regla, valor_actual, en_alarma, on_editar, on_eliminar, parent=None):
        super().__init__(parent)
        self.regla = regla
        self.setObjectName("statCard")
        self.setProperty("alarma", "true" if en_alarma else "false")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)

        fila_titulo = QHBoxLayout()
        icono = QLabel("🔴" if en_alarma else "🟢")
        fila_titulo.addWidget(icono)
        lbl_nombre = QLabel(f"<b>{regla['nombre']}</b>")
        lbl_nombre.setWordWrap(True)
        fila_titulo.addWidget(lbl_nombre, stretch=1)
        layout.addLayout(fila_titulo)

        condicion = _describir_condicion(regla)
        lbl_condicion = QLabel(condicion)
        lbl_condicion.setObjectName("muted")
        lbl_condicion.setWordWrap(True)
        layout.addWidget(lbl_condicion)

        texto_valor = "Sin datos" if valor_actual is None else f"Valor actual: {valor_actual:,.2f}".replace(",", ".")
        lbl_valor = QLabel(texto_valor)
        layout.addWidget(lbl_valor)

        texto_correo = (
            f"📧 Avisa por correo a {regla['correo_destino']}" if regla.get("notificar_correo") and regla.get("correo_destino")
            else "📧 Avisa por correo (al predeterminado)" if regla.get("notificar_correo")
            else "📧 Sin aviso por correo"
        )
        lbl_correo = QLabel(texto_correo)
        lbl_correo.setObjectName("muted")
        layout.addWidget(lbl_correo)

        if en_alarma and regla.get("mensaje"):
            lbl_mensaje = QLabel(f"⚠ {regla['mensaje']}")
            lbl_mensaje.setWordWrap(True)
            lbl_mensaje.setStyleSheet("color: #EF4444; font-weight: bold;")
            layout.addWidget(lbl_mensaje)

        fila_botones = QHBoxLayout()
        fila_botones.addStretch()
        btn_editar = QPushButton("Editar")
        btn_editar.clicked.connect(lambda: on_editar(regla))
        fila_botones.addWidget(btn_editar)
        btn_eliminar = QPushButton("X")
        btn_eliminar.setFixedWidth(28)
        btn_eliminar.clicked.connect(lambda: on_eliminar(regla))
        fila_botones.addWidget(btn_eliminar)
        layout.addLayout(fila_botones)


def _describir_condicion(regla):
    if regla["tipo"] == "Indicador":
        sujeto = f"Indicador «{regla['objetivo']}»"
    elif regla["tipo"] == "Métrica":
        sujeto = f"{regla['estadistico']} de «{regla['objetivo']}»"
    else:
        sujeto = f"Categoría más frecuente de «{regla['objetivo']}»"
    return f"{sujeto} {regla['operador']} {regla['umbral']:,.2f}".replace(",", ".")


class DialogoAlarma(QDialog):
    """Crear o editar una regla de alarma. Si se entrega `regla`, se abre
    en modo edición pre-llenado con sus valores actuales."""

    def __init__(self, nombres_indicadores, columnas_numericas, columnas_todas,
                 regla=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Alarma" if regla else "Nueva Alarma")
        self.setMinimumWidth(380)
        self._columnas_numericas = columnas_numericas
        self._columnas_todas = columnas_todas
        self._nombres_indicadores = nombres_indicadores

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Nombre (opcional):"))
        self.nombre_edit = QLineEdit(regla["nombre"] if regla else "")
        self.nombre_edit.setPlaceholderText("Se genera solo si lo dejas vacío")
        layout.addWidget(self.nombre_edit)

        layout.addWidget(QLabel("Tipo:"))
        self.tipo_combo = QComboBox()
        self.tipo_combo.addItems(TIPOS_ALARMA)
        self.tipo_combo.currentTextChanged.connect(self._refrescar_objetivo)
        layout.addWidget(self.tipo_combo)

        layout.addWidget(QLabel("Objetivo:"))
        self.objetivo_combo = QComboBox()
        layout.addWidget(self.objetivo_combo)

        self.lbl_estadistico = QLabel("Estadístico:")
        self.estadistico_combo = QComboBox()
        self.estadistico_combo.addItems(list(ESTADISTICOS_METRICA.keys()))
        layout.addWidget(self.lbl_estadistico)
        layout.addWidget(self.estadistico_combo)

        fila_condicion = QHBoxLayout()
        self.operador_combo = QComboBox()
        self.operador_combo.addItems(list(OPERADORES.keys()))
        fila_condicion.addWidget(self.operador_combo)
        self.umbral_edit = QLineEdit(str(regla["umbral"]) if regla else "")
        self.umbral_edit.setPlaceholderText("Número umbral")
        fila_condicion.addWidget(self.umbral_edit)
        layout.addLayout(fila_condicion)

        layout.addWidget(QLabel("Mensaje de aviso (opcional):"))
        self.mensaje_edit = QLineEdit(regla.get("mensaje", "") if regla else "")
        self.mensaje_edit.setPlaceholderText("Ej: revisar con el proveedor")
        layout.addWidget(self.mensaje_edit)

        self.checkbox_correo = QCheckBox("Avisar también por correo cuando se active")
        self.checkbox_correo.setChecked(bool(regla.get("notificar_correo")) if regla else False)
        layout.addWidget(self.checkbox_correo)

        layout.addWidget(QLabel("Correo destino (opcional -- si lo dejas vacío, usa el predeterminado):"))
        self.correo_destino_edit = QLineEdit(regla.get("correo_destino") or "" if regla else "")
        self.correo_destino_edit.setPlaceholderText("tu-correo@ejemplo.com")
        layout.addWidget(self.correo_destino_edit)

        botones = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        botones.accepted.connect(self._validar_y_aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

        if regla:
            self.tipo_combo.setCurrentText(regla["tipo"])
            self._refrescar_objetivo(regla["tipo"])
            self.objetivo_combo.setCurrentText(regla["objetivo"])
            if regla.get("estadistico"):
                self.estadistico_combo.setCurrentText(regla["estadistico"])
            self.operador_combo.setCurrentText(regla["operador"])
        else:
            self._refrescar_objetivo(self.tipo_combo.currentText())

    def _refrescar_objetivo(self, tipo):
        self.objetivo_combo.clear()
        if tipo == "Indicador":
            self.objetivo_combo.addItems(self._nombres_indicadores)
            self.lbl_estadistico.setVisible(False)
            self.estadistico_combo.setVisible(False)
        elif tipo == "Métrica":
            self.objetivo_combo.addItems(self._columnas_numericas)
            self.lbl_estadistico.setVisible(True)
            self.estadistico_combo.setVisible(True)
        else:  # Frecuencia
            self.objetivo_combo.addItems(self._columnas_todas)
            self.lbl_estadistico.setVisible(False)
            self.estadistico_combo.setVisible(False)

    def _validar_y_aceptar(self):
        if not self.objetivo_combo.currentText():
            _avisar_sin_objetivo(self)
            return
        try:
            float(self.umbral_edit.text().replace(",", "."))
        except ValueError:
            self.umbral_edit.setStyleSheet("border: 1px solid #EF4444;")
            return
        self.accept()

    def get_datos(self):
        tipo = self.tipo_combo.currentText()
        objetivo = self.objetivo_combo.currentText()
        umbral = float(self.umbral_edit.text().replace(",", "."))
        nombre = self.nombre_edit.text().strip() or f"{tipo}: {objetivo}"
        return {
            "nombre": nombre,
            "tipo": tipo,
            "objetivo": objetivo,
            "estadistico": self.estadistico_combo.currentText() if tipo == "Métrica" else None,
            "operador": self.operador_combo.currentText(),
            "umbral": umbral,
            "mensaje": self.mensaje_edit.text().strip(),
            "notificar_correo": 1 if self.checkbox_correo.isChecked() else 0,
            "correo_destino": self.correo_destino_edit.text().strip() or None,
        }


def _avisar_sin_objetivo(parent):
    """Aviso corto cuando no hay ningún objetivo disponible para elegir
    (ej. no hay indicadores creados todavía)."""
    QMessageBox.information(
        parent, "Elige un objetivo",
        "No hay nada para elegir en 'Objetivo' -- si es un Indicador, créalo "
        "primero en la pestaña Indicadores.",
    )