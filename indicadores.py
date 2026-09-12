"""
Indicadores (KPIs) configurables: la clase Indicador (cálculo + formato),
sus tarjetas de UI (IndicadorCard) y el diálogo para crear/editar uno
(DialogoIndicador).
"""
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTextEdit, QDialog, QDialogButtonBox,
)

from .formulas import evaluar_formula_segura, extraer_columnas_formula, FormulaSeguridadError

INDICADOR_OPERACIONES = {
    "Suma": "suma",
    "Promedio": "promedio",
    "Mediana": "mediana",
    "Mínimo": "minimo",
    "Máximo": "maximo",
    "Conteo (no nulos)": "conteo",
    "Conteo de valores únicos": "conteo_unico",
    "Fórmula personalizada": "formula",
}
INDICADOR_OPERACIONES_INV = {v: k for k, v in INDICADOR_OPERACIONES.items()}


class Indicador:
    """Un indicador que el propio usuario define: una operación básica sobre
    una columna, o una fórmula simple combinando columnas con pandas."""

    def __init__(self, nombre, operacion, columna=None, formula=None, formato="{value:,.0f}"):
        self.nombre = nombre
        self.operacion = operacion
        self.columna = columna
        self.formula = formula
        self.formato = formato or "{value:,.0f}"

    def calcular(self, df):
        if df is None or df.empty:
            return None
        try:
            if self.operacion == "formula":
                if not self.formula:
                    return None
                # Evaluador seguro basado en AST (lista blanca), no eval().
                # Ver _EvaluadorFormulaSegura para el detalle de qué se permite.
                return evaluar_formula_segura(self.formula, df)
            if not self.columna or self.columna not in df.columns:
                return None
            serie = df[self.columna]
            if self.operacion == "suma":
                return pd.to_numeric(serie, errors="coerce").sum()
            if self.operacion == "promedio":
                return pd.to_numeric(serie, errors="coerce").mean()
            if self.operacion == "mediana":
                return pd.to_numeric(serie, errors="coerce").median()
            if self.operacion == "minimo":
                return pd.to_numeric(serie, errors="coerce").min()
            if self.operacion == "maximo":
                return pd.to_numeric(serie, errors="coerce").max()
            if self.operacion == "conteo":
                return serie.count()
            if self.operacion == "conteo_unico":
                return serie.nunique()
        except Exception:
            return None
        return None

    def columnas_de_las_que_depende(self):
        """De qué columnas del DataFrame depende este indicador -- la base
        del linaje: sirve para saber, sin recalcular nada, si una columna
        con anomalías activas afecta a este indicador. Para una operación
        simple (Suma, Promedio, etc.) es solo `self.columna`; para una
        fórmula personalizada, se obtiene recorriendo su AST (ver
        formulas.extraer_columnas_formula). Nunca lanza una excepción hacia
        afuera -- una fórmula con error de sintaxis simplemente no aporta
        columnas conocidas todavía, en vez de romper la tarjeta del
        indicador."""
        if self.operacion == "formula":
            if not self.formula:
                return set()
            try:
                return extraer_columnas_formula(self.formula)
            except FormulaSeguridadError:
                return set()
        return {self.columna} if self.columna else set()

    def formatear(self, valor):
        if valor is None:
            return "-"
        try:
            return self.formato.format(value=valor)
        except Exception:
            try:
                return f"{valor:,.2f}"
            except Exception:
                return str(valor)

    def to_dict(self):
        return {
            "nombre": self.nombre,
            "operacion": self.operacion,
            "columna": self.columna,
            "formula": self.formula,
            "formato": self.formato,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            nombre=data.get("nombre", "Indicador"),
            operacion=data.get("operacion", "suma"),
            columna=data.get("columna"),
            formula=data.get("formula"),
            formato=data.get("formato", "{value:,.0f}"),
        )


def sugerir_indicadores_por_defecto(df):
    """Propone algunos indicadores razonables a partir de las columnas del
    DataFrame, solo como punto de partida: el usuario los puede editar o
    borrar después."""
    indicadores = []
    if df is None or df.empty:
        return indicadores

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        col = num_cols[0]
        indicadores.append(Indicador(f"Total {col}", "suma", col, formato="{value:,.0f}"))
        indicadores.append(Indicador(f"Promedio {col}", "promedio", col, formato="{value:,.2f}"))
        if len(num_cols) > 1:
            col2 = num_cols[1]
            indicadores.append(Indicador(
                f"Ratio {col}/{col2}", "formula",
                formula=f"df['{col}'].sum() / df['{col2}'].sum() * 100",
                formato="{value:.1f}%",
            ))

    texto_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in texto_cols[:2]:
        if df[col].nunique() < 50:
            indicadores.append(Indicador(f"Categorías únicas en {col}", "conteo_unico", col))

    return indicadores


class IndicadorCard(QFrame):
    """Tarjeta de un indicador: nombre, valor calculado, y botones para
    editarlo o eliminarlo. Si el indicador tiene una regla de Alarma
    asociada y está activada, se marca con marco rojo y un mensaje."""

    def __init__(self, indicador, valor, on_editar, on_eliminar, parent=None,
                 en_alarma=False, mensaje_alarma=None, columnas_con_anomalias=None):
        super().__init__(parent)
        self.indicador = indicador
        self.setObjectName("statCard")
        self.setProperty("alarma", "true" if en_alarma else "false")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)

        lbl_titulo = QLabel(indicador.nombre)
        lbl_titulo.setObjectName("statTitle")
        lbl_titulo.setWordWrap(True)
        layout.addWidget(lbl_titulo)

        lbl_valor = QLabel(indicador.formatear(valor))
        lbl_valor.setObjectName("statValue")
        layout.addWidget(lbl_valor)

        columnas_dependencia = sorted(indicador.columnas_de_las_que_depende())
        if columnas_dependencia:
            lbl_depende = QLabel("Depende de: " + ", ".join(columnas_dependencia))
            lbl_depende.setWordWrap(True)
            lbl_depende.setStyleSheet("color: #9CA3AF; font-size: 11px;")
            layout.addWidget(lbl_depende)

        # No es una Alarma (esa la configura el usuario a mano sobre el
        # indicador) -- esto es más simple: si Narrativa ya detectó una
        # anomalía en alguna columna de la que depende este indicador, se
        # avisa acá también, sin tener que ir a revisar la otra pestaña.
        columnas_afectadas = sorted(set(columnas_dependencia) & (columnas_con_anomalias or set()))
        if columnas_afectadas:
            lbl_dependencia_riesgo = QLabel(
                "⚠ " + ", ".join(f"'{c}'" for c in columnas_afectadas)
                + (" tiene" if len(columnas_afectadas) == 1 else " tienen")
                + " una anomalía activa (ver Narrativa)"
            )
            lbl_dependencia_riesgo.setWordWrap(True)
            lbl_dependencia_riesgo.setStyleSheet("color: #b45309; font-size: 11px; font-weight: bold;")
            layout.addWidget(lbl_dependencia_riesgo)

        if en_alarma and mensaje_alarma:
            lbl_alarma = QLabel(f"⚠ {mensaje_alarma}")
            lbl_alarma.setWordWrap(True)
            lbl_alarma.setStyleSheet("color: #EF4444; font-weight: bold;")
            layout.addWidget(lbl_alarma)

        fila_botones = QHBoxLayout()
        fila_botones.addStretch()
        btn_editar = QPushButton("Editar")
        btn_editar.clicked.connect(lambda: on_editar(indicador))
        fila_botones.addWidget(btn_editar)
        btn_eliminar = QPushButton("X")
        btn_eliminar.setFixedWidth(28)
        btn_eliminar.clicked.connect(lambda: on_eliminar(indicador))
        fila_botones.addWidget(btn_eliminar)
        layout.addLayout(fila_botones)


class DialogoIndicador(QDialog):
    """Crear o editar un indicador. Si se entrega `indicador`, se abre en
    modo edición pre-llenado con sus valores actuales."""

    def __init__(self, columnas, indicador=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Indicador" if indicador else "Nuevo Indicador")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Nombre:"))
        self.nombre_edit = QLineEdit(indicador.nombre if indicador else "")
        layout.addWidget(self.nombre_edit)

        layout.addWidget(QLabel("Operación:"))
        self.operacion_combo = QComboBox()
        self.operacion_combo.addItems(list(INDICADOR_OPERACIONES.keys()))
        if indicador:
            self.operacion_combo.setCurrentText(
                INDICADOR_OPERACIONES_INV.get(indicador.operacion, "Suma")
            )
        layout.addWidget(self.operacion_combo)

        self.lbl_columna = QLabel("Columna:")
        layout.addWidget(self.lbl_columna)
        self.columna_combo = QComboBox()
        self.columna_combo.addItems([str(c) for c in columnas])
        if indicador and indicador.columna in columnas:
            self.columna_combo.setCurrentText(str(indicador.columna))
        layout.addWidget(self.columna_combo)

        self.lbl_formula = QLabel(
            "Fórmula (usa df para tus datos, ej: df['Ventas'].sum() / df['Costos'].sum() * 100):"
        )
        layout.addWidget(self.lbl_formula)
        self.formula_edit = QTextEdit(indicador.formula if (indicador and indicador.formula) else "")
        self.formula_edit.setMaximumHeight(70)
        layout.addWidget(self.formula_edit)

        layout.addWidget(QLabel("Formato de despliegue (opcional):"))
        self.formato_edit = QLineEdit(indicador.formato if indicador else "{value:,.0f}")
        self.formato_edit.setToolTip("Ejemplos: {value:,.0f}  ·  ${value:,.2f}  ·  {value:.1f}%")
        layout.addWidget(self.formato_edit)

        self.operacion_combo.currentTextChanged.connect(self._actualizar_campos_visibles)
        self._actualizar_campos_visibles(self.operacion_combo.currentText())

        botones = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _actualizar_campos_visibles(self, operacion_label):
        es_formula = operacion_label == "Fórmula personalizada"
        self.lbl_columna.setVisible(not es_formula)
        self.columna_combo.setVisible(not es_formula)
        self.lbl_formula.setVisible(es_formula)
        self.formula_edit.setVisible(es_formula)

    def get_indicador(self):
        operacion_label = self.operacion_combo.currentText()
        operacion = INDICADOR_OPERACIONES.get(operacion_label, "suma")
        return Indicador(
            nombre=self.nombre_edit.text().strip() or "Indicador",
            operacion=operacion,
            columna=self.columna_combo.currentText() if operacion != "formula" else None,
            formula=self.formula_edit.toPlainText().strip() if operacion == "formula" else None,
            formato=self.formato_edit.text().strip() or "{value:,.0f}",
        )


# ----------------------------------------------------------------------------
# Transformación estilo Excel: crear una columna nueva combinando columnas
# existentes con una fórmula simple, aplicada a todas las filas de una sola
# vez -- el equivalente a escribir una fórmula en la primera celda de Excel y
# arrastrarla hacia abajo. Con esto, cualquier cálculo (el que antes solo se
# veía como una tarjeta en Indicadores o un recuadro de +Cálculo) puede
# convertirse en una columna real, disponible para Gráficos, Métricas,
# Frecuencias, Narrativa y Reporte -- igual que cualquier columna cargada desde
# archivo. Filosofía Hadar: que se arme haciendo clic, sin sintaxis que
# aprender.
# ----------------------------------------------------------------------------