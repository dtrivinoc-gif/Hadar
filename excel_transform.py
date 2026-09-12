"""
Diálogo del módulo "Excel: Calcular y Arrastrar": permite al usuario
escribir una fórmula (traducida desde sintaxis tipo Excel) y crear una
columna nueva a partir de ella.
"""
import re

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QMessageBox, QDialog, QDialogButtonBox,
)

from .formulas import evaluar_formula_segura, FormulaSeguridadError

TRANSFORM_FUNCIONES_AYUDA = (
    "Funciones disponibles: SUMA(a, b, ...) · PROMEDIO(a, b, ...) · "
    "MINIMO(a, b, ...) · MAXIMO(a, b, ...) · REDONDEAR(valor, decimales) · "
    "ABS(valor) · SI(condición, si_es_verdadero, si_es_falso). También puedes "
    "encadenar métodos como [Columna].sum(), .mean(), .round(2), .fillna(0)."
)


class DialogoTransformacionExcel(QDialog):
    """Crea una columna nueva a partir de una fórmula que combina columnas
    existentes de self.df, aplicada de una sola vez a todas las filas.

    Uso: DialogoTransformacionExcel(df, parent).exec(); si el usuario acepta,
    get_resultado() devuelve (df_con_la_columna_nueva, nombre_de_la_columna).
    """

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.df_resultado = None
        self.nombre_columna = None

        self.setWindowTitle("Excel: Calcular y Arrastrar")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)

        info = QLabel(
            "Crea una columna nueva combinando las columnas que ya tienes, igual que "
            "una fórmula de Excel arrastrada por toda la columna. Haz clic en una "
            "columna o en un operador para agregarlo a la fórmula."
        )
        info.setWordWrap(True)
        info.setObjectName("muted")
        layout.addWidget(info)

        layout.addWidget(QLabel("Nombre de la columna nueva:"))
        self.nombre_edit = QLineEdit()
        self.nombre_edit.setPlaceholderText("Ej: Margen, Total con IVA, Días de atraso...")
        layout.addWidget(self.nombre_edit)

        layout.addWidget(QLabel("Columnas (clic para insertar):"))
        columnas_scroll = QScrollArea()
        columnas_scroll.setWidgetResizable(True)
        columnas_scroll.setFixedHeight(120)
        columnas_contenedor = QWidget()
        columnas_grid = QGridLayout(columnas_contenedor)
        columnas_grid.setSpacing(6)
        columnas = [str(c) for c in df.columns]
        for i, col in enumerate(columnas):
            btn = QPushButton(col)
            btn.setToolTip(f"Insertar [{col}] en la fórmula")
            btn.clicked.connect(lambda _checked=False, c=col: self._insertar_texto(f"[{c}]"))
            columnas_grid.addWidget(btn, i // 3, i % 3)
        columnas_scroll.setWidget(columnas_contenedor)
        layout.addWidget(columnas_scroll)

        layout.addWidget(QLabel("Operadores:"))
        operadores_layout = QHBoxLayout()
        for simbolo in ["+", "-", "×", "÷", "(", ")", "^"]:
            btn = QPushButton(simbolo)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _checked=False, s=simbolo: self._insertar_texto(s))
            operadores_layout.addWidget(btn)
        operadores_layout.addStretch()
        layout.addLayout(operadores_layout)

        funciones_layout = QHBoxLayout()
        for etiqueta, plantilla in [
            ("SI(...)", "SI(, , )"),
            ("REDONDEAR(...)", "REDONDEAR(, 2)"),
            ("ABS(...)", "ABS()"),
        ]:
            btn = QPushButton(etiqueta)
            btn.clicked.connect(lambda _checked=False, p=plantilla: self._insertar_texto(p))
            funciones_layout.addWidget(btn)
        funciones_layout.addStretch()
        layout.addLayout(funciones_layout)

        layout.addWidget(QLabel("Fórmula:"))
        self.formula_edit = QLineEdit()
        self.formula_edit.setPlaceholderText("Ej: [Ventas] - [Costos]")
        self.formula_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        layout.addWidget(self.formula_edit)

        ayuda = QLabel(TRANSFORM_FUNCIONES_AYUDA)
        ayuda.setObjectName("muted")
        ayuda.setWordWrap(True)
        layout.addWidget(ayuda)

        btn_preview = QPushButton("Vista previa")
        btn_preview.clicked.connect(self._vista_previa)
        layout.addWidget(btn_preview)

        self.lbl_preview = QLabel("")
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setObjectName("muted")
        layout.addWidget(self.lbl_preview)

        botones = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        botones.accepted.connect(self._aceptar)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _insertar_texto(self, texto):
        self.formula_edit.insert(texto)
        self.formula_edit.setFocus()

    def _traducir_formula(self, texto_formula):
        """Convierte la fórmula 'estilo Excel' (con [Columna], × y ÷) en una
        expresión evaluable de pandas que referencia siempre self.df."""
        expresion = texto_formula.replace("×", "*").replace("÷", "/").replace("^", "**")
        expresion = re.sub(r"\[([^\]]+)\]", lambda m: f"df[{m.group(1)!r}]", expresion)
        return expresion

    def _calcular(self):
        """Evalúa la fórmula actual. Devuelve (serie_resultado, None) o
        (None, mensaje_de_error)."""
        texto = self.formula_edit.text().strip()
        if not texto:
            return None, "Escribe o arma una fórmula primero."

        expresion = self._traducir_formula(texto)
        try:
            # Evaluador seguro basado en AST (lista blanca), nunca eval().
            resultado = evaluar_formula_segura(expresion, self.df)
        except FormulaSeguridadError as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"No se pudo calcular la fórmula: {exc}"

        if isinstance(resultado, pd.Series):
            pass
        elif isinstance(resultado, np.ndarray):
            try:
                resultado = pd.Series(resultado, index=self.df.index)
            except Exception as exc:
                return None, f"El resultado no calza con la cantidad de filas: {exc}"
        elif np.isscalar(resultado):
            resultado = pd.Series([resultado] * len(self.df), index=self.df.index)
        else:
            return None, "La fórmula no produjo un resultado utilizable como columna."

        return resultado, None

    def _vista_previa(self):
        resultado, error = self._calcular()
        if error:
            self.lbl_preview.setText(f"⚠ {error}")
            return
        muestra = resultado.head(5)
        partes = []
        for v in muestra:
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                partes.append(f"{v:,.2f}")
            else:
                partes.append(str(v))
        self.lbl_preview.setText("Primeras filas: " + " | ".join(partes))

    def _aceptar(self):
        nombre = self.nombre_edit.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Falta el nombre", "Ponle un nombre a la columna nueva.")
            return

        resultado, error = self._calcular()
        if error:
            QMessageBox.warning(self, "Fórmula inválida", error)
            return

        if nombre in self.df.columns:
            respuesta = QMessageBox.question(
                self, "Columna existente",
                f"Ya existe una columna llamada '{nombre}'. ¿Quieres reemplazarla?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if respuesta != QMessageBox.StandardButton.Yes:
                return

        df_resultado = self.df.copy()
        df_resultado[nombre] = resultado
        self.df_resultado = df_resultado
        self.nombre_columna = nombre
        self.accept()

    def get_resultado(self):
        return self.df_resultado, self.nombre_columna


# ----------------------------------------------------------------------------
# Pestaña Reporte: lienzo tipo Excel+Word para armar reportes
# ----------------------------------------------------------------------------
# El lienzo de Reporte se ve como una hoja de papel: SIEMPRE con estos colores,
# sin importar si el resto de la app está en modo claro u oscuro (igual que una
# hoja de Word no cambia de color al activar el modo oscuro de Windows).
