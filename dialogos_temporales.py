"""
Diálogo donde el usuario configura a mano relaciones de orden temporal
entre columnas de fecha (ej. "aprobación" antes que "entrega"), usadas
por el detector de anomalías y guardadas en MemoriaHadar.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QDialog, QDialogButtonBox,
)

class ConfigurarRelacionesTemporalesDialog(QDialog):
    """Deja que el usuario le enseñe a la app relaciones de orden temporal
    que no calzan con los tokens fijos de TOKENS_TEMPORALES_PARES: elige dos
    columnas de fecha REALES del dataset actual y dice "esta va antes que
    esta otra". No requiere que el nombre de columna contenga ninguna
    palabra en particular -- el usuario apunta directo a sus columnas."""

    def __init__(self, columnas_fecha, pares_iniciales, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar relaciones de fechas")
        self.setMinimumWidth(520)
        self.columnas_fecha = list(columnas_fecha)
        self._filas = []  # cada item: (combo_anterior, combo_posterior, input_etiqueta, fila_widget)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Dile a Hadar qué fecha debe ocurrir antes que cuál otra en tus datos "
            "(ej. 'aprobación' antes que 'entrega'). Esto se suma a lo que la app "
            "ya detecta automáticamente y queda guardado para este mismo tipo de dataset."
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        if len(self.columnas_fecha) < 2:
            layout.addWidget(QLabel(
                "Se necesitan al menos dos columnas de fecha en los datos cargados "
                "para configurar una relación."
            ))

        self.filas_container = QVBoxLayout()
        layout.addLayout(self.filas_container)

        for par in pares_iniciales:
            self._agregar_fila(par.get("col_anterior"), par.get("col_posterior"), par.get("etiqueta"))
        if not pares_iniciales:
            self._agregar_fila()

        btn_agregar = QPushButton("+ Agregar relación")
        btn_agregar.clicked.connect(lambda: self._agregar_fila())
        layout.addWidget(btn_agregar)

        botones = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def _agregar_fila(self, col_anterior=None, col_posterior=None, etiqueta=None):
        fila_widget = QWidget()
        fila = QHBoxLayout(fila_widget)
        fila.setContentsMargins(0, 0, 0, 0)

        combo_anterior = QComboBox()
        combo_anterior.addItems(self.columnas_fecha)
        if col_anterior and col_anterior in self.columnas_fecha:
            combo_anterior.setCurrentText(col_anterior)

        lbl_flecha = QLabel("debe ir antes que")

        combo_posterior = QComboBox()
        combo_posterior.addItems(self.columnas_fecha)
        if col_posterior and col_posterior in self.columnas_fecha:
            combo_posterior.setCurrentText(col_posterior)
        elif len(self.columnas_fecha) > 1:
            combo_posterior.setCurrentIndex(1)

        input_etiqueta = QLineEdit()
        input_etiqueta.setPlaceholderText("Nombre de la relación (opcional, ej. 'aprobación → entrega')")
        if etiqueta:
            input_etiqueta.setText(etiqueta)

        btn_quitar = QPushButton("✕")
        btn_quitar.setFixedWidth(28)
        btn_quitar.clicked.connect(lambda: self._quitar_fila(fila_widget))

        fila.addWidget(combo_anterior, stretch=1)
        fila.addWidget(lbl_flecha)
        fila.addWidget(combo_posterior, stretch=1)
        fila.addWidget(input_etiqueta, stretch=2)
        fila.addWidget(btn_quitar)

        self.filas_container.addWidget(fila_widget)
        self._filas.append((combo_anterior, combo_posterior, input_etiqueta, fila_widget))

    def _quitar_fila(self, fila_widget):
        self._filas = [f for f in self._filas if f[3] is not fila_widget]
        fila_widget.setParent(None)
        fila_widget.deleteLater()

    def obtener_pares(self):
        """Devuelve la lista de pares configurados (ignora filas donde ambas
        columnas quedaron iguales, que no tienen sentido como relación)."""
        pares = []
        for combo_anterior, combo_posterior, input_etiqueta, _ in self._filas:
            col_anterior = combo_anterior.currentText()
            col_posterior = combo_posterior.currentText()
            if not col_anterior or not col_posterior or col_anterior == col_posterior:
                continue
            pares.append({
                "col_anterior": col_anterior,
                "col_posterior": col_posterior,
                "etiqueta": input_etiqueta.text().strip() or f"{col_anterior} → {col_posterior}",
            })
        return pares


