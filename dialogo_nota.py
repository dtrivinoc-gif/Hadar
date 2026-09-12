"""
Diálogo simple para agregar o editar la nota de una celda de la tabla
(pestaña Datos), con la opción de marcarla como anomalía.
"""
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QTextEdit, QCheckBox,
)

class DialogoNota(QDialog):
    """Diálogo simple para agregar/editar la nota de una celda de la tabla
    (pestaña Datos). Dejar el texto vacío y guardar borra la nota."""

    def __init__(self, fila_visible, col_name, valor, nota_existente="",
                 es_anomalia_existente=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Nota - Fila {fila_visible + 1}, Columna '{col_name}'")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        info = QLabel(f"📌 Celda: Fila {fila_visible + 1}, Columna '{col_name}'")
        info.setStyleSheet("font-weight: bold;")
        layout.addWidget(info)

        layout.addWidget(QLabel(f"Valor: {valor}"))
        layout.addWidget(QLabel("Nota:"))

        self.texto = QTextEdit()
        self.texto.setPlaceholderText("Escribe tu nota aquí... (vacío = quitar la nota)")
        self.texto.setMinimumHeight(100)
        if nota_existente:
            self.texto.setText(nota_existente)
        layout.addWidget(self.texto)

        self.chk_anomalia = QCheckBox("🚨 Marcar como anomalía")
        self.chk_anomalia.setChecked(es_anomalia_existente)
        layout.addWidget(self.chk_anomalia)

        botones = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        botones.accepted.connect(self.accept)
        botones.rejected.connect(self.reject)
        layout.addWidget(botones)

    def get_nota(self):
        return self.texto.toPlainText().strip()

    def es_anomalia(self):
        return self.chk_anomalia.isChecked()


