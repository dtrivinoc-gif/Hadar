"""
dialogos_limpieza.py — Diálogo de revisión antes de eliminar caracteres
especiales, para Hadar.

De las cuatro correcciones de limpieza.py, esta es la única que pide
revisión explícita carácter por carácter en vez de un solo botón
"Aplicar": un '@' fuera de lugar en un correo o un '-' en un RUT no es lo
mismo que un '#' pegado por error en un nombre, y la diferencia depende
del contexto de cada columna -- no se puede adivinar en general.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton,
)


class DialogoRevisarCaracteres(QDialog):
    """Lista cada (columna, carácter) encontrado con casilla de marcado.
    Al aceptar, self.claves_elegidas queda con el set de (columna,
    carácter) que la persona decidió eliminar -- nunca se aplica nada
    dentro del diálogo mismo, solo se recoge la elección."""

    def __init__(self, hallazgos_caracteres, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Revisar caracteres a eliminar")
        self.resize(480, 380)
        self.claves_elegidas: set[tuple[str, str]] = set()

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Marca qué caracteres quieres eliminar. Se sacan SOLO del texto de las\n"
            "celdas marcadas -- nada más de la tabla cambia. Revisa antes de aplicar:\n"
            "un símbolo puede ser basura en una columna y parte real del dato en otra."
        ))

        self.lista = QListWidget()
        for h in hallazgos_caracteres:
            caracter = h.detalle.get("caracter", "")
            ejemplos = h.detalle.get("ejemplos") or []
            ejemplo = ejemplos[0] if ejemplos else ""
            item = QListWidgetItem(
                f"'{h.columna}': quitar {caracter!r}  —  {len(h.filas)} celda(s), ej: {ejemplo!r}"
            )
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, (h.columna, caracter))
            self.lista.addItem(item)
        layout.addWidget(self.lista)

        fila_marcado = QHBoxLayout()
        btn_todos = QPushButton("Marcar todos")
        btn_todos.clicked.connect(lambda: self._marcar_todos(Qt.Checked))
        btn_ninguno = QPushButton("Desmarcar todos")
        btn_ninguno.clicked.connect(lambda: self._marcar_todos(Qt.Unchecked))
        fila_marcado.addWidget(btn_todos)
        fila_marcado.addWidget(btn_ninguno)
        layout.addLayout(fila_marcado)

        fila_final = QHBoxLayout()
        fila_final.addStretch()
        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.clicked.connect(self.reject)
        btn_aplicar = QPushButton("Aplicar los marcados")
        btn_aplicar.setObjectName("accentButton")
        btn_aplicar.clicked.connect(self._confirmar)
        fila_final.addWidget(btn_cancelar)
        fila_final.addWidget(btn_aplicar)
        layout.addLayout(fila_final)

    def _marcar_todos(self, estado):
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(estado)

    def _confirmar(self):
        self.claves_elegidas = {
            self.lista.item(i).data(Qt.UserRole)
            for i in range(self.lista.count())
            if self.lista.item(i).checkState() == Qt.Checked
        }
        self.accept()
