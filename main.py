"""
Punto de entrada de la aplicación. Ejecutar con: python -m hadar.main
(o crear un lanzador run_hadar.py en la raíz del proyecto, ver más abajo).
"""
import sys

from PySide6.QtWidgets import QApplication, QDialog

from .main_window import HadarApp
from .pantalla_inicio import PantallaInicio

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pantalla = PantallaInicio()
    if pantalla.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)  # cerró la pantalla de inicio sin elegir nada

    window = HadarApp()
    if pantalla.modo == "continuar" and pantalla.ruta_proyecto:
        window.abrir_proyecto_desde_ruta(pantalla.ruta_proyecto)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()