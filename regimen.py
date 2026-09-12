"""
regimen.py — Detección de cambios de comportamiento (quiebres de régimen)
en columnas numéricas a lo largo del tiempo, para Hadar.

Idea 2 del roadmap de "Relaciones entre variables": en vez de preguntar
si dos variables se relacionan (correlaciones.py) o si sus fallas
coinciden (cooccurrencia.py), esto pregunta si el comportamiento TÍPICO
de una sola variable cambió en algún punto del tiempo -- ej. "hasta el 12
de agosto rondaba los $15.000, desde entonces ronda los $9.000".

Usa el test de Pettitt: no asume que los datos siguen una distribución en
particular, y en vez de pedir que se elija a mano una fecha de corte para
comparar "antes" contra "después", el test ENCUENTRA el punto donde el
quiebre es más probable, y dice qué tan significativo es. Es una
herramienta estadística clásica para esto (se usa hace décadas en control
de calidad y series ambientales), no una invención para Hadar.

Alcance consciente: esto mide si UNA columna cambió de comportamiento,
no si la RELACIÓN/razón entre dos columnas cambió (ej. "unidades
vendidas por cada producto"). Esto último exigiría elegir qué par de
columnas dividir entre sí, lo cual depende del negocio de cada usuario y
no se puede adivinar en general sin arriesgarse a mostrar razones sin
sentido (ej. dividir un precio por un ID). Queda como una posible
ampliación futura, específica por dominio.

Requiere una columna de fecha (se detecta con detectar_columnas_fecha, en
anomalias.py, igual que en el resto de Narrativa) para poder ordenar
cronológicamente -- sin fecha, esto no se puede calcular y no aparece.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ALFA_DEFECTO = 0.05
MINIMO_FILAS_DEFECTO = 20      # con menos que esto, un quiebre "detectado" no es confiable
MAX_RESULTADOS_DEFECTO = 5
MINIMO_FRACCION_LADO = 0.1     # el quiebre debe dejar al menos 10% de los datos a cada lado


@dataclass
class QuiebreEncontrado:
    columna: str
    columna_fecha: str
    fecha_quiebre: object            # valor de la columna de fecha en el punto de quiebre
    mediana_antes: float
    mediana_despues: float
    n_antes: int
    n_despues: int
    p_valor: float

    def como_texto(self) -> str:
        cambio = "subió" if self.mediana_despues > self.mediana_antes else "bajó"
        return (
            f"'{self.columna}' cambió de comportamiento alrededor del {self._fecha_legible()}: "
            f"la mediana {cambio} de {self.mediana_antes:,.2f} a {self.mediana_despues:,.2f} "
            f"({self.n_antes:,} registros antes, {self.n_despues:,} después)."
        )

    def _fecha_legible(self) -> str:
        try:
            return pd.Timestamp(self.fecha_quiebre).strftime("%d-%m-%Y")
        except Exception:
            return str(self.fecha_quiebre)


def _pettitt(valores: np.ndarray):
    """
    Test de Pettitt. Devuelve (posicion_del_quiebre, p_valor), donde
    posicion_del_quiebre es la cantidad de filas que quedan en el grupo
    "antes" (0-based). Devuelve None si hay muy pocos datos.
    """
    n = len(valores)
    if n < MINIMO_FILAS_DEFECTO:
        return None

    rangos = pd.Series(valores).rank().values
    acumulado = np.cumsum(rangos)
    t = np.arange(1, n)
    U = 2 * acumulado[:-1] - t * (n + 1)
    K = float(np.max(np.abs(U)))
    posicion = int(t[int(np.argmax(np.abs(U)))])

    # Aproximación estándar del p-valor del test de Pettitt.
    p = 2 * np.exp((-6 * (K ** 2)) / (n ** 3 + n ** 2))
    return posicion, min(float(p), 1.0)


def detectar_quiebres_de_comportamiento(
    df: pd.DataFrame,
    columna_fecha: str,
    alfa: float = ALFA_DEFECTO,
    minimo_filas: int = MINIMO_FILAS_DEFECTO,
    max_resultados: int = MAX_RESULTADOS_DEFECTO,
) -> list[QuiebreEncontrado]:
    """
    Para cada columna numérica de df (excepto columna_fecha), ordenada
    cronológicamente, busca el punto donde su comportamiento típico
    cambió de forma más marcada, y se queda con los estadísticamente
    significativos después de corregir por la cantidad de columnas
    probadas (Bonferroni). Además exige que el quiebre no esté pegado a
    una punta de los datos (si no, "el quiebre" sería solo el primer o
    último registro, no un cambio real de comportamiento).
    """
    if columna_fecha not in df.columns:
        return []

    fechas = pd.to_datetime(df[columna_fecha], errors="coerce", format="mixed")
    columnas_numericas = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != columna_fecha
    ]
    if not columnas_numericas:
        return []

    alfa_corregido = alfa / len(columnas_numericas)

    encontrados: list[QuiebreEncontrado] = []
    for col in columnas_numericas:
        sub = pd.DataFrame({"fecha": fechas, "valor": df[col]}).dropna()
        sub = sub.sort_values("fecha")
        if len(sub) < minimo_filas:
            continue

        resultado = _pettitt(sub["valor"].values)
        if resultado is None:
            continue
        posicion, p = resultado

        n = len(sub)
        if posicion < n * MINIMO_FRACCION_LADO or posicion > n * (1 - MINIMO_FRACCION_LADO):
            continue
        if p >= alfa_corregido:
            continue

        antes = sub.iloc[:posicion]
        despues = sub.iloc[posicion:]

        encontrados.append(QuiebreEncontrado(
            columna=str(col), columna_fecha=columna_fecha,
            fecha_quiebre=despues["fecha"].iloc[0],
            mediana_antes=float(antes["valor"].median()),
            mediana_despues=float(despues["valor"].median()),
            n_antes=len(antes), n_despues=len(despues), p_valor=p,
        ))

    encontrados.sort(key=lambda q: q.p_valor)
    return encontrados[:max_resultados]


# ----------------------------------------------------------------------
# Demo / auto-test rápido
# ----------------------------------------------------------------------

def _demo():
    rng = np.random.default_rng(0)
    n = 200
    fechas = pd.date_range("2025-01-01", periods=n, freq="D")
    # precio_unitario: estable ~100 hasta la fila 150, después salta a ~160
    valores = np.concatenate([rng.normal(100, 5, 150), rng.normal(160, 5, n - 150)])
    cantidad = rng.integers(1, 50, n)  # sin quiebre, puro ruido

    df = pd.DataFrame({"fecha_venta": fechas, "precio_unitario": valores, "cantidad": cantidad})

    print("=== Quiebres de comportamiento detectados ===")
    for q in detectar_quiebres_de_comportamiento(df, "fecha_venta"):
        print(" -", q.como_texto(), f"| p = {q.p_valor:.2e}")


if __name__ == "__main__":
    _demo()
