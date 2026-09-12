"""
correlaciones.py — Detección de correlaciones numéricas estadísticamente
significativas entre columnas de una misma tabla, para Hadar.

Esto es la Fase 1 del capítulo "Relaciones entre variables" de Narrativa:
deliberadamente NO busca relaciones temporales ni causales todavía -- eso
queda para más adelante, cuando se justifique el esfuerzo extra que hace
falta para hacerlo con el mismo nivel de rigor. Por ahora, esto responde
una pregunta más chica pero sólida: "¿qué columnas numéricas se mueven
juntas, con el respaldo estadístico para poder decirlo con confianza?"

Dos decisiones de diseño valen la pena explicar:

1. Se usa la correlación de Spearman (basada en el ORDEN de los valores),
   no la de Pearson (basada en los valores en sí). Spearman es más
   resistente a que un solo valor atípico infle o esconda una correlación
   real -- justo el tipo de dato con el que trabaja Hadar, donde ya
   sabemos que puede haber anomalías mezcladas con los datos normales.

2. El nivel de significancia se corrige por la cantidad de pares de
   columnas que se prueban (corrección de Bonferroni). Si no se hiciera
   esto, entre más columnas numéricas tenga la tabla, más "hallazgos"
   aparecerían solo por haber probado muchas combinaciones a la vez, no
   porque la relación sea real.

Ninguna palabra de este módulo dice "causa": eso es una decisión
consciente, no un descuido -- correlación no implica causalidad, y el
texto que arma narrativa.py lo deja explícito para quien lee el informe.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


UMBRAL_R_DEFECTO = 0.5      # por debajo de esto, ni se considera lo bastante fuerte
ALFA_DEFECTO = 0.05         # nivel de significancia ANTES de corregir por múltiples pruebas
MINIMO_FILAS_DEFECTO = 10   # con menos datos que esto, no hay base para decir nada
MAX_RESULTADOS_DEFECTO = 8  # tope de correlaciones a mostrar, para no saturar el informe


@dataclass
class CorrelacionEncontrada:
    columna_1: str
    columna_2: str
    r: float                  # coeficiente de Spearman, de -1 a 1
    p_valor: float             # ya comparado contra el umbral corregido (Bonferroni)
    n: int                     # cantidad de filas usadas (pares sin nulos en ninguna de las dos)
    direccion: str              # "positiva" | "negativa"
    fuerza: str                  # "moderada" | "fuerte" | "muy fuerte"

    def como_texto(self) -> str:
        return (
            f"'{self.columna_1}' y '{self.columna_2}' tienen una correlación {self.fuerza} "
            f"{self.direccion} (r = {self.r:.2f}, con {self.n:,} filas de respaldo)."
        )


def _fuerza_de(r_abs: float) -> str:
    if r_abs >= 0.8:
        return "muy fuerte"
    if r_abs >= 0.65:
        return "fuerte"
    return "moderada"


def detectar_correlaciones_significativas(
    df: pd.DataFrame,
    umbral_r: float = UMBRAL_R_DEFECTO,
    alfa: float = ALFA_DEFECTO,
    minimo_filas: int = MINIMO_FILAS_DEFECTO,
    max_resultados: int = MAX_RESULTADOS_DEFECTO,
) -> list[CorrelacionEncontrada]:
    """
    Recorre todos los pares de columnas numéricas de df y devuelve las que
    tienen una correlación de Spearman fuerte Y estadísticamente
    significativa (después de corregir por la cantidad de pares probados).

    Devuelve como mucho max_resultados, ordenadas de la correlación más
    fuerte a la más débil (en valor absoluto).
    """
    columnas_numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    pares = list(combinations(columnas_numericas, 2))
    if not pares:
        return []

    alfa_corregido = alfa / len(pares)  # Bonferroni: entre más pares se prueban, más estricto hay que ser

    encontradas: list[CorrelacionEncontrada] = []
    for col1, col2 in pares:
        sub = df[[col1, col2]].dropna()
        if len(sub) < minimo_filas:
            continue
        # si una de las dos columnas no varía (todos los valores iguales),
        # la correlación no está matemáticamente definida -- se salta sin error.
        if sub[col1].nunique() < 2 or sub[col2].nunique() < 2:
            continue

        r, p = stats.spearmanr(sub[col1], sub[col2])
        if np.isnan(r):
            continue
        if abs(r) < umbral_r or p >= alfa_corregido:
            continue

        encontradas.append(CorrelacionEncontrada(
            columna_1=str(col1), columna_2=str(col2),
            r=round(float(r), 3), p_valor=float(p), n=len(sub),
            direccion="positiva" if r > 0 else "negativa",
            fuerza=_fuerza_de(abs(r)),
        ))

    encontradas.sort(key=lambda c: abs(c.r), reverse=True)
    return encontradas[:max_resultados]


# ----------------------------------------------------------------------
# Demo / auto-test rápido
# ----------------------------------------------------------------------

def _demo():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(100, 15, n)
    df = pd.DataFrame({
        "precio_unitario": base,
        "impuesto": base * 0.19 + rng.normal(0, 1, n),          # correlación fuerte real
        "cantidad": rng.integers(1, 50, n),                      # sin relación con precio
        "descuento_pct": 100 - base + rng.normal(0, 5, n),        # correlación negativa real
        "id_interno": np.arange(n),                                # constante creciente, sin sentido de negocio
    })
    # le metemos un outlier grosero en precio_unitario, a ver si Spearman aguanta
    df.loc[5, "precio_unitario"] = 999999

    print("=== Correlaciones significativas detectadas ===")
    for c in detectar_correlaciones_significativas(df):
        print(" -", c.como_texto(), f"| p = {c.p_valor:.2e}")


if __name__ == "__main__":
    _demo()
