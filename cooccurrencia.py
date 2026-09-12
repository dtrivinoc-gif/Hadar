"""
cooccurrencia.py — Detección de anomalías que ocurren juntas más seguido
de lo que el azar explicaría, dentro de una misma tabla, para Hadar.

Idea 1 del roadmap de "Relaciones entre variables": en vez de mirar si
los VALORES de dos columnas se mueven juntos (eso lo hace
correlaciones.py), esto mira si las FALLAS/ATIPICIDADES de dos columnas
ocurren en las mismas filas más seguido de lo esperable por azar --
"focos de colapso simultáneo", no columnas parecidas.

Usa el test exacto de Fisher sobre la tabla de contingencia 2x2:

                      B atípica    B normal
    A atípica            a            b
    A normal              c            d

Se prefiere Fisher a chi-cuadrado porque es exacto incluso con pocos
casos (las anomalías, por definición, son pocas frente al total de
filas) -- chi-cuadrado asume muestras grandes y se vuelve poco confiable
justo en el escenario típico de este módulo.

Se reporta el "lift": cuánto se multiplica la probabilidad de que B sea
atípica cuando A lo es, comparado con la probabilidad base de que B sea
atípica. Igual que en correlaciones.py, el nivel de significancia se
corrige por la cantidad de pares de columnas probados (Bonferroni), y
esto NUNCA dice "causa" -- solo que fallan juntas más seguido de lo normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from scipy import stats


ALFA_DEFECTO = 0.05
MINIMO_ANOMALIAS_DEFECTO = 3   # con menos que esto de cada lado, el hallazgo no es confiable
MAX_RESULTADOS_DEFECTO = 8


@dataclass
class CoocurrenciaEncontrada:
    columna_1: str
    columna_2: str
    filas_ambas: int       # filas atípicas en LAS DOS columnas a la vez
    filas_solo_1: int
    filas_solo_2: int
    lift: float              # cuánto se multiplica la probabilidad de B dado A, vs. la base
    p_valor: float
    n_total: int

    def como_texto(self) -> str:
        return (
            f"Cuando '{self.columna_1}' es atípica, la probabilidad de que "
            f"'{self.columna_2}' también lo sea se multiplica por {self.lift:.1f} "
            f"({self.filas_ambas} fila(s) con las dos a la vez, sobre {self.n_total:,} en total)."
        )


def indices_atipicos_por_columna(anomalias: list[dict]) -> dict[str, set]:
    """
    A partir de la lista de anomalías que entrega
    SemanticAnomalyDetector.detect_all(), arma {columna: {índices atípicos}}.
    Una columna puede aparecer en más de una anomalía (ej. duplicados Y
    quiebre de patrón a la vez) -- se juntan todos sus índices atípicos.
    """
    por_columna: dict[str, set] = {}
    for a in anomalias:
        indices = a.get("indices_atipicos") or []
        if not indices:
            continue
        for col in a.get("columnas") or []:
            por_columna.setdefault(col, set()).update(indices)
    return por_columna


def columnas_con_anomalias_suficientes(
    anomalias: list[dict], minimo: int = MINIMO_ANOMALIAS_DEFECTO
) -> list[str]:
    """Columnas que tienen al menos `minimo` filas atípicas -- son las
    únicas con las que vale la pena siquiera intentar este análisis."""
    por_columna = indices_atipicos_por_columna(anomalias)
    return [c for c, idxs in por_columna.items() if len(idxs) >= minimo]


def detectar_coocurrencia_de_anomalias(
    n_total_filas: int,
    anomalias: list[dict],
    alfa: float = ALFA_DEFECTO,
    minimo_anomalias: int = MINIMO_ANOMALIAS_DEFECTO,
    max_resultados: int = MAX_RESULTADOS_DEFECTO,
) -> list[CoocurrenciaEncontrada]:
    """
    Prueba todos los pares de columnas con anomalías suficientes y
    devuelve las que fallan juntas más seguido de lo que el azar
    explicaría (lift > 1, con significancia real). Ordenadas de mayor a
    menor lift.
    """
    por_columna = indices_atipicos_por_columna(anomalias)
    columnas = [c for c in por_columna if len(por_columna[c]) >= minimo_anomalias]
    if len(columnas) < 2 or n_total_filas <= 0:
        return []

    pares = list(combinations(columnas, 2))
    alfa_corregido = alfa / len(pares)

    encontradas: list[CoocurrenciaEncontrada] = []
    for col1, col2 in pares:
        idx1, idx2 = por_columna[col1], por_columna[col2]
        a = len(idx1 & idx2)
        if a < 1:
            continue  # sin ninguna fila con ambas a la vez, no hay nada que reportar

        b = len(idx1 - idx2)
        c = len(idx2 - idx1)
        d = max(n_total_filas - a - b - c, 0)

        _, p = stats.fisher_exact([[a, b], [c, d]])
        if p >= alfa_corregido:
            continue

        prob_2_dado_1 = a / (a + b) if (a + b) else 0.0
        prob_2_base = (a + c) / n_total_filas if n_total_filas else 0.0
        if prob_2_base <= 0:
            continue
        lift = prob_2_dado_1 / prob_2_base
        if lift <= 1:
            continue  # acá solo interesa "fallan juntas más", no "se evitan"

        encontradas.append(CoocurrenciaEncontrada(
            columna_1=str(col1), columna_2=str(col2),
            filas_ambas=a, filas_solo_1=b, filas_solo_2=c,
            lift=round(float(lift), 2), p_valor=float(p), n_total=n_total_filas,
        ))

    encontradas.sort(key=lambda c: c.lift, reverse=True)
    return encontradas[:max_resultados]


# ----------------------------------------------------------------------
# Demo / auto-test rápido
# ----------------------------------------------------------------------

def _demo():
    # 500 filas; dos columnas cuyas anomalías coinciden casi siempre
    # (20 filas en común de 25 anomalías cada una), y una tercera columna
    # con anomalías totalmente independientes de las otras dos.
    n = 500
    anomalias = [
        {"columnas": ["precio_unitario"], "indices_atipicos": list(range(0, 25))},
        {"columnas": ["margen"], "indices_atipicos": list(range(5, 30))},          # se solapa con precio_unitario
        {"columnas": ["stock"], "indices_atipicos": list(range(400, 425))},         # independiente
    ]
    print("=== Co-ocurrencias detectadas ===")
    for c in detectar_coocurrencia_de_anomalias(n, anomalias):
        print(" -", c.como_texto(), f"| p = {c.p_valor:.2e}")


if __name__ == "__main__":
    _demo()
