"""
deteccion_multivariada.py — Detección de anomalías por COMBINACIÓN de
columnas numéricas (a diferencia de anomalias.py, que revisa cada columna
por separado, y de aprendizaje_adaptativo.py, que compara contra el
historial del proyecto en el tiempo). Usa Isolation Forest (scikit-learn):
un modelo liviano, sin necesitar datos etiquetados, que aísla una fila
más rápido cuanto más distinta es de las demás -- las filas "raras por
combinación" quedan separadas con menos particiones que las típicas.

Esto NO reemplaza nada de lo que ya existe: es un chequeo adicional, para
el caso que los otros no cubren -- ninguna columna individual se ve mal
por separado, pero la combinación puntual de varias sí. Ejemplo: una
mantención con duración normal Y costo normal cada uno por su lado, pero
esa combinación específica (duración corta + costo alto) nunca había
pasado junta antes.

Tres decisiones de diseño explícitas, por la filosofía de Hadar:

1. DETERMINISMO: random_state fijo en todas las corridas de Isolation
   Forest. Mismos datos, mismo resultado siempre -- nunca dos corridas
   distintas sobre el mismo archivo.

2. EXPLICABILIDAD PARCIAL: Isolation Forest por sí solo da un score ("qué
   tan aislada quedó la fila") sin decir POR QUÉ. Para no romper el
   principio de "auditable" de Hadar, cada hallazgo se acompaña de las
   1-2 columnas que más se alejan de su propio valor típico en esa fila
   (mismo criterio robusto de Mediana/MAD que usa anomalias.py). No es
   una explicación perfecta del modelo interno, pero le da al usuario una
   pista concreta y verificable de por dónde mirar, en vez de un "confía
   en mí".

3. OPT-IN: este chequeo nunca corre solo. Necesita que el proyecto lo
   tenga activado explícitamente (igual que aprendizaje_adaptativo.py),
   y solo tiene sentido activarlo cuando evaluar_confiabilidad() da un
   resultado aceptable para ESE dataset -- ver el "termómetro" más abajo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MINIMO_FILAS_DEFECTO = 50          # con menos filas, "raro" no significa nada todavía
MINIMO_COLUMNAS_DEFECTO = 2        # con 1 sola columna numérica esto ya lo cubre anomalias.py
CONTAMINACION_DEFECTO = 0.02       # % de filas que se espera sean anómalas, de partida
N_ARBOLES_DEFECTO = 150
N_ARBOLES_DIAGNOSTICO = 50         # menos árboles para el auto-test: se corre muchas veces,
                                    # no hace falta la misma precisión que la corrida real
RANDOM_STATE_DEFECTO = 0           # fijo a propósito -- ver nota de determinismo arriba
MAX_COLUMNAS_RESPONSABLES = 2      # cuántas columnas se mencionan como "las más raras" por fila
MAX_FILAS_DIAGNOSTICO = 20000      # tope de filas para el auto-test en hardware modesto
                                    # (se muestrea, no se recorta el análisis real)
UMBRAL_CONFIABLE = 0.75            # tasa de detección mínima para recomendar activar el chequeo


@dataclass
class AnomaliaMultivariada:
    indice: object                # índice de fila en el DataFrame original
    score: float                  # score crudo de Isolation Forest (más negativo = más raro)
    columnas_responsables: list   # [(columna, z_robusto), ...] de más a menos raro


def _columnas_numericas_utiles(df: pd.DataFrame) -> list:
    """Columnas numéricas con variación real -- una columna constante no
    aporta nada a 'qué tan distinta es esta fila de las demás' y además
    puede romper la normalización interna del modelo."""
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]


def _z_robustos_por_columna(df_num: pd.DataFrame) -> pd.DataFrame:
    """Z-score robusto (Mediana/MAD) de cada valor contra su propia
    columna -- mismo criterio que usa anomalias.py, para que la 'pista'
    que se muestra sea consistente con el resto de la app."""
    medianas = df_num.median()
    mad = (df_num - medianas).abs().median()
    mad_seguro = mad.replace(0, np.nan)  # evita dividir por cero en columnas casi constantes
    z = 0.6745 * (df_num - medianas) / mad_seguro
    return z.fillna(0.0)


def detectar_anomalias_multivariadas(
    df: pd.DataFrame,
    contaminacion: float = CONTAMINACION_DEFECTO,
    minimo_filas: int = MINIMO_FILAS_DEFECTO,
    minimo_columnas: int = MINIMO_COLUMNAS_DEFECTO,
    n_arboles: int = N_ARBOLES_DEFECTO,
    random_state: int = RANDOM_STATE_DEFECTO,
) -> list[AnomaliaMultivariada]:
    """
    Devuelve las filas que Isolation Forest considera raras por la
    COMBINACIÓN de sus columnas numéricas, cada una con sus 1-2 columnas
    "más responsables" como pista de explicación.

    Devuelve lista vacía (sin error) si no hay suficientes filas o
    columnas numéricas útiles -- este chequeo simplemente no aporta nada
    con datasets chicos o de una sola dimensión, así que se salta en
    silencio en vez de forzar un resultado poco confiable.
    """
    columnas = _columnas_numericas_utiles(df)
    if len(df) < minimo_filas or len(columnas) < minimo_columnas:
        return []

    df_num = df[columnas].apply(pd.to_numeric, errors="coerce")
    validas = df_num.dropna()
    if len(validas) < minimo_filas:
        return []

    modelo = IsolationForest(
        n_estimators=n_arboles,
        contamination=contaminacion,
        random_state=random_state,
    )
    etiquetas = modelo.fit_predict(validas)          # -1 = anómalo, 1 = normal
    scores = modelo.score_samples(validas)           # más negativo = más raro

    z_robustos = _z_robustos_por_columna(validas)

    resultados = []
    for pos, idx in enumerate(validas.index):
        if etiquetas[pos] != -1:
            continue
        fila_z = z_robustos.loc[idx].abs().sort_values(ascending=False)
        columnas_responsables = [
            (col, float(fila_z[col])) for col in fila_z.index[:MAX_COLUMNAS_RESPONSABLES]
        ]
        resultados.append(AnomaliaMultivariada(
            indice=idx, score=float(scores[pos]), columnas_responsables=columnas_responsables,
        ))

    resultados.sort(key=lambda a: a.score)  # más raro primero (score más negativo)
    return resultados


# ----------------------------------------------------------------------
# Termómetro de confiabilidad: auto-test con anomalías sintéticas
# ----------------------------------------------------------------------

def evaluar_confiabilidad(
    df: pd.DataFrame,
    n_pruebas: int = 30,
    contaminacion: float = CONTAMINACION_DEFECTO,
    minimo_filas: int = MINIMO_FILAS_DEFECTO,
    minimo_columnas: int = MINIMO_COLUMNAS_DEFECTO,
    random_state: int = RANDOM_STATE_DEFECTO,
) -> dict | None:
    """
    Mide qué tan bien detecta el modelo anomalías REALES en ESTE dataset
    específico -- no un número de una tabla genérica, sino un resultado
    medido ahora mismo sobre los datos del usuario. Esto es lo que
    alimenta el "termómetro" que decide si vale la pena ofrecer el botón
    de activar ML para este análisis.

    Cómo: toma filas al azar del dataset, les fuerza un valor extremo en
    una columna numérica al azar, y cuenta cuántas de esas filas
    "fabricadas como raras" el modelo efectivamente marca como anómalas.
    Es lo más parecido a medir precisión real sin necesitar que el
    usuario etiquete nada a mano -- porque acá SÍ sabemos cuál es la
    respuesta correcta (la inventamos nosotros).

    Con datasets grandes, la prueba se corre sobre una MUESTRA (tope
    MAX_FILAS_DIAGNOSTICO) para no reentrenar el modelo decenas de veces
    sobre millones de filas en una PC modesta -- el análisis real
    (detectar_anomalias_multivariadas) sigue usando el dataset completo,
    esto es solo para medir qué tan confiable sería activarlo.

    Devuelve None si el dataset es muy chico para siquiera intentarlo
    (mismo criterio que detectar_anomalias_multivariadas). Si no, un
    dict con la tasa de detección y si alcanza el umbral recomendado.
    """
    columnas = _columnas_numericas_utiles(df)
    if len(df) < minimo_filas or len(columnas) < minimo_columnas:
        return None

    df_num = df[columnas].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df_num) < minimo_filas:
        return None
    # float64 en toda la copia de trabajo: columnas enteras (ej. cantidades)
    # no aceptan el valor extremo fraccionario que fabrica la prueba.
    df_num = df_num.astype(float)

    rng = np.random.default_rng(random_state)
    if len(df_num) > MAX_FILAS_DIAGNOSTICO:
        df_num = df_num.sample(MAX_FILAS_DIAGNOSTICO, random_state=random_state)

    n_pruebas = min(n_pruebas, max(5, len(df_num) // 10))  # no más del 10% del dataset
    filas_elegidas = rng.choice(df_num.index, size=n_pruebas, replace=False)

    detectados = 0
    for fila_idx in filas_elegidas:
        copia = df_num.copy()
        columna_objetivo = rng.choice(columnas)
        valor_original = copia.loc[fila_idx, columna_objetivo]
        desviacion = copia[columna_objetivo].std()
        if not desviacion or pd.isna(desviacion):
            desviacion = abs(valor_original) or 1.0
        # empuja el valor bien lejos de lo normal, alternando la dirección
        # para no fabricar siempre anomalías "hacia arriba"
        signo = 1 if rng.random() > 0.5 else -1
        copia.loc[fila_idx, columna_objetivo] = valor_original + signo * desviacion * 8

        modelo = IsolationForest(
            n_estimators=N_ARBOLES_DIAGNOSTICO,
            contamination=contaminacion,
            random_state=random_state,
        )
        etiquetas = modelo.fit_predict(copia)
        posicion = copia.index.get_loc(fila_idx)
        if etiquetas[posicion] == -1:
            detectados += 1

    tasa = detectados / n_pruebas
    return {
        "tasa_deteccion": round(tasa, 3),
        "n_pruebas": n_pruebas,
        "n_filas_dataset": len(df_num),
        "confiable": tasa >= UMBRAL_CONFIABLE,
    }


# ----------------------------------------------------------------------
# Demo / auto-test rápido
# ----------------------------------------------------------------------

def _demo():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "duracion_horas": rng.normal(4, 1, n).clip(0.5),
        "costo": rng.normal(200, 40, n),
        "cantidad_repuestos": rng.integers(1, 6, n),
    })
    # combinación rara a propósito: duración muy corta + costo muy alto,
    # cada columna por separado no se ve tan mal
    df.loc[10, "duracion_horas"] = 0.6
    df.loc[10, "costo"] = 950

    print("=== Anomalías multivariadas detectadas ===")
    for a in detectar_anomalias_multivariadas(df):
        print(f" - fila {a.indice}: score={a.score:.3f}, columnas={a.columnas_responsables}")

    print("\n=== Termómetro de confiabilidad ===")
    print(evaluar_confiabilidad(df))


if __name__ == "__main__":
    _demo()
