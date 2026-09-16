"""
deteccion_multivariada.py — Detección de anomalías por COMBINACIÓN de
columnas numéricas (a diferencia de anomalias.py, que revisa cada columna
por separado, y de aprendizaje_adaptativo.py, que compara contra el
historial del proyecto en el tiempo). Usa Isolation Forest (scikit-learn):
un modelo liviano, sin necesitar datos etiquetados, que aísla una fila
más rápido cuanto más distinta es de las demás.

Esto NO reemplaza nada de lo que ya existe: es un chequeo adicional, para
el caso que los otros no cubren -- ninguna columna individual se ve mal
por separado, pero la combinación puntual de varias sí.

Decisiones de diseño, en orden de cómo se fueron corrigiendo (queda
documentado el porqué, no solo el qué -- por si el día de mañana alguien
repite alguno de estos errores):

1. DETERMINISMO: random_state fijo en todas las corridas.

2. CONTAMINATION, no un umbral propio sobre el score: se probó reemplazar
   "contamination" por un Z-score robusto (Mediana/MAD) sobre el score
   crudo de Isolation Forest, pensando que sería más "a la Hadar" que un
   parámetro nativo de sklearn. Medido y descartado: el score de Isolation
   Forest no se distribuye como una columna de datos normal, y ese
   criterio marcaba ~5% de un dataset 100% limpio como anómalo (falsos
   positivos). "contamination" es el uso estándar y probado de la
   librería -- se mantiene, con la salvedad clara de que devuelve "las K
   filas más raras de este dataset", un tope acotado y predecible, no una
   promesa de que existan exactamente esa cantidad de anomalías reales.

3. EXPLICABILIDAD ROBUSTA A COLUMNAS CASI CONSTANTES: la primera versión
   de la pista "qué columnas explican esta fila" usaba Mediana/MAD y, si
   una columna tenía MAD=0 (más de la mitad de sus valores idénticos --
   común en columnas con muchos ceros), dividía por cero y el resultado se
   reemplazaba por 0.0. Eso significaba que la columna que de verdad
   causaba el aislamiento (por ser distinta a un valor por lo demás
   constante) quedaba invisible en la explicación. Ahora, cuando MAD=0,
   cualquier valor distinto de la mediana recibe un z fijo alto en vez de
   0 -- no se puede graduar "qué tan raro" sin variación de referencia,
   pero al menos no se oculta que ESA columna es la responsable.

4. AUTO-TEST (evaluar_confiabilidad) BATCHEADO Y REALMENTE MULTIVARIADO:
   la primera versión reentrenaba Isolation Forest una vez POR CADA fila
   de prueba (ej. 30 reentrenos), algo lento e innecesario -- ahora se
   inyectan todas las filas sintéticas en una sola copia del dataset y se
   entrena una sola vez. Además, la prueba original empujaba una sola
   columna a +-8 desviaciones, lo cual mide detección UNIVARIADA (algo que
   ya cubre anomalias.py), no la combinación de columnas que es la razón
   de ser de este módulo. Ahora se mueven 2 columnas a la vez, cada una
   por debajo del umbral que ya usaría anomalias.py sola (así ninguna
   columna es individualmente "anómala"), y lo único raro es la
   combinación -- que es exactamente lo que este chequeo existe para
   encontrar.

5. OPT-IN: nunca corre solo, necesita que el proyecto lo tenga activado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MINIMO_FILAS_DEFECTO = 50
MINIMO_COLUMNAS_DEFECTO = 2
CONTAMINACION_DEFECTO = 0.02
UMBRAL_Z_DEFECTO = 3.0              # mismo umbral por defecto que usa anomalias.py (umbral_z)
N_ARBOLES_DEFECTO = 150
N_ARBOLES_DIAGNOSTICO = 100
RANDOM_STATE_DEFECTO = 0
MAX_COLUMNAS_RESPONSABLES = 2
MAX_FILAS_DIAGNOSTICO = 20000
UMBRAL_CONFIABLE = 0.75
Z_MAD_CERO = 10.0                   # z asignado cuando una columna casi constante (MAD=0)
                                     # se sale de su único valor típico -- ver nota #3 arriba


@dataclass
class AnomaliaMultivariada:
    indice: object
    score: float                   # score crudo de Isolation Forest (más negativo = más raro)
    columnas_responsables: list    # [(columna, z_robusto), ...] de más a menos raro


def _columnas_numericas_utiles(df: pd.DataFrame) -> list:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]


def _z_robustos_por_columna(df_num: pd.DataFrame) -> pd.DataFrame:
    """Z-score robusto (Mediana/MAD) de cada valor contra su propia
    columna. Si la columna es casi constante (MAD=0), cualquier valor
    distinto de la mediana recibe Z_MAD_CERO en vez de 0 -- ver nota de
    diseño #3 arriba: MAD=0 no significa 'nada raro acá', significa que no
    hay forma de graduar qué tan raro es, así que se marca como muy raro
    en vez de invisible."""
    resultado = {}
    for col in df_num.columns:
        valores = df_num[col].to_numpy()
        mediana = np.median(valores)
        mad = np.median(np.abs(valores - mediana))
        if mad > 0:
            resultado[col] = 0.6745 * (valores - mediana) / mad
        else:
            resultado[col] = np.where(valores != mediana, Z_MAD_CERO, 0.0)
    return pd.DataFrame(resultado, index=df_num.index)


def detectar_anomalias_multivariadas(
    df: pd.DataFrame,
    contaminacion: float = CONTAMINACION_DEFECTO,
    minimo_filas: int = MINIMO_FILAS_DEFECTO,
    minimo_columnas: int = MINIMO_COLUMNAS_DEFECTO,
    n_arboles: int = N_ARBOLES_DEFECTO,
    random_state: int = RANDOM_STATE_DEFECTO,
) -> list[AnomaliaMultivariada]:
    """Devuelve las filas que Isolation Forest considera raras por la
    COMBINACIÓN de sus columnas numéricas -- en la práctica, hasta
    aproximadamente `contaminacion` de las filas del dataset (ver nota de
    diseño #2: es un tope acotado, no una cuenta exacta de anomalías
    reales). Cada hallazgo trae sus 1-2 columnas "más responsables" como
    pista de explicación."""
    columnas = _columnas_numericas_utiles(df)
    if len(df) < minimo_filas or len(columnas) < minimo_columnas:
        return []

    df_num = df[columnas].apply(pd.to_numeric, errors="coerce")
    validas = df_num.dropna()
    if len(validas) < minimo_filas:
        return []

    modelo = IsolationForest(n_estimators=n_arboles, contamination=contaminacion,
                              random_state=random_state)
    etiquetas = modelo.fit_predict(validas)
    scores = modelo.score_samples(validas)
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

    resultados.sort(key=lambda a: a.score)
    return resultados


# ----------------------------------------------------------------------
# Termómetro de confiabilidad: auto-test con anomalías sintéticas
# ----------------------------------------------------------------------

def evaluar_confiabilidad(
    df: pd.DataFrame,
    n_pruebas: int = 30,
    contaminacion: float = CONTAMINACION_DEFECTO,
    umbral_z: float = UMBRAL_Z_DEFECTO,
    minimo_filas: int = MINIMO_FILAS_DEFECTO,
    minimo_columnas: int = MINIMO_COLUMNAS_DEFECTO,
    random_state: int = RANDOM_STATE_DEFECTO,
) -> dict | None:
    """Mide, sobre ESTE dataset, qué tan bien detectaría el chequeo real
    una anomalía MULTIVARIADA -- dos columnas movidas cada una por debajo
    del umbral que ya usaría anomalias.py sola, para que lo único raro sea
    la combinación (ver nota de diseño #4: la versión anterior probaba
    detección univariada, que no es lo que este chequeo mide en la
    práctica). Todas las filas de prueba se inyectan de una sola vez y el
    modelo se entrena una única vez -- no una vez por prueba."""
    columnas = _columnas_numericas_utiles(df)
    if len(df) < minimo_filas or len(columnas) < minimo_columnas:
        return None

    df_num = df[columnas].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df_num) < minimo_filas:
        return None
    df_num = df_num.astype(float)

    rng = np.random.default_rng(random_state)
    if len(df_num) > MAX_FILAS_DIAGNOSTICO:
        df_num = df_num.sample(MAX_FILAS_DIAGNOSTICO, random_state=random_state)

    n_pruebas = min(n_pruebas, max(5, len(df_num) // 10))
    filas_elegidas = rng.choice(df_num.index, size=n_pruebas, replace=False)

    medianas = df_num.median()
    mads = (df_num - medianas).abs().median()
    # Qué tan lejos empujar cada columna: justo por debajo del umbral que ya
    # usaría anomalias.py sola, para que ninguna columna sea individualmente
    # "anómala" -- lo único raro debe ser la combinación de ambas.
    magnitud_individual = umbral_z * 0.7

    copia = df_num.copy()
    n_columnas_por_prueba = min(2, len(columnas))
    for fila_idx in filas_elegidas:
        columnas_prueba = rng.choice(columnas, size=n_columnas_por_prueba, replace=False)
        for col in columnas_prueba:
            mad = mads[col]
            escala = (mad / 0.6745) if mad > 0 else (copia[col].std() or abs(medianas[col]) or 1.0)
            signo = 1 if rng.random() > 0.5 else -1
            copia.loc[fila_idx, col] = medianas[col] + signo * escala * magnitud_individual

    modelo = IsolationForest(n_estimators=N_ARBOLES_DIAGNOSTICO, contamination=contaminacion,
                              random_state=random_state)
    etiquetas = modelo.fit_predict(copia)
    posiciones = [copia.index.get_loc(f) for f in filas_elegidas]
    detectados = sum(1 for p in posiciones if etiquetas[p] == -1)

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
        "casi_siempre_cero": np.where(rng.random(n) < 0.9, 0.0, rng.normal(50, 5, n)),
    })
    df.loc[10, "duracion_horas"] = 0.6
    df.loc[10, "costo"] = 950
    df.loc[20, "casi_siempre_cero"] = 300.0  # para probar el caso MAD=0

    print("=== Anomalías multivariadas detectadas ===")
    for a in detectar_anomalias_multivariadas(df):
        print(f" - fila {a.indice}: score={a.score:.3f}, columnas={a.columnas_responsables}")

    print("\n=== Termómetro de confiabilidad ===")
    print(evaluar_confiabilidad(df))


if __name__ == "__main__":
    _demo()