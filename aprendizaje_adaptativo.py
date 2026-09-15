"""
Aprendizaje adaptativo por proyecto: NO es un modelo de Machine Learning
entrenado -- es una línea base (promedio y variabilidad) por columna que
se va afinando sola con cada foto ("snapshot") nueva de los datos del
proyecto. Cada proyecto guarda su propia línea base dentro de su propio
.hadarproy; nunca se comparte entre proyectos ni entre dominios (por
diseño: ver la conversación sobre por qué Hadar es generalista y un
modelo entrenado en un dominio no debería usarse en otro).

Se activa opcionalmente por proyecto (ver HadarApp.ml_activado en
main_window.py) y se actualiza en dos momentos: al abrir el proyecto y
al guardarlo (abrir_proyecto_desde_ruta / _guardar_proyecto).

Por qué "línea base" y no un modelo de verdad:
  - No necesita datos etiquetados ("esto fue una falla real") -- cosa
    que Hadar nunca tiene, sea el dominio que sea.
  - Es explicable: "el valor está a X desviaciones del promedio que
    este proyecto viene mostrando" se puede decir en una frase.
  - No crece sin límite: se guarda solo el estado acumulado (media,
    desviación, cuántas fotos van), no el historial completo de fotos.
"""
import pandas as pd

# Cuánto peso le da cada actualización a la foto más reciente frente a
# lo que ya se sabía. Más alto = se adapta más rápido a cambios (pero
# es más sensible a un mal dato puntual); más bajo = más estable pero
# tarda más en reflejar un cambio real y sostenido.
PESO_FOTO_NUEVA = 0.3


def actualizar_linea_base(linea_base, tablas):
    """Toma una foto de 'tablas' (el estado actual del proyecto) y
    devuelve la línea base actualizada -- no modifica 'linea_base' en el
    lugar, devuelve una copia nueva.

    linea_base: dict {nombre_tabla: {nombre_columna: {"media": float,
        "desviacion": float, "n": int}}}. Puede venir vacío ({}) la
        primera vez que se activa ML en un proyecto.
    tablas: dict {nombre_tabla: DataFrame} -- todas las tablas del
        proyecto, tal como están en pantalla ahora mismo.
    """
    nueva = {tabla: dict(columnas) for tabla, columnas in linea_base.items()}

    for nombre_tabla, df in tablas.items():
        columnas_tabla = dict(nueva.get(nombre_tabla, {}))

        for columna in df.columns:
            if not pd.api.types.is_numeric_dtype(df[columna]):
                continue
            valor_foto = df[columna].mean(skipna=True)
            if pd.isna(valor_foto):
                continue
            valor_foto = float(valor_foto)

            estado = columnas_tabla.get(str(columna))
            if not estado or estado.get("n", 0) == 0:
                columnas_tabla[str(columna)] = {"media": valor_foto, "desviacion": 0.0, "n": 1}
                continue

            media_anterior = estado["media"]
            varianza_anterior = estado["desviacion"] ** 2
            media_nueva = PESO_FOTO_NUEVA * valor_foto + (1 - PESO_FOTO_NUEVA) * media_anterior
            varianza_nueva = (
                PESO_FOTO_NUEVA * (valor_foto - media_nueva) ** 2
                + (1 - PESO_FOTO_NUEVA) * varianza_anterior
            )
            columnas_tabla[str(columna)] = {
                "media": media_nueva,
                "desviacion": varianza_nueva ** 0.5,
                "n": estado["n"] + 1,
            }

        nueva[nombre_tabla] = columnas_tabla

    return nueva


def es_anomalo_segun_linea_base(linea_base, nombre_tabla, columna, valor,
                                 umbral_desviaciones=3.0, minimo_fotos=5):
    """True si 'valor' se sale del rango normal aprendido para esa
    columna de ESE proyecto. Devuelve False (no marca nada) si todavía
    no hay suficientes fotos acumuladas como para confiar en la línea
    base -- un proyecto recién creado con ML activado no debe empezar a
    marcar anomalías con una sola foto de historial."""
    estado = linea_base.get(nombre_tabla, {}).get(str(columna))
    if not estado or estado["n"] < minimo_fotos or estado["desviacion"] == 0:
        return False
    return abs(valor - estado["media"]) > umbral_desviaciones * estado["desviacion"]
