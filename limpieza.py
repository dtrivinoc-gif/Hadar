"""
limpieza.py — Detección de "suciedad" técnica en una tabla, para Hadar.

Esto es deliberadamente DISTINTO de anomalias.py: una anomalía (un precio
mucho más alto que el resto) puede ser un problema real de negocio que
hay que investigar, no algo para borrar. Lo que detecta este módulo es
otra cosa -- ruido puramente técnico, donde la corrección casi siempre es
obvia una vez que se ve:

  - Filas duplicadas exactas (la misma fila cargada dos veces).
  - Espacios en blanco sueltos al inicio o final de un texto ("Juan " en
    vez de "Juan" -- técnicamente distinto para el software aunque se vea
    igual).
  - Variantes de formato de la misma categoría ("Chile", "chile ",
    "CHILE" tratadas como 3 valores distintos cuando son el mismo).

A propósito este módulo NUNCA modifica el DataFrame -- solo señala,
fila por fila y celda por celda, qué encontró y por qué. Aplicar el
cambio (o no) queda siempre en manos de la persona, nunca automático.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .ontologia import _similitud_semantica, UMBRAL_SIMILITUD_SEMANTICA


MAX_VARIANTES_POR_GRUPO = 6   # tope de variantes distintas a listar en la descripción de un grupo


@dataclass
class HallazgoLimpieza:
    tipo: str                       # "duplicado" | "espacio_en_blanco" | "formato_inconsistente"
    filas: list                      # row_labels afectados (para duplicado, TODO el grupo)
    columna: str | None              # None para duplicados (afecta la fila completa)
    descripcion: str                  # texto listo para mostrar en la lista de la sidebar
    detalle: dict = field(default_factory=dict)  # info extra (ej. valor original vs. sugerido)


def detectar_duplicados_exactos(df: pd.DataFrame) -> list[HallazgoLimpieza]:
    """Grupos de filas que son idénticas en TODAS sus columnas."""
    if df.empty:
        return []
    mascara = df.duplicated(keep=False)
    if not mascara.any():
        return []

    hallazgos = []
    # Agrupa por el contenido completo de la fila (como texto, para que
    # tipos mixtos no rompan la comparación) y arma un hallazgo por grupo.
    subset = df[mascara]
    claves = subset.astype(str).agg("|".join, axis=1)
    for _, grupo in subset.groupby(claves).groups.items():
        filas = list(grupo)
        if len(filas) < 2:
            continue
        hallazgos.append(HallazgoLimpieza(
            tipo="duplicado", filas=filas, columna=None,
            descripcion=f"{len(filas)} filas idénticas entre sí (filas {_listar_filas(filas)})",
        ))
    return hallazgos


def detectar_espacios_en_blanco(df: pd.DataFrame) -> list[HallazgoLimpieza]:
    """Celdas de texto con espacios sueltos al inicio o al final."""
    hallazgos = []
    columnas_texto = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    for col in columnas_texto:
        serie = df[col]
        for row_label, valor in serie.items():
            if not isinstance(valor, str):
                continue
            limpio = valor.strip()
            if limpio != valor and limpio != "":
                hallazgos.append(HallazgoLimpieza(
                    tipo="espacio_en_blanco", filas=[row_label], columna=col,
                    descripcion=f"'{col}' tiene espacios de más: {valor!r} → {limpio!r}",
                    detalle={"valor_original": valor, "valor_sugerido": limpio},
                ))
    return hallazgos


def _normalizar_para_comparar(valor: str) -> str:
    return " ".join(valor.strip().lower().split())


def detectar_formato_inconsistente(df: pd.DataFrame) -> list[HallazgoLimpieza]:
    """
    Valores de texto que representan lo mismo pero con mayúsculas/espacios
    distintos (ej. 'Chile' y 'chile '), dentro de la misma columna. Se
    propone como forma "canónica" la variante que aparece más seguido; el
    resto queda como hallazgo. Si todos las variantes de un grupo aparecen
    la misma cantidad de veces, se usa la primera en orden alfabético
    (desempate simple y predecible, sin azar).
    """
    hallazgos = []
    columnas_texto = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
    for col in columnas_texto:
        serie = df[col].dropna()
        serie = serie[serie.apply(lambda v: isinstance(v, str) and v.strip() != "")]
        if serie.empty:
            continue

        grupos: dict[str, dict[str, list]] = {}
        for row_label, valor in serie.items():
            clave = _normalizar_para_comparar(valor)
            grupos.setdefault(clave, {}).setdefault(valor, []).append(row_label)

        for clave, variantes in grupos.items():
            if len(variantes) < 2:
                continue  # una sola forma de escribirlo: no hay nada que reportar

            def _puntaje_desempate(valor: str):
                # 0 = tiene mayúsculas y minúsculas mezcladas (ej. "Chile", lo más
                # habitual como forma "bien escrita"); 1 = todo mayúsculas o todo
                # minúsculas. Menor puntaje = mejor candidato a forma canónica.
                es_pura = valor == valor.upper() or valor == valor.lower()
                return (1 if es_pura else 0, valor)

            canonica = sorted(
                variantes.items(), key=lambda kv: (-len(kv[1]), _puntaje_desempate(kv[0]))
            )[0][0]

            variantes_listadas = list(variantes.keys())[:MAX_VARIANTES_POR_GRUPO]
            texto_variantes = ", ".join(f"{v!r}" for v in variantes_listadas)

            for valor, filas in variantes.items():
                if valor == canonica:
                    continue
                hallazgos.append(HallazgoLimpieza(
                    tipo="formato_inconsistente", filas=filas, columna=col,
                    descripcion=(
                        f"'{col}' tiene {len(variantes)} formas de escribir lo mismo "
                        f"({texto_variantes}) — se sugiere unificar en {canonica!r}"
                    ),
                    detalle={"valor_original": valor, "valor_sugerido": canonica},
                ))
    return hallazgos


CARACTERES_PERMITIDOS_DEFECTO = set(" .,;:()-_'\"/@%+°ºª")


def detectar_caracteres_especiales(
    df: pd.DataFrame, caracteres_permitidos: set | None = None
) -> list[HallazgoLimpieza]:
    """
    Detecta símbolos "raros" en columnas de texto -- cualquier carácter
    que no sea letra (incluye acentos y ñ, vía str.isalnum()), número,
    espacio, o puntuación común de uso normal (definida en
    CARACTERES_PERMITIDOS_DEFECTO, que a propósito incluye '@' para no
    marcar los correos como sucios).

    Se agrupa por (columna, carácter exacto) -- no celda por celda -- para
    que la persona decida por separado si CADA carácter tiene sentido en
    esa columna (ej. un '#' suelto en una columna de nombres probablemente
    sea basura, pero en una columna de códigos de producto puede ser
    parte del formato real).
    """
    permitidos = caracteres_permitidos if caracteres_permitidos is not None else CARACTERES_PERMITIDOS_DEFECTO
    columnas_texto = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]

    por_clave: dict[tuple, dict] = {}
    for col in columnas_texto:
        for row_label, valor in df[col].items():
            if not isinstance(valor, str):
                continue
            for ch in valor:
                if ch.isalnum() or ch.isspace() or ch in permitidos:
                    continue
                entrada = por_clave.setdefault((col, ch), {"filas": [], "ejemplos": []})
                entrada["filas"].append(row_label)
                if len(entrada["ejemplos"]) < 3 and valor not in entrada["ejemplos"]:
                    entrada["ejemplos"].append(valor)

    hallazgos = []
    for (col, ch), datos in por_clave.items():
        ejemplos_txt = ", ".join(f"{e!r}" for e in datos["ejemplos"])
        hallazgos.append(HallazgoLimpieza(
            tipo="caracter_especial", filas=datos["filas"], columna=col,
            descripcion=(
                f"'{col}' tiene el carácter {ch!r} en {len(datos['filas'])} celda(s) "
                f"(ej: {ejemplos_txt})"
            ),
            detalle={"caracter": ch, "ejemplos": datos["ejemplos"]},
        ))
    hallazgos.sort(key=lambda h: len(h.filas), reverse=True)
    return hallazgos


PALABRAS_NUMERICAS_ES = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintidós": 22, "veintitres": 23,
    "veintitrés": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintiséis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
}


def _palabra_a_numero(texto: str) -> float | None:
    """Convierte una palabra numérica en español a número. Cubre 0-100 y
    compuestos simples de dos palabras ('treinta y cinco' = 35) -- no
    cubre miles ni millones, eso queda fuera de alcance a propósito para
    no meterse en ambigüedades de redacción más complejas."""
    texto = texto.strip().lower()
    if texto in PALABRAS_NUMERICAS_ES:
        return float(PALABRAS_NUMERICAS_ES[texto])
    if " y " in texto:
        izquierda, _, derecha = texto.partition(" y ")
        izquierda, derecha = izquierda.strip(), derecha.strip()
        if izquierda in PALABRAS_NUMERICAS_ES and derecha in PALABRAS_NUMERICAS_ES:
            base, resto = PALABRAS_NUMERICAS_ES[izquierda], PALABRAS_NUMERICAS_ES[derecha]
            if base >= 30 and base % 10 == 0 and resto < 10:
                return float(base + resto)
    return None


def _formatear_numero(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def _intentar_convertir_a_numero(valor) -> float | None:
    """
    Intenta leer un valor de celda como número, en este orden:
    1. Ya es un número escrito tal cual ('5', '5.5', '5,5').
    2. Es una palabra numérica en español ('cuatro').
    3. Es un número con texto pegado o después, tipo unidad ('6 unidades', '7kg').
    4. Es una palabra numérica con texto pegado ('nueve unidades').
    Devuelve None si no se pudo interpretar de ninguna forma -- en ese
    caso la celda se deja intacta, no se fuerza nada.
    """
    if not isinstance(valor, str):
        return None
    texto = valor.strip()
    if texto == "":
        return None

    try:
        return float(texto.replace(",", "."))
    except ValueError:
        pass

    palabra_completa = _palabra_a_numero(texto.lower())
    if palabra_completa is not None:
        return palabra_completa

    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*[a-záéíóúñ]+\.?$", texto, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass

    primera_palabra = texto.split()[0].lower() if texto.split() else ""
    if len(texto.split()) > 1:
        valor_palabra = _palabra_a_numero(primera_palabra)
        if valor_palabra is not None:
            return valor_palabra

    return None


UMBRAL_FRACCION_CONVERTIBLE = 0.9  # al menos 90% de la columna debe poder leerse como número


def detectar_numeros_como_texto(df: pd.DataFrame) -> list[HallazgoLimpieza]:
    """
    Busca columnas que DEBERÍAN ser numéricas pero quedaron como texto
    porque se mezclaron formas distintas de escribir números (dígitos,
    palabras, dígitos con una unidad pegada). Solo se considera una
    columna candidata si la GRAN MAYORÍA de sus valores se puede
    interpretar como número -- si no, es más probable que sea una columna
    de texto genuina con algún número suelto, no una columna numérica sucia.
    """
    hallazgos = []
    columnas_candidatas = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]

    for col in columnas_candidatas:
        serie = df[col].dropna()
        serie = serie[serie.apply(lambda v: isinstance(v, str) and v.strip() != "")]
        if len(serie) < 3:
            continue

        conversiones = serie.apply(_intentar_convertir_a_numero)
        fraccion_convertible = conversiones.notna().mean()
        if fraccion_convertible < UMBRAL_FRACCION_CONVERTIBLE:
            continue

        for (fila, valor), (_, convertido) in zip(serie.items(), conversiones.items()):
            if convertido is None or (isinstance(convertido, float) and pd.isna(convertido)):
                continue
            if valor.strip() == _formatear_numero(convertido):
                continue  # ya estaba limpio, nada que reportar en esta celda
            hallazgos.append(HallazgoLimpieza(
                tipo="numero_en_texto", filas=[fila], columna=col,
                descripcion=f"'{col}' tiene un número escrito como texto: {valor!r} → {_formatear_numero(convertido)}",
                detalle={"valor_original": valor, "valor_sugerido": convertido},
            ))

    return hallazgos


def aplicar_numeros_como_texto(df: pd.DataFrame, hallazgos: list[HallazgoLimpieza]) -> pd.DataFrame:
    """
    Reemplaza cada celda marcada por su valor numérico. Al final, para
    cada columna tocada, intenta convertirla a tipo numérico real -- si
    alguna celda de esa columna sigue sin poder interpretarse como número
    (una que no se marcó porque no se entendía), la columna queda como
    texto, con las celdas que SÍ se pudieron corregir ya en su forma
    numérica limpia -- nunca se fuerza un tipo de dato que rompería algo.
    """
    df2 = df.copy()
    columnas_tocadas = set()
    for h in hallazgos:
        if h.tipo != "numero_en_texto":
            continue
        if h.columna not in columnas_tocadas:
            df2[h.columna] = df2[h.columna].astype(object)  # para poder mezclar texto y número antes de convertir del todo
            columnas_tocadas.add(h.columna)
        for fila in h.filas:
            df2.at[fila, h.columna] = h.detalle["valor_sugerido"]

    for col in columnas_tocadas:
        try:
            df2[col] = pd.to_numeric(df2[col])
        except (ValueError, TypeError):
            pass
    return df2


def _listar_filas(filas: list, tope: int = 6) -> str:
    etiquetas = [str(f) for f in filas[:tope]]
    if len(filas) > tope:
        etiquetas.append(f"+{len(filas) - tope} más")
    return ", ".join(etiquetas)


def detectar_limpieza_sugerida(df: pd.DataFrame) -> list[HallazgoLimpieza]:
    """
    Punto de entrada único: junta las cinco detecciones, en un orden fijo
    (duplicados primero, por ser lo más urgente de resolver -- infla
    conteos y promedios directamente).
    """
    hallazgos = []
    hallazgos.extend(detectar_duplicados_exactos(df))
    hallazgos.extend(detectar_espacios_en_blanco(df))
    hallazgos.extend(detectar_formato_inconsistente(df))
    hallazgos.extend(detectar_caracteres_especiales(df))
    hallazgos.extend(detectar_numeros_como_texto(df))
    return hallazgos


def agrupar_por_tipo(hallazgos: list[HallazgoLimpieza]) -> dict[str, list[HallazgoLimpieza]]:
    """Agrupa los hallazgos por tipo, en el mismo orden fijo de siempre --
    para armar los botones de 'Aplicar por categoría' en la interfaz."""
    orden = [
        "duplicado", "espacio_en_blanco", "formato_inconsistente",
        "caracter_especial", "numero_en_texto",
    ]
    grupos: dict[str, list[HallazgoLimpieza]] = {t: [] for t in orden}
    for h in hallazgos:
        grupos.setdefault(h.tipo, []).append(h)
    return {t: hs for t, hs in grupos.items() if hs}


# ----------------------------------------------------------------------
# Funciones que APLICAN una corrección -- todas devuelven una COPIA nueva
# del DataFrame, nunca modifican el original. Aplicar (o no) y CUÁNDO
# aplicar es siempre una decisión explícita de quien usa Hadar, nunca
# algo que pase solo.
# ----------------------------------------------------------------------

def aplicar_eliminar_duplicados(df: pd.DataFrame, hallazgos: list[HallazgoLimpieza]) -> pd.DataFrame:
    """Conserva la PRIMERA fila de cada grupo de duplicados exactos y
    elimina el resto. Si dos hallazgos de duplicado comparten alguna fila
    (no debería pasar, pero por si acaso) no se elimina dos veces."""
    filas_a_eliminar = set()
    for h in hallazgos:
        if h.tipo != "duplicado":
            continue
        filas_a_eliminar.update(h.filas[1:])  # conserva la primera de cada grupo
    if not filas_a_eliminar:
        return df.copy()
    return df.drop(index=list(filas_a_eliminar))


def aplicar_espacios_en_blanco(df: pd.DataFrame, hallazgos: list[HallazgoLimpieza]) -> pd.DataFrame:
    """Recorta los espacios de más SOLO en las celdas marcadas -- no en
    toda la columna a ciegas, para no tocar nada que no se haya revisado."""
    df2 = df.copy()
    for h in hallazgos:
        if h.tipo != "espacio_en_blanco":
            continue
        for fila in h.filas:
            df2.at[fila, h.columna] = h.detalle["valor_sugerido"]
    return df2


def aplicar_formato_inconsistente(df: pd.DataFrame, hallazgos: list[HallazgoLimpieza]) -> pd.DataFrame:
    """Reemplaza cada variante por la forma canónica sugerida (la más
    frecuente, con desempate hacia el formato tipo 'Título')."""
    df2 = df.copy()
    for h in hallazgos:
        if h.tipo != "formato_inconsistente":
            continue
        for fila in h.filas:
            df2.at[fila, h.columna] = h.detalle["valor_sugerido"]
    return df2


def aplicar_quitar_caracteres(
    df: pd.DataFrame, hallazgos: list[HallazgoLimpieza], claves_a_aplicar: set[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """
    Quita caracteres especiales de las columnas marcadas. claves_a_aplicar
    es un set de (columna, carácter) -- si se pasa, SOLO se aplican esas
    combinaciones puntuales (para cuando la persona revisó la lista y
    decidió, por ejemplo, sacar '#' pero dejar '@'); si se deja en None,
    se aplican TODOS los hallazgos de tipo caracter_especial que se
    pasaron en `hallazgos`.
    """
    df2 = df.copy()
    for h in hallazgos:
        if h.tipo != "caracter_especial":
            continue
        clave = (h.columna, h.detalle["caracter"])
        if claves_a_aplicar is not None and clave not in claves_a_aplicar:
            continue
        caracter = h.detalle["caracter"]
        col = h.columna
        df2[col] = df2[col].apply(lambda v, c=caracter: v.replace(c, "") if isinstance(v, str) else v)
    return df2


# ----------------------------------------------------------------------
# Formato inconsistente ENTRE tablas relacionadas
# ----------------------------------------------------------------------
# A diferencia de detectar_formato_inconsistente() (que mira una sola
# tabla), esto mira columnas de TABLAS DISTINTAS que representan el mismo
# concepto -- ej. 'pais' en Clientes y 'pais_origen' en Proveedores.
# Reutiliza el mismo comparador semántico de nombres de columna que ya
# usa ontologia.py para decidir qué columnas se relacionan entre tablas
# (el que evita comparar 'categoria_id' con 'cliente_id' solo porque
# ambas terminan en '_id') -- acá decide qué columnas de texto de tablas
# distintas vale la pena mirar juntas antes de compararlas.

MAX_VALORES_UNICOS_CATEGORIA = 200  # sobre esto, probablemente es texto libre o un identificador, no una categoría


def _es_columna_categorica(serie: pd.Series) -> bool:
    """
    Filtro de sensatez: no tiene sentido juntar una columna de texto libre
    (comentarios, descripciones) o un identificador (que casi nunca se
    repite) como si fuera una categoría compartida entre tablas -- eso
    generaría comparaciones sin sentido, no hallazgos útiles.
    """
    valores = serie.dropna()
    valores = valores[valores.apply(lambda v: isinstance(v, str) and v.strip() != "")]
    if valores.empty:
        return False
    distintos = valores.map(_normalizar_para_comparar).nunique()
    if distintos > MAX_VALORES_UNICOS_CATEGORIA:
        return False
    # Con muestras muy chicas, la proporción de valores únicos es ruidosa
    # (2 filas con 2 países distintos ya da 100% "único" sin que eso
    # signifique que sea un identificador) -- este filtro solo tiene
    # sentido con una base mínima de datos.
    if len(valores) >= 10 and distintos / len(valores) > 0.8:
        return False  # casi todos los valores son distintos entre sí: más identificador que categoría
    return True


@dataclass
class VarianteCrossTabla:
    tabla: str
    columna: str
    fila: object
    valor_original: str


@dataclass
class GrupoFormatoCrossTabla:
    columnas: list                       # [(tabla, columna), ...] involucradas
    forma_canonica: str
    variantes: list                       # VarianteCrossTabla, una por celda a corregir
    descripcion: str


def detectar_formato_inconsistente_entre_tablas(
    tablas: dict[str, pd.DataFrame],
) -> list[GrupoFormatoCrossTabla]:
    """
    Para cada grupo de columnas de tablas DISTINTAS que el comparador
    semántico considera "la misma entidad", junta todos sus valores y
    busca variantes de formato -- igual idea que
    detectar_formato_inconsistente(), pero cruzando tablas.
    """
    if len(tablas) < 2:
        return []

    candidatas = []  # [(tabla, columna)]
    for nombre_tabla, df in tablas.items():
        columnas_texto = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        for col in columnas_texto:
            if _es_columna_categorica(df[col]):
                candidatas.append((nombre_tabla, col))

    # Unión-find simple: agrupa columnas semánticamente iguales de tablas distintas.
    padre = {c: c for c in candidatas}

    def encontrar(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a, b):
        ra, rb = encontrar(a), encontrar(b)
        if ra != rb:
            padre[ra] = rb

    for i in range(len(candidatas)):
        for j in range(i + 1, len(candidatas)):
            tabla1, col1 = candidatas[i]
            tabla2, col2 = candidatas[j]
            if tabla1 == tabla2:
                continue  # eso ya lo cubre detectar_formato_inconsistente()
            sim = _similitud_semantica(col1, col2)
            if sim is not None and sim >= UMBRAL_SIMILITUD_SEMANTICA:
                unir(candidatas[i], candidatas[j])

    grupos_por_raiz: dict = {}
    for c in candidatas:
        grupos_por_raiz.setdefault(encontrar(c), []).append(c)

    resultado = []
    for miembros in grupos_por_raiz.values():
        tablas_involucradas = {t for t, _ in miembros}
        if len(tablas_involucradas) < 2:
            continue  # el grupo entero cae en una sola tabla: no es "entre tablas"

        por_valor_normalizado: dict[str, dict] = {}
        for tabla, columna in miembros:
            for fila, valor in tablas[tabla][columna].items():
                if not isinstance(valor, str) or valor.strip() == "":
                    continue
                clave = _normalizar_para_comparar(valor)
                entrada = por_valor_normalizado.setdefault(clave, {})
                entrada.setdefault(valor, []).append((tabla, columna, fila))

        for variantes_dict in por_valor_normalizado.values():
            if len(variantes_dict) < 2:
                continue

            def _puntaje_desempate(valor):
                es_pura = valor == valor.upper() or valor == valor.lower()
                return (1 if es_pura else 0, valor)

            canonica = sorted(
                variantes_dict.items(), key=lambda kv: (-len(kv[1]), _puntaje_desempate(kv[0]))
            )[0][0]

            variantes_finales = [
                VarianteCrossTabla(tabla=t, columna=c, fila=f, valor_original=valor)
                for valor, ubicaciones in variantes_dict.items() if valor != canonica
                for (t, c, f) in ubicaciones
            ]
            if not variantes_finales:
                continue

            nombres_columnas = ", ".join(f"{t}.{c}" for t, c in sorted(miembros))
            variantes_listadas = list(variantes_dict.keys())[:MAX_VARIANTES_POR_GRUPO]
            texto_variantes = ", ".join(f"{v!r}" for v in variantes_listadas)

            resultado.append(GrupoFormatoCrossTabla(
                columnas=sorted(miembros), forma_canonica=canonica, variantes=variantes_finales,
                descripcion=(
                    f"{nombres_columnas} representan lo mismo y tienen {len(variantes_dict)} "
                    f"formas de escribirlo ({texto_variantes}) — se sugiere unificar en "
                    f"{canonica!r} ({len(variantes_finales)} celda(s) a corregir)"
                ),
            ))

    resultado.sort(key=lambda g: len(g.variantes), reverse=True)
    return resultado


def aplicar_formato_cross_tabla(
    tablas: dict[str, pd.DataFrame],
    grupos: list[GrupoFormatoCrossTabla],
    grupos_a_aplicar: set | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Devuelve un diccionario NUEVO {tabla: DataFrame}: las tablas no
    afectadas se devuelven tal cual, las afectadas como copia modificada.
    grupos_a_aplicar es un set de índices (posición en `grupos`) a
    aplicar; si es None, se aplican TODOS los grupos pasados.
    """
    tablas_nuevas = dict(tablas)
    copiadas: set[str] = set()

    for i, grupo in enumerate(grupos):
        if grupos_a_aplicar is not None and i not in grupos_a_aplicar:
            continue
        for variante in grupo.variantes:
            if variante.tabla not in copiadas:
                tablas_nuevas[variante.tabla] = tablas[variante.tabla].copy()
                copiadas.add(variante.tabla)
            tablas_nuevas[variante.tabla].at[variante.fila, variante.columna] = grupo.forma_canonica

    return tablas_nuevas


# ----------------------------------------------------------------------
# Demo / auto-test rápido
# ----------------------------------------------------------------------

def _demo():
    df = pd.DataFrame({
        "cliente_id": [1, 2, 3, 3, 4],
        "pais": ["Chile", "chile ", "Perú", "Perú", "CHILE"],
        "nombre": ["Ana", "Beto ", "Carla", "Carla", " Diego#"],
        "monto": [100, 200, 300, 300, 400],
    })
    print("=== Hallazgos de limpieza detectados ===")
    hallazgos = detectar_limpieza_sugerida(df)
    for h in hallazgos:
        print(f" - [{h.tipo}] {h.descripcion}")

    print("\n=== Agrupados por tipo (para los botones de 'Aplicar') ===")
    for tipo, hs in agrupar_por_tipo(hallazgos).items():
        print(f" - {tipo}: {len(hs)} hallazgo(s)")

    print("\n=== Aplicando TODAS las correcciones, una por una ===")
    df_limpio = df.copy()
    df_limpio = aplicar_eliminar_duplicados(df_limpio, hallazgos)
    df_limpio = aplicar_espacios_en_blanco(df_limpio, hallazgos)
    df_limpio = aplicar_formato_inconsistente(df_limpio, hallazgos)
    df_limpio = aplicar_quitar_caracteres(df_limpio, hallazgos)
    print(df_limpio)

    print("\n=== Formato inconsistente ENTRE tablas ===")
    clientes = pd.DataFrame({"cliente_id": [1, 2, 3], "pais": ["Chile", "chile ", "Perú"]})
    proveedores = pd.DataFrame({"proveedor_id": [10, 20], "pais_origen": ["CHILE", "Perú "]})
    tablas = {"Clientes": clientes, "Proveedores": proveedores}
    grupos = detectar_formato_inconsistente_entre_tablas(tablas)
    for g in grupos:
        print(" -", g.descripcion)
    tablas_limpias = aplicar_formato_cross_tabla(tablas, grupos)
    print(tablas_limpias["Clientes"])
    print(tablas_limpias["Proveedores"])


if __name__ == "__main__":
    _demo()