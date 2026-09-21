"""
linea_tiempo.py — Lógica pura (sin PySide6) para la pestaña "Línea de
Tiempo": construcción de series de hitos, pares inicio/fin para las
conexiones entre puntos, histograma de densidad para el filtro global,
el cruce entre anomalías detectadas (anomalias.py) y su fecha real, y
el formateo de duraciones en lenguaje natural.

Filosofía generalista: NADA de esto asume un dominio. Toda columna de
fecha se descubre con detectar_columnas_fecha() (anomalias.py) y toda
relación inicio/fin viene de:
  1. pares por token (TOKENS_TEMPORALES_PARES) -- heurística de nombre
     de columna, sirve igual para venta ("pedido"->"entrega") que para
     mantenimiento preventivo ("inicio"->"fin") o cualquier otro dominio
     con procesos de dos tiempos.
  2. pares_temporales_personalizados definidos a mano por el usuario
     para ese esquema (memoria.py), que siempre tienen prioridad porque
     representan una relación que la persona confirmó explícitamente.
Nunca se asume que existe una columna con un nombre en particular.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomalias import TOKENS_TEMPORALES_PARES


# ----------------------------------------------------------------------
# Descubrimiento de pares inicio/fin (para las conexiones entre puntos)
# ----------------------------------------------------------------------

@dataclass
class ParTemporal:
    col_inicio: str
    col_fin: str
    etiqueta: str
    origen: str  # "automatico" | "manual"


def _sin_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def sugerir_pares_temporales(columnas_fecha, pares_personalizados=None):
    """Junta los pares manuales guardados en memoria.py (siempre primero,
    porque el usuario ya los confirmó) con los pares automáticos por
    token de TOKENS_TEMPORALES_PARES. No repite un mismo par dos veces
    aunque coincida por ambas vías."""
    pares: list[ParTemporal] = []
    vistos = set()

    for p in (pares_personalizados or []):
        clave = (p["col_anterior"], p["col_posterior"])
        if clave in vistos:
            continue
        vistos.add(clave)
        pares.append(ParTemporal(
            col_inicio=p["col_anterior"],
            col_fin=p["col_posterior"],
            etiqueta=p.get("etiqueta") or f"{p['col_anterior']} → {p['col_posterior']}",
            origen="manual",
        ))

    columnas_norm = {c: _sin_acentos(c) for c in columnas_fecha}
    for token_a, token_b, etiqueta in TOKENS_TEMPORALES_PARES:
        col_a = next((c for c, n in columnas_norm.items() if token_a in n), None)
        col_b = next((c for c, n in columnas_norm.items() if token_b in n), None)
        if col_a and col_b and col_a != col_b:
            clave = (col_a, col_b)
            if clave in vistos:
                continue
            vistos.add(clave)
            pares.append(ParTemporal(col_inicio=col_a, col_fin=col_b, etiqueta=etiqueta, origen="automatico"))

    return pares


# ----------------------------------------------------------------------
# Formato de fecha (orden día/mes/año) -- detección automática
# ----------------------------------------------------------------------

# Año primero: 2026-04-27, 2026/04/27, 2026.04.27 (con o sin hora después).
_RE_ANIO_PRIMERO = re.compile(r"^\s*(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})(?!\d)")
# Año al final: 27/04/2026, 4-27-26, 27.04.2026 (con o sin hora después).
_RE_ANIO_AL_FINAL = re.compile(r"^\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4}|\d{2})(?!\d)")


@dataclass
class FormatoFecha:
    orden: str | None   # "ymd" | "dmy" | "mdy" | None (no se pudo decidir)
    dayfirst: bool      # lo que hay que pasarle a pandas.to_datetime
    seguro: bool        # False = se asumió DD/MM por defecto, sin evidencia
    descripcion: str    # texto corto para mostrar en pantalla ("" = no aplica)


def detectar_formato_fecha(serie: pd.Series, max_muestras: int = 2000) -> FormatoFecha:
    """Decide cómo leer una columna de fechas escritas como texto, mirando
    los datos en vez de preguntarle al usuario.

    Reglas (todas verificables a ojo, nada de heurística opaca):
      - Si empieza con 4 dígitos (2026-04-27) es AAAA/MM/DD: en ese orden
        el mes siempre va antes que el día, así que se lee con
        dayfirst=False. IMPORTANTE: con dayfirst=True pandas leería
        2026-04-05 como el 4 de mayo, no el 5 de abril.
      - Si termina en año (27/04/2026): un primer número mayor a 12 solo
        puede ser un día (=> DD/MM); un segundo número mayor a 12 solo
        puede ser un día (=> MM/DD).
      - Si todos los valores caben en ambos órdenes (nada mayor a 12) es
        genuinamente ambiguo: se deja el DD/MM de siempre y se marca
        seguro=False para que la pantalla lo diga en vez de fingir certeza.

    Determinista: la muestra usa random_state fijo, así que el mismo
    archivo siempre da el mismo resultado."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return FormatoFecha(orden=None, dayfirst=True, seguro=True, descripcion="")

    textos = serie.dropna().astype(str)
    if textos.empty:
        return FormatoFecha(orden=None, dayfirst=True, seguro=False, descripcion="")
    if len(textos) > max_muestras:
        textos = textos.sample(max_muestras, random_state=0)

    anio_primero = textos.str.extract(_RE_ANIO_PRIMERO).dropna()
    anio_final = textos.str.extract(_RE_ANIO_AL_FINAL).dropna()
    n_ymd, n_dm = len(anio_primero), len(anio_final)

    if n_ymd == 0 and n_dm == 0:
        return FormatoFecha(orden=None, dayfirst=True, seguro=False, descripcion="")

    if n_ymd >= n_dm:
        mezcla = n_dm > 0
        return FormatoFecha(
            orden="ymd", dayfirst=False, seguro=not mezcla,
            descripcion="AAAA/MM/DD, con algunas filas en otro orden" if mezcla else "AAAA/MM/DD",
        )

    primero = anio_final[0].astype(int)
    segundo = anio_final[1].astype(int)
    primero_es_dia = bool((primero > 12).any())
    segundo_es_dia = bool((segundo > 12).any())

    if primero_es_dia and segundo_es_dia:
        return FormatoFecha(orden=None, dayfirst=True, seguro=False,
                            descripcion="formatos mezclados, se asumió DD/MM/AAAA")
    if primero_es_dia:
        return FormatoFecha(orden="dmy", dayfirst=True, seguro=True, descripcion="DD/MM/AAAA")
    if segundo_es_dia:
        return FormatoFecha(orden="mdy", dayfirst=False, seguro=True, descripcion="MM/DD/AAAA")
    return FormatoFecha(orden=None, dayfirst=True, seguro=False,
                        descripcion="DD/MM/AAAA asumido (ninguna fecha lo confirma)")


def parsear_fechas(serie: pd.Series, dayfirst: bool = True) -> pd.Series:
    """ÚNICO punto donde Hadar convierte texto a fecha en la Línea de
    Tiempo y en el filtro por rango de main_window -- así el gráfico y los
    filtros nunca leen la misma columna de dos formas distintas."""
    return pd.to_datetime(serie, errors="coerce", format="mixed", dayfirst=dayfirst)


# ----------------------------------------------------------------------
# Modo Hito
# ----------------------------------------------------------------------

def construir_serie_hitos(df: pd.DataFrame, columna_fecha: str, dayfirst: bool = True) -> pd.Series:
    """Timestamps válidos de una columna de fecha para el Modo Hito,
    conservando el índice original del DataFrame (para poder volver a
    la fila exacta al hacer clic en un punto).

    dayfirst decide cómo se lee una fecha ambigua tipo "03/04/2024":
    True = 3 de abril (Chile y la mayoría de países), False = 4 de marzo
    (Estados Unidos) y también AAAA/MM/DD. Quien llama normalmente no lo
    elige a mano: la UI lo saca de detectar_formato_fecha(). El default
    True se mantiene solo porque io_datos.cast_valor_a_dtype ya asume lo
    mismo al escribir celdas a mano."""
    fechas = parsear_fechas(df[columna_fecha], dayfirst=dayfirst)
    return fechas.dropna()


def tiene_componente_horario(fechas: pd.Series) -> bool:
    """True si la columna de fecha trae hora real (no todos los valores
    caen justo en medianoche). Un archivo exportado solo con fecha
    (sin hora) parsea todo a las 00:00:00 -- en ese caso un eje "hora
    del día" mostraría todo pegado en cero y mentiría, así que hay que
    detectarlo antes de usarlo, no asumirlo."""
    if fechas.empty:
        return False
    return not (
        (fechas.dt.hour == 0).all()
        and (fechas.dt.minute == 0).all()
        and (fechas.dt.second == 0).all()
    )


def minutos_desde_medianoche(fechas: pd.Series) -> pd.Series:
    """Minutos transcurridos desde medianoche (0-1439.99) para cada
    timestamp, preservando el índice original -- usado como eje Y del
    Modo Hito cuando tiene_componente_horario() da True."""
    return fechas.dt.hour * 60 + fechas.dt.minute + fechas.dt.second / 60


def construir_series_hitos_multiples(df: pd.DataFrame, columnas, dayfirst: bool = True) -> dict:
    """Como construir_serie_hitos pero para varias columnas de fecha a la
    vez -- para cuando un dataset tiene más de una fecha relevante (ej.
    fecha_envio/fecha_entrega, fecha_ultima_mantencion/fecha_siguiente)
    y se quieren ver todas juntas, cada una con su propio color. Devuelve
    {columna: Serie de timestamps válidos}, salteando columnas vacías o
    inexistentes -- ninguna columna es obligatoria."""
    resultado = {}
    for col in columnas:
        if col not in df.columns:
            continue
        serie = construir_serie_hitos(df, col, dayfirst=dayfirst)
        if not serie.empty:
            resultado[col] = serie
    return resultado


# ----------------------------------------------------------------------
# Pares inicio/fin por fila + incoherencias cronológicas
# ----------------------------------------------------------------------

def construir_intervalos(df: pd.DataFrame, par: ParTemporal, dayfirst: bool = True,
                         dayfirst_fin: bool | None = None) -> pd.DataFrame:
    """DataFrame con inicio, fin, duración y marca de incoherencia (fin
    anterior a inicio) para conectar cada fila con una línea. El índice se conserva
    para poder ubicar la fila real de cada barra. Ver construir_serie_hitos
    para qué decide dayfirst. dayfirst_fin permite que la columna de fin
    tenga otro formato que la de inicio (None = el mismo)."""
    inicio = parsear_fechas(df[par.col_inicio], dayfirst=dayfirst)
    fin = parsear_fechas(df[par.col_fin], dayfirst=dayfirst if dayfirst_fin is None else dayfirst_fin)
    validas = inicio.notna() & fin.notna()
    resultado = pd.DataFrame({
        "inicio": inicio[validas],
        "fin": fin[validas],
    }, index=df.index[validas])
    resultado["duracion"] = resultado["fin"] - resultado["inicio"]
    resultado["incoherente"] = resultado["duracion"] < pd.Timedelta(0)
    return resultado


# ----------------------------------------------------------------------
# Histograma de densidad (para el panel superior + slider de rango)
# ----------------------------------------------------------------------

_ESCALAS_BIN = [
    ("hora", pd.Timedelta(hours=1)),
    ("dia", pd.Timedelta(days=1)),
    ("semana", pd.Timedelta(weeks=1)),
    ("mes", pd.Timedelta(days=30)),
    ("trimestre", pd.Timedelta(days=91)),
    ("anio", pd.Timedelta(days=365)),
]


def elegir_escala_bins(fechas: pd.Series, objetivo_bins: int = 60):
    """Elige automáticamente el ancho de bin (hora/día/semana/mes/...)
    que deja el histograma con un número de barras legible -- ni un muro
    de barras de una hora en 3 años de datos, ni 2 barras gigantes en
    una semana de datos. Es la parte "Tufte" de la especificación: el
    ancho se decide por los datos, nunca queda fijo a mano."""
    if fechas.empty:
        return "dia", pd.Timedelta(days=1)
    rango = fechas.max() - fechas.min()
    if rango <= pd.Timedelta(0):
        return "dia", pd.Timedelta(days=1)
    for nombre, ancho in _ESCALAS_BIN:
        if rango / ancho <= objetivo_bins:
            return nombre, ancho
    return _ESCALAS_BIN[-1]


def construir_histograma(fechas: pd.Series, objetivo_bins: int = 60):
    """Devuelve (bordes_bins, conteos) listos para dibujar como barras
    de densidad. bordes_bins tiene un elemento más que conteos (son los
    límites de cada barra, no sus centros)."""
    if fechas.empty:
        return np.array([]), np.array([])
    _, ancho = elegir_escala_bins(fechas, objetivo_bins)
    piso = "D" if ancho >= pd.Timedelta(days=1) else "h"
    inicio = fechas.min().floor(piso)
    fin = fechas.max() + ancho
    bordes = pd.date_range(inicio, fin, freq=ancho)
    if len(bordes) < 2:
        bordes = pd.date_range(inicio, inicio + ancho * 2, freq=ancho)
    conteos, _ = np.histogram(
        fechas.to_numpy().astype("int64"), bins=bordes.to_numpy().astype("int64")
    )
    return bordes, conteos


# ----------------------------------------------------------------------
# Cruce de anomalías (anomalias.py) con la línea de tiempo
# ----------------------------------------------------------------------

def anomalias_con_fecha(anomalias: list[dict], df: pd.DataFrame, columna_fecha_referencia: str, dayfirst: bool = True):
    """Cruza cada anomalía detectada con la fecha real de la(s) fila(s)
    donde ocurrió, usando la columna de fecha elegida como referencia
    para la Línea de Tiempo. Una anomalía que afecta muchas filas (ej.
    'quiebre_patron' agrupado) queda anclada en la fecha MEDIANA de sus
    filas afectadas -- el marcador representa el centro real del
    fenómeno, no el primer o último caso encontrado. Ver
    construir_serie_hitos para qué decide dayfirst."""
    if columna_fecha_referencia not in df.columns:
        return []
    fechas_df = parsear_fechas(df[columna_fecha_referencia], dayfirst=dayfirst)

    resultado = []
    for a in anomalias:
        indices = a.get("indices_atipicos")
        if not indices:
            idx_unico = a.get("fila_indice")
            indices = [idx_unico] if idx_unico is not None else []
        indices_validos = [i for i in indices if i in fechas_df.index and pd.notna(fechas_df.loc[i])]
        if not indices_validos:
            continue
        fecha_marcador = fechas_df.loc[indices_validos].median()
        resultado.append({"fecha": fecha_marcador, "anomalia": a})
    return resultado


# ----------------------------------------------------------------------
# Duraciones entre dos puntos conectados (automáticos o manuales)
# ----------------------------------------------------------------------

_UNIDADES_DURACION = [
    ("año", "años", 365.25 * 86400),
    ("mes", "meses", 30 * 86400),
    ("día", "días", 86400),
    ("hora", "horas", 3600),
    ("minuto", "minutos", 60),
]


def formatear_duracion(delta: pd.Timedelta) -> str:
    """Convierte un Timedelta a texto natural con como máximo 2 unidades
    (ej. '1 año, 5 meses', '3 días, 5 horas', '45 minutos'). Nunca baja a
    segundos -- a esa escala ya no aporta a una lectura de negocio. Una
    duración negativa (el punto 'fin' quedó antes que el 'inicio') se
    muestra en valor absoluto con una nota explícita, no como un número
    negativo que el usuario tendría que interpretar."""
    segundos = delta.total_seconds()
    absolutos = abs(segundos)
    partes = []
    restante = absolutos
    for singular, plural_form, tam in _UNIDADES_DURACION:
        cantidad = int(restante // tam)
        if cantidad > 0:
            etiqueta = singular if cantidad == 1 else plural_form
            partes.append(f"{cantidad} {etiqueta}")
            restante -= cantidad * tam
        if len(partes) == 2:
            break
    if not partes:
        minutos = max(1, round(absolutos / 60))
        partes = [f"{minutos} {'minuto' if minutos == 1 else 'minutos'}"]
    texto = ", ".join(partes)
    if segundos < 0:
        texto += " (orden invertido)"
    return texto


def resumen_duraciones(duraciones) -> dict | None:
    """Promedio, mínimo y máximo de una lista/Serie de Timedelta, ya
    formateados como texto natural. None si no hay ninguna duración
    válida. Las incoherentes (negativas) SÍ entran al cálculo -- son
    parte real de los datos, no un error a esconder; el promedio las
    refleja tal cual, y cada valor queda igual marcado si es negativo."""
    validas = [d for d in duraciones if pd.notna(d)]
    if not validas:
        return None
    promedio = sum(validas, pd.Timedelta(0)) / len(validas)
    return {
        "n": len(validas),
        "promedio": formatear_duracion(promedio),
        "minimo": formatear_duracion(min(validas)),
        "maximo": formatear_duracion(max(validas)),
        "n_incoherentes": sum(1 for d in validas if d.total_seconds() < 0),
    }