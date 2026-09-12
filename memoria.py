"""
Memoria persistente local (SQLite) de los procesos de negocio detectados
y las anomalías encontradas en el tiempo. Le da a Narrativa continuidad
entre distintas cargas del mismo tipo de dataset (fingerprint de esquema).
"""
import os
import sys
import hashlib
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

def _sin_acentos_memoria(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


def _normalizar_nombre_columna_memoria(nombre):
    return _sin_acentos_memoria(nombre).strip().replace(" ", "_")


def _tipo_generico_memoria(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return "numerico"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "fecha"
    muestra = serie.dropna()
    if not muestra.empty:
        parseado = pd.to_datetime(muestra, errors="coerce", format="mixed")
        if parseado.notna().mean() >= 0.8:
            return "fecha"
    return "categorico"


def generar_fingerprint_esquema(df):
    """Devuelve (fingerprint_corto, firma_legible), estable ante cambios de
    orden de columnas y de mayúsculas/acentos en los nombres; cambia si
    cambian las columnas o sus tipos genéricos."""
    firma_columnas = sorted(
        f"{_normalizar_nombre_columna_memoria(col)}:{_tipo_generico_memoria(df[col])}"
        for col in df.columns
    )
    firma_texto = "|".join(firma_columnas)
    fingerprint = hashlib.sha256(firma_texto.encode("utf-8")).hexdigest()[:16]
    return fingerprint, firma_texto


def ruta_directorio_datos_usuario():
    """Carpeta de datos persistente por usuario, FUERA de la carpeta donde
    vive el ejecutable. Necesario porque una app empaquetada con PyInstaller
    normalmente no tiene permiso de escritura junto a sí misma -- si la
    memoria se guardara ahí, fallaría en la máquina de un usuario real aunque
    funcione perfecto en la máquina de desarrollo."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    carpeta = os.path.join(base, "Hadar Data Analytics")
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def ruta_memoria_db():
    return os.path.join(ruta_directorio_datos_usuario(), "hadar_memoria.db")


@contextmanager
def _conexion_memoria(ruta_db):
    con = sqlite3.connect(ruta_db)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _ahora_memoria():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dias_entre_memoria(fecha_iso_a, fecha_iso_b):
    a = datetime.fromisoformat(fecha_iso_a)
    b = datetime.fromisoformat(fecha_iso_b)
    return abs((b - a).days)


def _frase_recurrencia_memoria(veces, dias):
    if veces <= 1:
        return None
    if dias == 0:
        return f"Ya se había detectado antes ({veces} veces en total)."
    return (
        f"Este problema ya apareció hace {dias} día(s) y sigue sin resolverse "
        f"({veces} veces detectado en total)."
    )


def _frase_impacto_acumulado(impacto_acumulado, columnas=None):
    """Frase lista para mostrar con la suma del impacto (diferencia contra
    lo típico, ver anomalias.py) de todas las veces que esta anomalía se
    detectó sin resolverse. None si no hay ningún impacto numérico que
    sumar (ej. anomalías que no son de tipo 'quiebre_patron', que no traen
    ese dato)."""
    if impacto_acumulado is None:
        return None
    columna_txt = f" en '{columnas[0]}'" if columnas else ""
    direccion = "por encima" if impacto_acumulado > 0 else "por debajo"
    return (
        f"Sumando todas las veces que esto pasó sin resolverse, el impacto acumulado "
        f"estimado{columna_txt} es de {abs(impacto_acumulado):,.2f} {direccion} de lo típico."
    )


class MemoriaHadar:
    """Memoria persistente local (un solo archivo SQLite) de los procesos de
    negocio detectados y las anomalías que se les ha encontrado en el tiempo.
    Ver módulo standalone memoria_hadar.py para la versión documentada con
    pruebas -- esta es la misma lógica, ya cableada a HadarApp.
    """

    def __init__(self, ruta_db=None):
        self.ruta_db = ruta_db or ruta_memoria_db()
        self._crear_tablas()

    def _crear_tablas(self):
        with _conexion_memoria(self.ruta_db) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS procesos (
                    fingerprint TEXT PRIMARY KEY,
                    nombre TEXT,
                    firma_esquema TEXT,
                    primera_vez TEXT,
                    ultima_vez TEXT,
                    veces_cargado INTEGER DEFAULT 1
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS anomalias_historicas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT,
                    tipo TEXT,
                    columnas TEXT,
                    descripcion TEXT,
                    gravedad TEXT,
                    fecha_deteccion TEXT,
                    resuelta INTEGER DEFAULT 0,
                    impacto REAL
                )
            """)
            # Migración suave: si la tabla ya existía de antes de que
            # existiera el impacto acumulado, le sumamos la columna sin
            # perder el historial ya guardado.
            columnas_existentes = {
                fila[1] for fila in con.execute("PRAGMA table_info(anomalias_historicas)")
            }
            if "impacto" not in columnas_existentes:
                con.execute("ALTER TABLE anomalias_historicas ADD COLUMN impacto REAL")
            # Relaciones de orden temporal definidas a mano por el usuario para
            # este esquema de datos (ej. "aprobacion" debe ir antes que
            # "entrega"), usando los nombres de columna REALES -- no dependen
            # de que un token fijo calce con el nombre de la columna. Se
            # guardan por fingerprint para que no haya que repetir la
            # configuración cada vez que se carga el mismo tipo de dataset.
            con.execute("""
                CREATE TABLE IF NOT EXISTS pares_temporales_personalizados (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT,
                    col_anterior TEXT,
                    col_posterior TEXT,
                    etiqueta TEXT
                )
            """)

    @staticmethod
    def _clave_identidad(anomalia):
        columnas = tuple(sorted(anomalia.get("columnas") or []))
        return anomalia["tipo"], columnas

    def registrar_carga(self, df, nombre_proceso=None):
        """Registra (o actualiza) que se cargó un dataset con esta
        estructura. Devuelve el fingerprint para usar en el resto de la
        sesión (ej. al generar el análisis narrativo)."""
        fingerprint, firma = generar_fingerprint_esquema(df)
        ahora = _ahora_memoria()
        with _conexion_memoria(self.ruta_db) as con:
            existente = con.execute(
                "SELECT fingerprint FROM procesos WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if existente:
                con.execute(
                    "UPDATE procesos SET ultima_vez = ?, veces_cargado = veces_cargado + 1 "
                    "WHERE fingerprint = ?",
                    (ahora, fingerprint),
                )
            else:
                con.execute(
                    "INSERT INTO procesos "
                    "(fingerprint, nombre, firma_esquema, primera_vez, ultima_vez) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (fingerprint, nombre_proceso or "Proceso sin nombre", firma, ahora, ahora),
                )
        return fingerprint

    def enriquecer_con_memoria(self, fingerprint, anomalias_actuales):
        """Para cada anomalía detectada AHORA: busca si ya se había visto
        antes (mismo tipo + columnas) en este proceso y le agrega campos de
        memoria. Registra también la ocurrencia de hoy para la próxima vez
        -- pero solo una vez por día: si el usuario genera la narrativa
        varias veces el mismo día sobre los mismos datos, no se infla
        artificialmente "veces_detectada". Devuelve una lista NUEVA (no muta
        la entrada)."""
        ahora = _ahora_memoria()
        hoy = ahora[:10]  # _ahora_memoria() es ISO 8601 -> primeros 10 chars = fecha
        enriquecidas = []
        with _conexion_memoria(self.ruta_db) as con:
            for anomalia in anomalias_actuales:
                tipo, columnas = self._clave_identidad(anomalia)
                columnas_str = ",".join(columnas)
                impacto_actual = anomalia.get("diferencia_absoluta")

                previas = con.execute(
                    "SELECT fecha_deteccion, resuelta, impacto FROM anomalias_historicas "
                    "WHERE fingerprint = ? AND tipo = ? AND columnas = ? "
                    "ORDER BY fecha_deteccion ASC",
                    (fingerprint, tipo, columnas_str),
                ).fetchall()
                ya_registrada_hoy = bool(previas) and previas[-1][0][:10] == hoy

                # La frase de recurrencia solo debe hablar de la racha ACTUAL
                # sin resolver: si en algún momento se marcó "resuelta", la
                # racha se corta ahí -- de lo contrario, "Marcar como
                # resuelta" no cambiaría nada en el texto la próxima vez.
                # El impacto acumulado sigue la misma racha: si se resolvió,
                # el acumulado también vuelve a cero (es una racha nueva).
                racha_sin_resolver = []
                for fecha, resuelta, impacto in previas:
                    racha_sin_resolver = [] if resuelta else racha_sin_resolver + [(fecha, impacto)]

                anomalia_enriquecida = dict(anomalia)
                if previas:
                    veces_total = len(previas) if ya_registrada_hoy else len(previas) + 1
                    if racha_sin_resolver:
                        dias = _dias_entre_memoria(racha_sin_resolver[0][0], ahora)
                        veces_racha = len(racha_sin_resolver) if ya_registrada_hoy else len(racha_sin_resolver) + 1
                        frase = _frase_recurrencia_memoria(veces_racha, dias)

                        impactos = [imp for (_, imp) in racha_sin_resolver if imp is not None]
                        if not ya_registrada_hoy and impacto_actual is not None:
                            impactos = impactos + [impacto_actual]
                        impacto_acumulado = sum(impactos) if impactos else None
                    else:
                        # Todas las detecciones anteriores están resueltas:
                        # esto es una reaparición nueva, no una racha vieja.
                        dias = 0
                        frase = "Habías marcado esta anomalía como resuelta, pero volvió a aparecer."
                        veces_racha = 1
                        impacto_acumulado = impacto_actual
                    anomalia_enriquecida.update({
                        "veces_detectada": veces_total,
                        "primera_deteccion": previas[0][0],
                        "dias_sin_resolver": dias,
                        "frase_memoria": frase,
                        "impacto_acumulado": impacto_acumulado,
                        # Con una sola ocurrencia en esta racha no hay nada
                        # que "acumular" todavía -- misma vara que ya usa
                        # _frase_recurrencia_memoria (nada con veces <= 1).
                        "frase_impacto_acumulado": (
                            _frase_impacto_acumulado(impacto_acumulado, anomalia.get("columnas"))
                            if veces_racha > 1 else None
                        ),
                    })
                else:
                    anomalia_enriquecida.update({
                        "veces_detectada": 1,
                        "primera_deteccion": ahora,
                        "dias_sin_resolver": 0,
                        "frase_memoria": None,
                        "impacto_acumulado": impacto_actual,
                        "frase_impacto_acumulado": None,  # primera vez: nada que "acumular" todavía
                    })

                if not ya_registrada_hoy:
                    con.execute(
                        "INSERT INTO anomalias_historicas "
                        "(fingerprint, tipo, columnas, descripcion, gravedad, fecha_deteccion, impacto) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (fingerprint, tipo, columnas_str, anomalia.get("descripcion", ""),
                         anomalia.get("gravedad", "media"), ahora, impacto_actual),
                    )
                enriquecidas.append(anomalia_enriquecida)
        return enriquecidas

    def historial_proceso(self, fingerprint):
        columnas_tabla = ["tipo", "columnas", "descripcion", "gravedad",
                           "fecha_deteccion", "resuelta"]
        with _conexion_memoria(self.ruta_db) as con:
            filas = con.execute(
                f"SELECT {', '.join(columnas_tabla)} FROM anomalias_historicas "
                "WHERE fingerprint = ? ORDER BY fecha_deteccion",
                (fingerprint,),
            ).fetchall()
        return [dict(zip(columnas_tabla, fila)) for fila in filas]

    def marcar_resuelta(self, fingerprint, tipo, columnas):
        columnas_str = ",".join(sorted(columnas))
        with _conexion_memoria(self.ruta_db) as con:
            con.execute(
                "UPDATE anomalias_historicas SET resuelta = 1 "
                "WHERE fingerprint = ? AND tipo = ? AND columnas = ? AND resuelta = 0",
                (fingerprint, tipo, columnas_str),
            )

    def obtener_pares_temporales(self, fingerprint):
        """Devuelve las relaciones de orden temporal ("esta columna debe ir
        antes que esta otra") que el usuario configuró a mano para este
        esquema de datos. Lista de dicts con nombres de columna reales."""
        with _conexion_memoria(self.ruta_db) as con:
            filas = con.execute(
                "SELECT col_anterior, col_posterior, etiqueta "
                "FROM pares_temporales_personalizados WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchall()
        return [
            {"col_anterior": a, "col_posterior": p, "etiqueta": e}
            for a, p, e in filas
        ]

    def guardar_pares_temporales(self, fingerprint, pares):
        """Reemplaza por completo la configuración de relaciones temporales
        personalizadas de este fingerprint por `pares` (lista de dicts con
        col_anterior/col_posterior/etiqueta)."""
        with _conexion_memoria(self.ruta_db) as con:
            con.execute(
                "DELETE FROM pares_temporales_personalizados WHERE fingerprint = ?",
                (fingerprint,),
            )
            con.executemany(
                "INSERT INTO pares_temporales_personalizados "
                "(fingerprint, col_anterior, col_posterior, etiqueta) VALUES (?, ?, ?, ?)",
                [(fingerprint, p["col_anterior"], p["col_posterior"], p.get("etiqueta") or
                  f"{p['col_anterior']} → {p['col_posterior']}") for p in pares],
            )


# ----------------------------------------------------------------------------
# Pestaña Narrativa: detección de anomalías genérica + informe HTML tipo
# "libro" + preguntas sugeridas. No asume el dominio del dataset (nunca
# nombres de columna fijos como "ventas" o "clientes"): se detecta por tipo
# de dato y por tokens en el nombre de columna, igual que el resto de la app.
# ----------------------------------------------------------------------------