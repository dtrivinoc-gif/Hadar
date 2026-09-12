"""
ontologia.py — Motor de inferencia de relaciones entre tablas para Hadar.

Este módulo NO depende de PySide6 ni de nada de la interfaz. Su única
responsabilidad es: dado un diccionario de tablas (nombre -> DataFrame),
adivinar qué columnas de una tabla probablemente se relacionan con qué
columnas de otra tabla, y devolver esa lista con un nivel de confianza
y una explicación en español, pensada para mostrarse tal cual al usuario.

Idea general (en simple):
  - Si dos columnas se llaman parecido (ej. "id_cliente" en ambas tablas),
    es una pista fuerte.
  - Si además los VALORES se parecen (ej. los mismos números o códigos
    aparecen en las dos columnas), es una pista todavía más fuerte —
    de hecho es la prueba real de que hay una relación, no solo una
    coincidencia de nombre.
  - Si los tipos de datos son incompatibles (una es texto libre tipo
    "comentarios" y la otra es un número), se descarta esa relación.
  - Además, se intenta adivinar cuál de las dos columnas es la "tabla
    principal" (ej. Clientes.id_cliente, que no se repite) y cuál es la
    "tabla que la referencia" (ej. Pedidos.id_cliente, que sí se repite).

No hace falta que el usuario entienda nada de esto: el resultado es una
lista de relaciones sugeridas, ordenadas por confianza, listas para
dibujarse como líneas entre cajitas.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd


# ----------------------------------------------------------------------
# Configuración de umbrales (fáciles de retocar más adelante)
# ----------------------------------------------------------------------

UMBRAL_CONFIANZA_DEFECTO = 0.40      # bajo esto, ni se sugiere
PESO_NOMBRE = 0.40                   # cuánto pesa que los nombres se parezcan
PESO_TIPO = 0.10                     # cuánto pesa que los tipos calcen
PESO_VALORES = 0.50                  # cuánto pesa que los valores se solapen
MAX_FILAS_MUESTRA_VALORES = 5000     # para no comparar sets gigantes
MAX_RELACIONES_POR_PAR_DE_TABLAS = 5  # evita saturar el diagrama


# ----------------------------------------------------------------------
# Estructura de salida
# ----------------------------------------------------------------------

@dataclass
class RelacionSugerida:
    tabla_origen: str
    columna_origen: str
    tabla_destino: str
    columna_destino: str
    confianza: float                 # 0.0 a 1.0
    razon: str                       # explicación en español, para mostrar al usuario
    tabla_principal: str | None = None   # cuál de las dos tablas parece ser la "principal" (PK)
    columna_principal: str | None = None
    certeza_direccion: str = "sin_definir"  # "alta" | "media" | "sin_definir"
    # "alta": una de las dos columnas es casi toda valores únicos (huella clásica de PK),
    #          o el usuario confirmó/corrigió la dirección a mano (ver direccion_confirmada).
    # "media": ninguna es claramente única, pero una tabla es notoriamente más chica
    #          que la otra (pista más débil, pensada como padre de una relación 1-a-N).
    # "sin_definir": no hay pista suficiente -- tabla_principal queda en None y el
    #          motor de contagio (u otra parte de la app) no debería asumir dirección
    #          sin que el usuario la confirme a mano.
    direccion_confirmada: bool = False  # True si un humano eligió/corrigió tabla_principal
    #          a mano en el diagrama (distinto de 'manual', que es toda la relación
    #          agregada a mano). Sirve para que "Detectar automáticamente de nuevo"
    #          no pise la corrección del usuario.
    manual: bool = False             # True si el usuario la agregó a mano (no la detectó el motor)
    detalle: dict = field(default_factory=dict)  # datos extra por si la UI los quiere (opcional)

    def como_texto(self) -> str:
        """Frase corta lista para mostrar en la interfaz."""
        return (
            f"{self.tabla_origen}.{self.columna_origen}  ↔  "
            f"{self.tabla_destino}.{self.columna_destino}  "
            f"({int(self.confianza * 100)}% de confianza) — {self.razon}"
        )


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------

def _normalizar_nombre(nombre: str) -> str:
    """
    'ID_Cliente', 'id cliente', 'Id-Cliente' -> 'idcliente'
    Saca tildes, mayúsculas, espacios, guiones y guiones bajos, para que
    la comparación de nombres no falle por detalles de formato.
    """
    texto = str(nombre).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[\s_\-\.]+", "", texto)
    return texto


def _similitud_nombres(col1: str, col2: str) -> float:
    """Similitud de texto entre 0 y 1 usando los nombres normalizados."""
    n1, n2 = _normalizar_nombre(col1), _normalizar_nombre(col2)
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    return SequenceMatcher(None, n1, n2).ratio()


def _tipos_compatibles(serie1: pd.Series, serie2: pd.Series) -> bool:
    """
    True si tiene sentido comparar los VALORES de estas dos columnas
    (ambas numéricas, o ambas texto/categoría). Evita, por ejemplo,
    comparar una columna de fechas con una de texto libre.
    """
    es_num1, es_num2 = pd.api.types.is_numeric_dtype(serie1), pd.api.types.is_numeric_dtype(serie2)
    es_fecha1, es_fecha2 = pd.api.types.is_datetime64_any_dtype(serie1), pd.api.types.is_datetime64_any_dtype(serie2)

    if es_fecha1 or es_fecha2:
        # las fechas casi nunca son claves de relación tabla-tabla en este contexto
        return es_fecha1 and es_fecha2
    if es_num1 != es_num2:
        return False
    return True


def _muestra_valores(serie: pd.Series) -> set:
    """Set de valores no nulos, recortado para no comparar columnas gigantes."""
    s = serie.dropna()
    if len(s) > MAX_FILAS_MUESTRA_VALORES:
        s = s.sample(MAX_FILAS_MUESTRA_VALORES, random_state=0)
    # normaliza texto para que "ABC" y "abc" no se consideren distintos
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
    return set(s.tolist())


def _solape_valores(serie1: pd.Series, serie2: pd.Series) -> float:
    """
    Qué tan solapados están los valores de dos columnas, de 0 a 1.
    Se mide contra el más chico de los dos conjuntos (para que una
    tabla "Clientes" con 50 ids y una "Pedidos" con 5000 filas, donde
    los 50 ids aparecen todos, dé un solape alto y correcto).
    """
    v1, v2 = _muestra_valores(serie1), _muestra_valores(serie2)
    if not v1 or not v2:
        return 0.0
    interseccion = v1 & v2
    mas_chico = min(len(v1), len(v2))
    if mas_chico == 0:
        return 0.0
    return len(interseccion) / mas_chico


def _tokens_columna(nombre_columna: str) -> list[str]:
    """
    Separa un nombre de columna en 'palabras': 'id_cliente' -> ['id','cliente'],
    'IdCliente' -> ['id','cliente'], 'idcliente' -> ['idcliente'] (sin separador
    no se puede partir, se deja como una sola palabra).
    """
    texto = str(nombre_columna).strip()
    # separa por guiones/espacios/puntos y por cambios minúscula->mayúscula (camelCase)
    texto = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", texto)
    partes = re.split(r"[\s_\-\.]+", texto)
    return [_normalizar_nombre(p) for p in partes if p]


TOKENS_GENERICOS = {"id", "codigo", "cod", "rut", "folio", "numero", "key", "clave", "pk", "fk"}
UMBRAL_SIMILITUD_SEMANTICA = 0.6  # bajo esto, se asume que son entidades distintas


def _tokens_semanticos(nombre_columna: str) -> list[str]:
    """
    Los 'tokens con significado' de un nombre de columna, descartando
    sufijos/prefijos genéricos como 'id', 'codigo', 'folio', etc.
    'categoria_id' -> ['categoria'], 'id_cliente' -> ['cliente'],
    'cantidad' -> ['cantidad'] (no tiene sufijo genérico, es toda la palabra).
    """
    return [t for t in _tokens_columna(nombre_columna) if t not in TOKENS_GENERICOS]


def _similitud_semantica(col1: str, col2: str) -> float | None:
    """
    Compara solo la parte 'con nombre de entidad' de dos columnas, ignorando
    el sufijo genérico (_id, _codigo, etc). Es la pista que evita cruces
    como 'categoria_id' con 'cliente_id': el string completo se parece
    bastante (ambos terminan en 'id'), pero la entidad real -- categoria
    vs. cliente -- no tiene nada que ver.

    Devuelve un número de 0 a 1 (más alto = misma entidad), o None si
    alguna de las dos columnas es un identificador genérico puro sin parte
    semántica (ej. una columna llamada literalmente 'id') -- en ese caso
    no hay nada que comparar y no se aplica este filtro.
    """
    sem1, sem2 = _tokens_semanticos(col1), _tokens_semanticos(col2)
    if not sem1 or not sem2:
        return None
    mejor = 0.0
    for t1 in sem1:
        for t2 in sem2:
            mejor = max(mejor, SequenceMatcher(None, t1, t2).ratio())
    return mejor


def _parece_columna_clave(nombre_columna: str) -> bool:
    """
    Pistas típicas de que una columna es un identificador (id, código, etc.).
    Compara por PALABRA COMPLETA (tokens), no por subcadena, para que
    'cantidad' no se confunda con 'id' solo porque contiene esas letras.
    """
    patrones = {"id", "codigo", "cod", "rut", "folio", "numero", "key", "clave", "pk", "fk"}
    tokens = _tokens_columna(nombre_columna)
    if any(t in patrones for t in tokens):
        return True
    # nombres pegados sin separador tipo 'idcliente' / 'clienteid'
    nombre_completo = _normalizar_nombre(nombre_columna)
    return nombre_completo.startswith("id") or nombre_completo.endswith("id")


UMBRAL_UNICIDAD_PK = 0.95          # "casi todos los valores únicos" = huella de PK
UMBRAL_UNICIDAD_MINIMA_TAMANO = 0.5  # bajo esto, ni vale la pena mirar el tamaño
UMBRAL_TAMANO_NOTORIO = 0.6        # la tabla chica debe tener a lo más este % de filas de la grande


def _tabla_principal(
    serie1: pd.Series, tabla1: str, total_filas1: int,
    serie2: pd.Series, tabla2: str, total_filas2: int,
):
    """
    Adivina cuál de las dos tablas es la 'principal' (lista maestra, ej.
    Productos) y cuál es la que 'referencia' (ej. Detalle_Ventas), usando
    dos pistas en orden de confianza:

      1. Unicidad de la columna (pista fuerte): si casi todos los valores
         de una columna son distintos entre sí, esa columna se comporta
         como una llave primaria -- típico de la tabla principal.
      2. Tamaño de la tabla (pista más débil, solo como desempate): en una
         relación 1-a-N, la tabla principal casi siempre tiene MENOS filas
         que la tabla que la referencia (ej. 50 productos vs 8.000 líneas
         de venta). Solo se usa si ninguna columna dio una señal clara de
         unicidad, y solo si una tabla es notoriamente más chica que la
         otra (para no adivinar a ciegas cuando son de tamaño parecido).

    Devuelve (nombre_tabla_principal | None, certeza: "alta" | "media" | "sin_definir").
    """
    u1 = serie1.nunique(dropna=True) / max(len(serie1.dropna()), 1)
    u2 = serie2.nunique(dropna=True) / max(len(serie2.dropna()), 1)

    # Pista 1 (fuerte): unicidad de columna.
    if u1 > UMBRAL_UNICIDAD_PK and u2 <= UMBRAL_UNICIDAD_PK:
        return tabla1, "alta"
    if u2 > UMBRAL_UNICIDAD_PK and u1 <= UMBRAL_UNICIDAD_PK:
        return tabla2, "alta"

    # Pista 2 (débil, solo desempate): tamaño relativo de las tablas.
    # Se exige que la columna candidata a principal tenga al menos una
    # unicidad razonable (si no, ni siquiera parece una llave) y que la
    # diferencia de tamaño entre tablas sea notoria, no cualquier cosa.
    if total_filas1 > 0 and total_filas2 > 0:
        if u1 >= UMBRAL_UNICIDAD_MINIMA_TAMANO and total_filas1 <= total_filas2 * UMBRAL_TAMANO_NOTORIO:
            return tabla1, "media"
        if u2 >= UMBRAL_UNICIDAD_MINIMA_TAMANO and total_filas2 <= total_filas1 * UMBRAL_TAMANO_NOTORIO:
            return tabla2, "media"

    return None, "sin_definir"


# ----------------------------------------------------------------------
# Función principal
# ----------------------------------------------------------------------

def inferir_relaciones(
    tablas: dict[str, pd.DataFrame],
    umbral_confianza: float = UMBRAL_CONFIANZA_DEFECTO,
) -> list[RelacionSugerida]:
    """
    Punto de entrada del motor.

    Parámetros
    ----------
    tablas : dict {nombre_tabla: DataFrame}
        Ej: {"Clientes": df_clientes, "Pedidos": df_pedidos}
    umbral_confianza : float
        Relaciones por debajo de esto no se incluyen en el resultado.

    Devuelve
    --------
    Lista de RelacionSugerida, ordenada de mayor a menor confianza.
    Si solo hay 0 o 1 tabla, devuelve lista vacía (no hay nada que
    relacionar — así la pestaña Datos sabe que no debe mostrar el botón).
    """
    nombres_tablas = list(tablas.keys())
    if len(nombres_tablas) < 2:
        return []

    candidatas: list[RelacionSugerida] = []

    for i in range(len(nombres_tablas)):
        for j in range(i + 1, len(nombres_tablas)):
            t1, t2 = nombres_tablas[i], nombres_tablas[j]
            df1, df2 = tablas[t1], tablas[t2]

            relaciones_del_par: list[RelacionSugerida] = []

            for col1 in df1.columns:
                for col2 in df2.columns:
                    sim_nombre = _similitud_nombres(col1, col2)
                    ambas_son_clave = _parece_columna_clave(col1) and _parece_columna_clave(col2)

                    # Filtro de nombre. Ojo: cuando AMBAS columnas son de tipo
                    # 'id numérico' (ej. id_cliente vs id_pedido), exigimos más
                    # parecido de nombre que en el caso general — si no, dos
                    # columnas de ids chicos (1,2,3...) casi siempre se van a
                    # 'solapar' por pura casualidad y generan relaciones falsas.
                    umbral_nombre = 0.5 if ambas_son_clave else 0.35
                    if sim_nombre < umbral_nombre:
                        continue

                    # Filtro semántico: aunque el nombre completo se parezca
                    # (ej. 'categoria_id' y 'cliente_id' comparten el sufijo
                    # '_id'), si la parte que nombra la ENTIDAD no tiene nada
                    # que ver ('categoria' vs. 'cliente'), se descarta antes
                    # de mirar los valores -- si no, columnas chicas de rango
                    # parecido (ids, cantidades) generan cruces sin sentido
                    # solo por casualidad numérica.
                    sim_semantica = _similitud_semantica(col1, col2)
                    if sim_semantica is not None and sim_semantica < UMBRAL_SIMILITUD_SEMANTICA:
                        continue

                    if not _tipos_compatibles(df1[col1], df2[col2]):
                        continue

                    solape = _solape_valores(df1[col1], df2[col2])

                    confianza = (
                        PESO_NOMBRE * sim_nombre
                        + PESO_TIPO * 1.0
                        + PESO_VALORES * solape
                    )

                    if confianza < umbral_confianza:
                        continue

                    # Arma la razón en español, priorizando la pista más fuerte
                    partes_razon = []
                    if sim_nombre >= 0.8:
                        partes_razon.append("los nombres de columna son casi iguales")
                    elif sim_nombre >= 0.5:
                        partes_razon.append("los nombres de columna se parecen")
                    if solape >= 0.7:
                        partes_razon.append("la mayoría de los valores coinciden")
                    elif solape >= 0.3:
                        partes_razon.append("varios valores coinciden")
                    if not partes_razon:
                        partes_razon.append("hay una coincidencia parcial de nombre y tipo de dato")

                    principal, certeza_direccion = _tabla_principal(
                        df1[col1], t1, len(df1), df2[col2], t2, len(df2)
                    )
                    col_principal = col1 if principal == t1 else (col2 if principal == t2 else None)

                    if certeza_direccion == "media":
                        partes_razon.append(f"{principal} parece la tabla principal por tener menos filas")
                    elif certeza_direccion == "sin_definir":
                        partes_razon.append("no está claro cuál tabla es la principal, revisa la dirección a mano")

                    razon = ", ".join(partes_razon).capitalize()

                    relaciones_del_par.append(
                        RelacionSugerida(
                            tabla_origen=t1,
                            columna_origen=col1,
                            tabla_destino=t2,
                            columna_destino=col2,
                            confianza=round(min(confianza, 1.0), 3),
                            razon=razon,
                            tabla_principal=principal,
                            columna_principal=col_principal,
                            certeza_direccion=certeza_direccion,
                            detalle={"similitud_nombre": round(sim_nombre, 3), "solape_valores": round(solape, 3)},
                        )
                    )

            # De todas las combinaciones de columnas entre estas dos tablas,
            # nos quedamos solo con las mejores (para no llenar el diagrama
            # de líneas redundantes o ruidosas).
            relaciones_del_par.sort(key=lambda r: r.confianza, reverse=True)
            candidatas.extend(relaciones_del_par[:MAX_RELACIONES_POR_PAR_DE_TABLAS])

    candidatas.sort(key=lambda r: r.confianza, reverse=True)
    return candidatas


# ----------------------------------------------------------------------
# Demo / auto-test rápido con datos de ejemplo (Clientes / Pedidos / Productos)
# ----------------------------------------------------------------------

def _demo():
    clientes = pd.DataFrame({
        "id_cliente": [1, 2, 3, 4, 5],
        "nombre": ["Ana", "Beto", "Carla", "Diego", "Elena"],
        "region": ["RM", "Biobío", "RM", "Valparaíso", "RM"],
    })

    productos = pd.DataFrame({
        "id_producto": [100, 101, 102],
        "nombre_producto": ["Teclado", "Mouse", "Monitor"],
        "precio": [15990, 8990, 129990],
    })

    pedidos = pd.DataFrame({
        "id_pedido": range(1, 11),
        "id_cliente": [1, 1, 2, 3, 3, 3, 4, 5, 5, 2],
        "id_producto": [100, 101, 102, 100, 100, 101, 102, 100, 101, 102],
        "cantidad": [1, 2, 1, 1, 3, 1, 1, 2, 1, 1],
        "comentario": ["ok", "urgente", "", "", "regalo", "", "", "", "ok", ""],
    })

    tablas = {"Clientes": clientes, "Pedidos": pedidos, "Productos": productos}

    print("=== Relaciones sugeridas ===\n")
    relaciones = inferir_relaciones(tablas)
    if not relaciones:
        print("(no se encontraron relaciones por encima del umbral)")
    for r in relaciones:
        marca_principal = f"  [tabla principal: {r.tabla_principal}]" if r.tabla_principal else ""
        print(" - " + r.como_texto() + marca_principal)

    print("\n=== Caso de 1 sola tabla (no debe sugerir nada) ===")
    print(inferir_relaciones({"Clientes": clientes}))


if __name__ == "__main__":
    _demo()