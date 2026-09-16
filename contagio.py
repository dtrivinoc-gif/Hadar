"""
contagio.py — Motor de propagación de anomalías (Índice de Contagio) para Hadar.

Dada una anomalía detectada en una tabla (ej. un precio raro en Productos),
calcula qué otras tablas "aguas abajo" (conectadas por relaciones ya
confirmadas en ontologia.py) tienen filas que dependen de esas filas
anómalas -- y por lo tanto quedan "infectadas" por cascada.

Importante: esto NO es machine learning. Es un recorrido de grafo (BFS)
sobre relaciones que ontologia.py ya detectó (o que el usuario confirmó a
mano), usando los valores reales de las columnas clave para encontrar
coincidencias exactas. No hay nada probabilístico ni entrenado: mismos
datos + misma relación = mismo resultado, siempre.

Simplificación consciente: se asume que cada tabla tiene UNA columna que
actúa como su llave "hacia afuera" (su PK lógica) y que todas las
relaciones donde esa tabla es la principal usan esa misma columna. Es el
caso normal (ej. Productos siempre se referencia por producto_id). Si en
algún caso real una tabla necesitara más de una columna clave distinta
para relacionarse con distintas tablas hijas, este motor no lo cubre
todavía -- quedaría para una vuelta futura si aparece un caso así.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ontologia import RelacionSugerida


@dataclass
class TablaInfectada:
    tabla: str                  # nombre de la tabla hija afectada
    columna_clave: str          # columna por la que se encontró el contagio en esta tabla
    indices_infectados: list    # índices de fila afectados en esta tabla
    origen: str                 # de qué tabla vino el contagio (padre inmediato)
    salto: int                  # distancia en pasos desde la tabla con la anomalía original


def _relaciones_utilizables(relaciones: list[RelacionSugerida]) -> list[RelacionSugerida]:
    """
    Solo relaciones con dirección conocida sirven para propagar: una
    relación 'sin_definir' no dice hacia dónde correr el contagio, así que
    se ignora acá (aunque siga dibujándose tal cual en el esquema). Las
    manuales (agregadas a mano por el usuario) siempre se usan, porque un
    humano ya confirmó la dirección.
    """
    return [
        r for r in relaciones
        if r.tabla_principal is not None and (r.manual or r.certeza_direccion in ("alta", "media"))
    ]


def _hijos_de(tabla: str, relaciones: list[RelacionSugerida]):
    """
    Para una tabla que actúa como 'principal' en alguna relación, devuelve
    por dónde se puede seguir propagando: lista de (columna_en_tabla,
    tabla_hija, columna_en_tabla_hija).
    """
    hijos = []
    for r in relaciones:
        if r.tabla_principal != tabla:
            continue
        if r.tabla_origen == tabla:
            hijos.append((r.columna_origen, r.tabla_destino, r.columna_destino))
        else:
            hijos.append((r.columna_destino, r.tabla_origen, r.columna_origen))
    return hijos


def columna_clave_de_tabla(tabla: str, relaciones: list[RelacionSugerida]) -> str | None:
    """
    Deduce la columna que una tabla usa como llave "hacia afuera" (su PK
    lógica) mirando en cuántas relaciones confirmadas aparece como
    tabla_principal. Si aparece con más de una columna distinta (caso raro
    -- ver simplificación al inicio del archivo), se queda con la más
    frecuente. Devuelve None si la tabla nunca es principal de nada (no
    tiene tablas hijas, así que no puede contagiar hacia ninguna parte).
    """
    conteo: dict[str, int] = {}
    for r in relaciones:
        if r.tabla_principal == tabla and r.columna_principal:
            conteo[r.columna_principal] = conteo.get(r.columna_principal, 0) + 1
    if not conteo:
        return None
    return max(conteo, key=conteo.get)


def calcular_indice_contagio(
    tabla_origen: str,
    indices_afectados: list,
    columna_clave: str,
    tablas: dict[str, pd.DataFrame],
    relaciones: list[RelacionSugerida],
    max_saltos: int = 5,
) -> list[TablaInfectada]:
    """
    Punto de entrada del motor.

    Parámetros
    ----------
    tabla_origen : nombre de la tabla donde se detectó la anomalía (ej. "Productos")
    indices_afectados : índices de fila donde ocurrió la anomalía en
        tabla_origen (normalmente el 'indices_atipicos' que ya entrega
        SemanticAnomalyDetector)
    columna_clave : la columna de tabla_origen que sirve de llave hacia las
        tablas hijas -- normalmente su PK (ej. "producto_id"), NO
        necesariamente la columna donde ocurrió la anomalía (que podría
        ser "precio_unitario")
    tablas : diccionario {nombre: DataFrame} de TODAS las tablas cargadas
        (self.tablas en main_window.py)
    relaciones : relaciones YA CONFIRMADAS por el usuario
        (self.relaciones_ontologia)
    max_saltos : límite de profundidad, solo como salvaguarda ante un
        esquema con ciclos raros

    Devuelve
    --------
    Lista de TablaInfectada, una entrada por cada (tabla, padre inmediato)
    con al menos una fila infectada, en el orden en que se fue encontrando
    (recorrido por niveles / BFS: primero los hijos directos, después los
    nietos, etc.).
    """
    df_origen = tablas.get(tabla_origen)
    if df_origen is None or columna_clave not in df_origen.columns:
        return []

    indices_validos = [i for i in indices_afectados if i in df_origen.index]
    if not indices_validos:
        return []

    valores_clave = set(df_origen.loc[indices_validos, columna_clave].dropna().tolist())
    if not valores_clave:
        return []

    utilizables = _relaciones_utilizables(relaciones)

    resultado: list[TablaInfectada] = []
    frontera = [(tabla_origen, valores_clave)]
    visitados_por_tabla: dict[str, set] = {tabla_origen: set(valores_clave)}
    salto = 0

    while frontera and salto < max_saltos:
        siguiente_frontera = []
        for tabla_actual, valores_actuales in frontera:
            for col_padre, tabla_hija, col_hija in _hijos_de(tabla_actual, utilizables):
                df_hija = tablas.get(tabla_hija)
                if df_hija is None or col_hija not in df_hija.columns:
                    continue

                mascara = df_hija[col_hija].isin(valores_actuales)
                indices_hija = list(df_hija.index[mascara])
                if not indices_hija:
                    continue

                resultado.append(TablaInfectada(
                    tabla=tabla_hija,
                    columna_clave=col_hija,
                    indices_infectados=indices_hija,
                    origen=tabla_actual,
                    salto=salto + 1,
                ))

                # Si esta tabla hija es a su vez "principal" de otra relación
                # más abajo, seguimos propagando con SU columna clave.
                valores_nuevos = set()
                for r2 in utilizables:
                    if r2.tabla_principal != tabla_hija or not r2.columna_principal:
                        continue
                    if r2.columna_principal in df_hija.columns:
                        valores_nuevos |= set(
                            df_hija.loc[indices_hija, r2.columna_principal].dropna().tolist()
                        )

                ya_visto = visitados_por_tabla.get(tabla_hija, set())
                valores_nuevos -= ya_visto
                if valores_nuevos:
                    visitados_por_tabla[tabla_hija] = ya_visto | valores_nuevos
                    siguiente_frontera.append((tabla_hija, valores_nuevos))

        frontera = siguiente_frontera
        salto += 1

    return resultado


def _padres_de(tabla: str, relaciones: list[RelacionSugerida]):
    """
    Complemento de _hijos_de: para una tabla que NO es la 'principal' en
    una relación (o sea, es el lado 'muchos' que referencia a otra por una
    columna tipo llave foránea), devuelve por dónde subir hacia su tabla
    padre: lista de (columna_en_tabla, tabla_padre, columna_en_tabla_padre).
    """
    padres = []
    for r in relaciones:
        if r.tabla_principal is None or r.tabla_principal == tabla:
            continue
        if r.tabla_origen == tabla and r.tabla_destino == r.tabla_principal:
            padres.append((r.columna_origen, r.tabla_destino, r.columna_destino))
        elif r.tabla_destino == tabla and r.tabla_origen == r.tabla_principal:
            padres.append((r.columna_destino, r.tabla_origen, r.columna_origen))
    return padres


@dataclass
class NodoLinaje:
    """Un nodo del árbol que arma explorar_linaje(). A propósito es un
    ÁRBOL (no un grafo libre de nodos y líneas): la gran mayoría de los
    casos reales no tienen ciclos, y una lista anidada con sangría se lee
    de arriba hacia abajo sin ambigüedad de layout -- a diferencia de un
    grafo dibujado con líneas cruzándose."""
    tabla: str
    columna_clave: str
    valor_clave: object
    indice_fila: object              # índice real en tablas[tabla], o None si no se encontró la fila
    es_anomalia: bool
    relacion_con_padre: str | None   # "hijo" o "padre" (respecto al nodo de arriba en el árbol), None en la raíz
    hijos: list = None
    truncado: bool = False           # True si había más filas relacionadas de las que se muestran acá

    def __post_init__(self):
        if self.hijos is None:
            self.hijos = []


def explorar_linaje(
    tabla_inicial: str,
    valor_id_inicial,
    columna_id_inicial: str,
    tablas: dict[str, pd.DataFrame],
    relaciones: list[RelacionSugerida],
    filas_anomalas: dict | None = None,
    max_saltos: int = 2,
    max_ramas_por_nodo: int = 8,
    max_nodos_total: int = 60,
) -> "NodoLinaje | None":
    """
    Motor de la sub-pestaña "Linaje" de Narrativa. A diferencia de
    calcular_indice_contagio (que solo mira hacia tablas HIJAS, para medir
    impacto en cascada de una anomalía puntual), esto camina en AMBOS
    sentidos -- también hacia tablas padre -- porque el objetivo acá es
    otro: partiendo de un ID (o de una fila anómala puntual), mostrar todo
    lo que está conectado con él, no solo lo que depende de él.

    Se corta con max_saltos (profundidad del árbol), max_ramas_por_nodo
    (cuántas filas relacionadas se muestran como máximo desde un mismo
    nodo -- para no listar 10.000 pedidos de un cliente) y max_nodos_total
    (tope duro de todo el árbol junto). Cuando se corta algo, el nodo
    afectado queda con truncado=True para que la interfaz pueda avisar
    "hay más de las que se muestran".

    filas_anomalas: {nombre_tabla: set(índices anómalos en esa tabla)} --
    típicamente self._anomalias_por_tabla en main_window.py. Si se omite,
    ningún nodo se marca como anómalo (el árbol igual se arma).

    Devuelve None si tabla_inicial/columna_id_inicial no existen.
    """
    df_inicial = tablas.get(tabla_inicial)
    if df_inicial is None or columna_id_inicial not in df_inicial.columns:
        return None

    filas_anomalas = filas_anomalas or {}

    def es_anomala(tabla, indice):
        return indice is not None and indice in filas_anomalas.get(tabla, set())

    coincidencias = df_inicial.index[df_inicial[columna_id_inicial] == valor_id_inicial]
    indice_inicial = coincidencias[0] if len(coincidencias) else None

    utilizables = _relaciones_utilizables(relaciones)
    contador_nodos = [1]  # ya cuenta la raíz

    raiz = NodoLinaje(
        tabla=tabla_inicial, columna_clave=columna_id_inicial, valor_clave=valor_id_inicial,
        indice_fila=indice_inicial, es_anomalia=es_anomala(tabla_inicial, indice_inicial),
        relacion_con_padre=None,
    )
    if indice_inicial is None:
        return raiz  # el ID no existe en la tabla -- árbol de un solo nodo, sin hijos

    def expandir(nodo: NodoLinaje, visitados: set, salto: int):
        if salto >= max_saltos or contador_nodos[0] >= max_nodos_total:
            return
        df_actual = tablas.get(nodo.tabla)
        if df_actual is None:
            return

        conexiones = (
            [(c, t, ch, "hijo") for c, t, ch in _hijos_de(nodo.tabla, utilizables)]
            + [(c, t, ch, "padre") for c, t, ch in _padres_de(nodo.tabla, utilizables)]
        )

        for col_propia, tabla_vecina, col_vecina, tipo in conexiones:
            if col_propia not in df_actual.columns:
                continue
            valor_propio = df_actual.at[nodo.indice_fila, col_propia]
            if pd.isna(valor_propio):
                continue
            df_vecina = tablas.get(tabla_vecina)
            if df_vecina is None or col_vecina not in df_vecina.columns:
                continue

            # No volver por donde se vino (evita el rebote padre->hijo->el mismo padre).
            indices_vecinos = [
                i for i in df_vecina.index[df_vecina[col_vecina] == valor_propio]
                if (tabla_vecina, i) not in visitados
            ]
            if not indices_vecinos:
                continue

            truncado_aqui = len(indices_vecinos) > max_ramas_por_nodo
            for indice_vecino in indices_vecinos[:max_ramas_por_nodo]:
                if contador_nodos[0] >= max_nodos_total:
                    truncado_aqui = True
                    break
                hijo = NodoLinaje(
                    tabla=tabla_vecina, columna_clave=col_vecina,
                    valor_clave=df_vecina.at[indice_vecino, col_vecina],
                    indice_fila=indice_vecino,
                    es_anomalia=es_anomala(tabla_vecina, indice_vecino),
                    relacion_con_padre=tipo,
                )
                contador_nodos[0] += 1
                nodo.hijos.append(hijo)
                visitados.add((tabla_vecina, indice_vecino))
                expandir(hijo, visitados, salto + 1)
            if truncado_aqui:
                nodo.truncado = True

    expandir(raiz, {(tabla_inicial, indice_inicial)}, 0)
    return raiz


def resumen_en_texto(resultado: list[TablaInfectada]) -> str:
    """
    Frase lista para mostrar en el informe. En vez de solo contar filas,
    apunta a la consecuencia práctica: qué se resuelve si se corrige la
    fuente del error. Agrupa por tabla (una tabla alcanzada por más de un
    camino no debería contarse dos veces la misma fila).
    """
    if not resultado:
        return (
            "Esta anomalía no se propaga: ninguna fila de las tablas conectadas "
            "depende de este registro."
        )

    por_tabla: dict[str, set] = {}
    for t in resultado:
        por_tabla.setdefault(t.tabla, set()).update(t.indices_infectados)

    total_filas = sum(len(idxs) for idxs in por_tabla.values())
    tablas_afectadas = sorted(por_tabla.keys())

    if len(tablas_afectadas) == 1:
        tabla = tablas_afectadas[0]
        n = len(por_tabla[tabla])
        plural = "registro" if n == 1 else "registros"
        return (
            f"Corregir este registro resolvería el mismo error en {n} {plural} "
            f"de {tabla}, que hoy dependen de él."
        )

    detalle = ", ".join(f"{len(idxs)} en {tabla}" for tabla, idxs in sorted(por_tabla.items()))
    plural_total = "registro" if total_filas == 1 else "registros"
    return (
        f"Corregir este registro resolvería el mismo error en {total_filas:,} {plural_total} "
        f"repartidos en {len(tablas_afectadas)} tablas: {detalle}."
    )


# ----------------------------------------------------------------------
# Demo / auto-test rápido, con una estructura parecida a la real del
# usuario (Categorias -> Productos -> Detalle_Ventas <- Ventas <- Clientes)
# ----------------------------------------------------------------------

def _demo():
    from .ontologia import inferir_relaciones

    categorias = pd.DataFrame({
        "categoria_id": [1, 2, 3],
        "nombre_categoria": ["Electro", "Cocina", "Ropa"],
    })
    productos = pd.DataFrame({
        "producto_id": [100, 101, 102, 103],
        "categoria_id": [1, 1, 2, 3],
        "nombre_producto": ["Teclado", "Mouse", "Olla", "Polera"],
        "precio_unitario": [15990, 8990, 999999, 7990],  # 999999 = anomalía
    })
    clientes = pd.DataFrame({
        "cliente_id": [1, 2, 3],
        "nombre": ["Ana", "Beto", "Carla"],
    })
    ventas = pd.DataFrame({
        "venta_id": [1000, 1001, 1002, 1003, 1004],
        "cliente_id": [1, 1, 2, 3, 3],
    })
    detalle_ventas = pd.DataFrame({
        "detalle_id": range(1, 9),
        "venta_id": [1000, 1000, 1001, 1002, 1002, 1003, 1004, 1004],
        "producto_id": [100, 102, 101, 102, 103, 102, 100, 102],
        "cantidad": [1, 2, 1, 1, 3, 1, 1, 2],
    })

    tablas = {
        "Categorias": categorias, "Productos": productos, "Clientes": clientes,
        "Ventas": ventas, "Detalle_Ventas": detalle_ventas,
    }
    relaciones = inferir_relaciones(tablas)

    print("=== Relaciones detectadas ===")
    for r in relaciones:
        print(" -", r.como_texto(), "| certeza:", r.certeza_direccion)

    # Simula la anomalía: producto_id 102 tiene un precio_unitario absurdo.
    fila_anomala = productos.index[productos["producto_id"] == 102]

    print("\n=== Índice de Contagio (anomalía de precio en Productos, producto_id=102) ===")
    resultado = calcular_indice_contagio(
        tabla_origen="Productos",
        indices_afectados=list(fila_anomala),
        columna_clave="producto_id",
        tablas=tablas,
        relaciones=relaciones,
    )
    for t in resultado:
        print(f" - Salto {t.salto}: {t.origen} → {t.tabla}.{t.columna_clave}, "
              f"{len(t.indices_infectados)} fila(s) infectada(s): {t.indices_infectados}")

    print("\n" + resumen_en_texto(resultado))


if __name__ == "__main__":
    _demo()