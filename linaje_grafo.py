"""
Convierte el árbol que arma contagio.explorar_linaje() en cajas conectadas, en
vez de una lista con sangría: la misma información (qué registros de otras
tablas están conectados con el que se buscó, siguiendo las relaciones del
esquema, hacia tablas padre e hijas) dibujada al estilo Palantir.

Las flechas SIEMPRE van de la tabla "padre" (la referenciada, la del lado
"uno") hacia la tabla "hija" (la que la referencia, el lado "muchos"), sin
importar en qué sentido se recorrió el árbol para llegar a ella -- así el
diagrama se lee de izquierda a derecha con un solo sentido, aunque
explorar_linaje() haya caminado en ambas direcciones.

Dos cosas más, pensadas para alguien que no es analista de datos:
- Si una fila tiene una columna de texto que sirve de nombre (nombre,
  descripción, título...), se usa esa como subtítulo de la caja en vez del
  ID crudo ("Leche" en vez de "id_producto = 176"). El ID crudo se conserva
  siempre disponible (tooltip, panel de detalle).
- Cuando el mismo padre tiene 3 o más hijos de la misma tabla sin ninguna
  anomalía entre ellos (el caso típico: "este producto aparece en 40 filas
  de ventas"), se juntan en UNA sola caja con el conteo, en vez de repetir
  la misma caja muchas veces. Una anomalía nunca se agrupa: siempre queda
  en su propia caja, visible.

Este módulo NO usa PySide6 (se puede probar suelto); dibujarlo es trabajo de
linaje_ui.py.
"""
from dataclasses import dataclass, field

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas es una dependencia del proyecto
    pd = None

UMBRAL_AGRUPAR = 3   # 3 o más hermanos iguales (misma tabla, sin anomalías) se juntan en una caja

# Columnas que sirven de "nombre legible" de una fila, en orden de preferencia.
# Se busca por coincidencia parcial (case-insensitive) en el nombre de columna.
_PISTAS_NOMBRE = [
    "nombre", "descripcion", "descripción", "titulo", "título", "detalle",
    "razon_social", "razón_social", "producto", "cliente", "categoria",
    "categoría", "articulo", "artículo", "etiqueta",
]


@dataclass
class NodoCajaLinaje:
    id: str
    tabla: str
    columna: str
    valor: str              # texto del ID (o, si es grupo, algo como "3 registros")
    es_anomalia: bool
    truncado: bool           # había más filas relacionadas de las que se muestran (ver explorar_linaje)
    es_raiz: bool
    es_grupo: bool = False   # True: la caja junta varios registros parecidos (ver UMBRAL_AGRUPAR)
    nombre_legible: str = None   # ej. "Leche" en vez de "id_producto = 176"; None si no se encontró
    miembros: list = field(default_factory=list)   # si es_grupo: [(valor, nombre_legible_o_None), ...]
    capa: int = 0            # columna del diagrama; 0 = donde se hizo la búsqueda
    fila: int = 0


@dataclass
class AristaLinaje:
    desde: str
    hasta: str


@dataclass
class GrafoLinaje:
    nodos: list = field(default_factory=list)
    aristas: list = field(default_factory=list)

    def nodo(self, nid):
        for n in self.nodos:
            if n.id == nid:
                return n
        return None

    def raiz(self):
        for n in self.nodos:
            if n.es_raiz:
                return n
        return None

    def _recorrer(self, nid, hacia_adelante):
        visto, pendientes = [], [nid]
        while pendientes:
            actual = pendientes.pop()
            for a in self.aristas:
                origen, destino = (a.desde, a.hasta) if hacia_adelante else (a.hasta, a.desde)
                if origen == actual and destino not in visto and destino != nid:
                    visto.append(destino)
                    pendientes.append(destino)
        return visto

    def ancestros(self, nid):
        """Ids de las tablas que se referencian desde `nid` (directa o indirectamente):
        lo que lo alimenta."""
        return self._recorrer(nid, hacia_adelante=False)

    def descendientes(self, nid):
        """Ids de las tablas que referencian a `nid` (directa o indirectamente):
        lo que `nid` alimenta."""
        return self._recorrer(nid, hacia_adelante=True)

    def directos(self, nid, hacia_adelante):
        """Solo los vecinos INMEDIATOS (una flecha), no toda la cadena."""
        return [
            (a.hasta if hacia_adelante else a.desde) for a in self.aristas
            if (a.desde if hacia_adelante else a.hasta) == nid
        ]


def _texto_valor(valor):
    """1234.0 -> '1234' (un ID numérico no debería verse con '.0'); el resto,
    tal cual como texto."""
    if isinstance(valor, float) and valor == int(valor):
        return str(int(valor))
    return str(valor)


def nombre_legible_de_fila(df, indice_fila, excluir_columna=None):
    """Busca en `df.loc[indice_fila]` una columna de texto que sirva de nombre
    (ver _PISTAS_NOMBRE) y devuelve su valor como texto, o None si no hay
    ninguna columna así, la fila no existe, o el valor está vacío."""
    if pd is None or df is None or indice_fila is None or indice_fila not in df.index:
        return None
    for pista in _PISTAS_NOMBRE:
        for col in df.columns:
            if col == excluir_columna or pista not in str(col).lower():
                continue
            # Una columna de texto de verdad, no un ID numérico que por casualidad
            # contiene la pista (ej. "id_producto" contiene "producto").
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                valor = df.at[indice_fila, col]
            except Exception:
                continue
            if pd.notna(valor) and str(valor).strip():
                return str(valor).strip()
    return None


def _texto_grupo(tabla, n, columna, valor):
    return f"{n} registros con {columna} = {valor}"


def construir_grafo_linaje(raiz, tablas=None):
    """raiz: NodoLinaje devuelto por contagio.explorar_linaje(). None, o sin
    la fila encontrada (indice_fila is None), da un grafo vacío.
    tablas: dict {nombre_tabla: DataFrame} (típicamente self.tablas de
    HadarApp) -- opcional; si se pasa, se usa para mostrar un nombre legible
    en vez del ID crudo. Sin `tablas`, el diagrama sigue funcionando, solo
    que muestra "columna = valor" en cada caja.

    Las cajas se acomodan en dos pasadas:
      1) cada caja recibe su CAPA (columna) según la relación padre/hijo.
      2) cada caja recibe su FILA centrada respecto de sus propias cajas
         hijas: se recorre el árbol de abajo hacia arriba, las cajas sin
         hijos (hojas) se numeran en orden, y cada caja con hijos queda en
         el promedio de la fila de SUS hijos. Como cada rama ocupa un tramo
         de filas que no se cruza con el de las demás, el resultado siempre
         queda sin cajas superpuestas y las líneas quedan centradas -- sin
         necesitar una grilla de enteros ni revisar colisiones.
    """
    grafo = GrafoLinaje()
    if raiz is None or raiz.indice_fila is None:
        return grafo

    def df_de(tabla):
        return (tablas or {}).get(tabla)

    capas = {}

    def asignar_capas(nodo, nid, capa_nodo):
        capas[nid] = capa_nodo
        for i, hijo in enumerate(nodo.hijos):
            # "padre": el nuevo nodo es la tabla referenciada -> queda a la izquierda
            # (más cerca del origen de los datos). "hijo": referencia a nodo -> a la derecha.
            delta = -1 if hijo.relacion_con_padre == "padre" else 1
            asignar_capas(hijo, f"{nid}.{i}", capa_nodo + delta)

    asignar_capas(raiz, "r", 0)

    hijos_diagrama = {}   # nid -> [ids de las cajas hijas EN EL DIAGRAMA, tras agrupar]

    def agregar_caja(nodo, nid, es_raiz):
        grafo.nodos.append(NodoCajaLinaje(
            id=nid, tabla=nodo.tabla, columna=nodo.columna_clave,
            valor=_texto_valor(nodo.valor_clave), es_anomalia=nodo.es_anomalia,
            truncado=nodo.truncado, es_raiz=es_raiz, capa=capas[nid],
            nombre_legible=nombre_legible_de_fila(df_de(nodo.tabla), nodo.indice_fila),
        ))

    def agregar_grupo(hijos, nid):
        primero = hijos[0]
        miembros = [
            (_texto_valor(h.valor_clave), nombre_legible_de_fila(df_de(h.tabla), h.indice_fila))
            for h in hijos
        ]
        grafo.nodos.append(NodoCajaLinaje(
            id=nid, tabla=primero.tabla, columna=primero.columna_clave,
            valor=_texto_grupo(primero.tabla, len(hijos), primero.columna_clave, primero.valor_clave),
            es_anomalia=False, truncado=any(h.truncado for h in hijos), es_raiz=False,
            es_grupo=True, miembros=miembros, capa=capas[nid],
        ))

    def construir(nodo, nid, es_raiz):
        agregar_caja(nodo, nid, es_raiz)
        propios = []

        # Agrupa hermanos de la MISMA tabla y misma dirección (padre/hijo) cuando hay
        # 3 o más y ninguno tiene una anomalía -- una anomalía siempre queda visible
        # en su propia caja.
        restantes = list(nodo.hijos)
        indices_originales = {id(h): i for i, h in enumerate(nodo.hijos)}
        while restantes:
            hijo = restantes[0]
            iguales = [h for h in restantes if h.tabla == hijo.tabla
                      and h.relacion_con_padre == hijo.relacion_con_padre]
            for h in iguales:
                restantes.remove(h)
            if len(iguales) >= UMBRAL_AGRUPAR and not any(h.es_anomalia for h in iguales):
                gid = f"{nid}.g{indices_originales[id(iguales[0])]}"
                capas[gid] = capas[f"{nid}.{indices_originales[id(iguales[0])]}"]
                agregar_grupo(iguales, gid)
                desde, hasta = (gid, nid) if capas[gid] < capas[nid] else (nid, gid)
                grafo.aristas.append(AristaLinaje(desde, hasta))
                propios.append(gid)
            else:
                for h in iguales:
                    i = indices_originales[id(h)]
                    hid = f"{nid}.{i}"
                    construir(h, hid, False)
                    desde, hasta = (hid, nid) if capas[hid] < capas[nid] else (nid, hid)
                    grafo.aristas.append(AristaLinaje(desde, hasta))
                    propios.append(hid)
        if propios:
            hijos_diagrama[nid] = propios

    construir(raiz, "r", True)

    contador_hojas = [0]
    memo_filas = {}

    def fila_de(nid):
        if nid in memo_filas:
            return memo_filas[nid]
        hijos = hijos_diagrama.get(nid)
        if not hijos:
            f = contador_hojas[0]
            contador_hojas[0] += 1
        else:
            f = sum(fila_de(h) for h in hijos) / len(hijos)
        memo_filas[nid] = f
        return f

    fila_de("r")   # una sola pasada desde la raíz: cubre todo el árbol y numera cada hoja una vez
    for n in grafo.nodos:
        n.fila = memo_filas[n.id]

    # El centrado (promedio de los hijos) casi siempre deja todo sin superponerse, PERO
    # cuando el camino padre/hijo hace zigzag (una caja vuelve a la misma capa que un
    # ANCESTRO suyo, en vez de seguir alejándose), el promedio puede coincidir por
    # casualidad con la fila de una caja de otra rama en esa misma columna. Este último
    # repaso, columna por columna, corre las que quedaron demasiado cerca (o empatadas)
    # hacia abajo lo mínimo necesario -- conserva el orden que dejó el centrado, solo
    # separa lo que se habría pisado.
    por_capa = {}
    for n in grafo.nodos:
        por_capa.setdefault(n.capa, []).append(n)
    for lista in por_capa.values():
        lista.sort(key=lambda x: (x.fila, x.id))
        for i in range(1, len(lista)):
            minimo = lista[i - 1].fila + 1.0
            if lista[i].fila < minimo:
                lista[i].fila = minimo

    return grafo


def _titulo_nodo(nodo):
    if nodo.es_grupo:
        return f"{nodo.valor}"
    if nodo.nombre_legible:
        return nodo.nombre_legible
    return f"{nodo.columna} = {nodo.valor}"


def resumen_en_lenguaje_simple(grafo):
    """Una frase (2-3 oraciones) en español llano que resume lo que se ve en
    el diagrama, para alguien que abre esto sin saber qué es una relación de
    esquema. Usa solo los vecinos DIRECTOS de la raíz (no toda la cadena),
    que es lo que de verdad orienta de un vistazo. '' si el grafo está vacío."""
    raiz = grafo.raiz() if grafo.nodos else None
    if raiz is None:
        return ""
    nombre_raiz = _titulo_nodo(raiz)
    frase = f"Este registro de {raiz.tabla} ({nombre_raiz})"
    frase += " tiene una anomalía detectada" if raiz.es_anomalia else " está conectado con otras tablas"
    frase += "."

    antes = [grafo.nodo(nid) for nid in grafo.directos(raiz.id, hacia_adelante=False)]
    despues = [grafo.nodo(nid) for nid in grafo.directos(raiz.id, hacia_adelante=True)]

    def _lista(nodos):
        return ", ".join(f"{n.tabla} ({_titulo_nodo(n)})" for n in nodos)

    if antes:
        frase += f" Antes de este registro: {_lista(antes)}."
    if despues:
        frase += f" Depende de él: {_lista(despues)}."
    if not antes and not despues:
        frase += " No tiene otras tablas conectadas con las relaciones definidas."
    return frase