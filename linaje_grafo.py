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

Este módulo NO usa PySide6 (se puede probar suelto); dibujarlo es trabajo de
linaje_ui.py.
"""
from dataclasses import dataclass, field


@dataclass
class NodoCajaLinaje:
    id: str
    tabla: str
    columna: str
    valor: str            # ya convertido a texto (ver _texto_valor)
    es_anomalia: bool
    truncado: bool         # había más filas relacionadas de las que se muestran (ver explorar_linaje)
    es_raiz: bool
    capa: int = 0          # columna del diagrama; 0 = donde se hizo la búsqueda
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


def _texto_valor(valor):
    """1234.0 -> '1234' (un ID numérico no debería verse con '.0'); el resto,
    tal cual como texto."""
    if isinstance(valor, float) and valor == int(valor):
        return str(int(valor))
    return str(valor)


def construir_grafo_linaje(raiz):
    """raiz: NodoLinaje devuelto por contagio.explorar_linaje(). None, o sin
    la fila encontrada (indice_fila is None), da un grafo vacío."""
    grafo = GrafoLinaje()
    if raiz is None or raiz.indice_fila is None:
        return grafo

    capas = {}

    def asignar_capas(nodo, nid, capa_nodo):
        capas[nid] = capa_nodo
        for i, hijo in enumerate(nodo.hijos):
            # "padre": el nuevo nodo es la tabla referenciada -> queda a la izquierda
            # (más cerca del origen de los datos). "hijo": referencia a nodo -> a la derecha.
            delta = -1 if hijo.relacion_con_padre == "padre" else 1
            asignar_capas(hijo, f"{nid}.{i}", capa_nodo + delta)

    asignar_capas(raiz, "r", 0)
    ocupadas = {}

    def colocar(nodo, nid, es_raiz, fila_preferida):
        capa = capas[nid]
        fila = fila_preferida
        while (capa, fila) in ocupadas:
            fila += 1
        ocupadas[(capa, fila)] = True
        grafo.nodos.append(NodoCajaLinaje(
            id=nid, tabla=nodo.tabla, columna=nodo.columna_clave,
            valor=_texto_valor(nodo.valor_clave), es_anomalia=nodo.es_anomalia,
            truncado=nodo.truncado, es_raiz=es_raiz, capa=capa, fila=fila,
        ))
        for i, hijo in enumerate(nodo.hijos):
            hid = f"{nid}.{i}"
            colocar(hijo, hid, False, fila)
            desde, hasta = (hid, nid) if capas[hid] < capa else (nid, hid)
            grafo.aristas.append(AristaLinaje(desde, hasta))

    colocar(raiz, "r", True, 0)
    return grafo
