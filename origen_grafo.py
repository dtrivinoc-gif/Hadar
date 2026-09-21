"""
Grafo de "Origen de los datos": las mismas cosas que cuenta el capítulo de
Narrativa (de dónde vino la tabla, qué se le hizo, qué columnas calculó Hadar,
qué indicadores usan qué), pero como cajas conectadas de izquierda a derecha.

Este módulo NO usa PySide6: solo arma los nodos y las conexiones a partir de la
bitácora de procedencia (procedencia.py), así se puede probar suelto. Dibujarlo
es trabajo de origen_ui.py.

Qué se dibuja (a propósito, poco):
  - Origen: el archivo/hoja o la base de la que salió la tabla.
  - Pasos sobre los datos: las limpiezas y los cambios a mano, EN ORDEN. Los
    pasos seguidos del mismo tipo se agrupan en UNA caja ("Limpieza, 5 pasos"),
    para que el flujo no se alargue; el detalle de cada paso va en el panel.
  - Columnas calculadas: cuelgan del último paso que existía cuando se
    calcularon. Si después hubo pasos más nuevos, se ve a simple vista que la
    columna quedó "atrás" (y lleva la etiqueta "Ojo" si además esos pasos
    tocaron datos de los que sale).
  - Indicadores: cuelgan del estado final de la tabla y de las columnas
    calculadas que usan.
Las columnas "de partida" (las que vienen del archivo) no se dibujan como caja:
se nombran en el detalle de la columna que las usa.

La redacción de los detalles repite a propósito la del capítulo de Narrativa
(procedencia.py y narrativa.py); si cambias una, revisa la otra.
"""
from dataclasses import dataclass, field

from .procedencia import (
    TIPO_COLUMNA_CALCULADA, TIPO_LIMPIEZA, TIPO_EDICION_MANUAL,
    FRASE_OPERACION_INDICADOR,
    evento_de_origen, evento_de_columna, resolver_columnas_base, usa_su_propio_valor_anterior,
    estado_de_actualizacion, esta_desactualizada, frase_desactualizacion,
    expresion_a_texto_amigable, fecha_legible, _nombre_de_fuente,
)

NODO_ORIGEN = "origen"
NODO_LIMPIEZA = "limpieza"
NODO_MANOS = "manos"
NODO_CALCULADA = "calculada"
NODO_INDICADOR = "indicador"


@dataclass
class NodoOrigen:
    id: str
    tipo: str                    # NODO_*
    titulo: str
    subtitulo: str
    detalle: list = field(default_factory=list)   # líneas en lenguaje llano para el panel
    ojo: bool = False            # puede estar desactualizado
    capa: int = 0                # columna del diagrama (0 = más a la izquierda)
    fila: int = 0


@dataclass
class AristaOrigen:
    desde: str
    hasta: str


@dataclass
class GrafoOrigen:
    nodos: list = field(default_factory=list)
    aristas: list = field(default_factory=list)
    # Relaciones "Ojo": un paso (limpieza / cambios a mano) que modificó datos de los que
    # salía una columna calculada DESPUÉS de calcularla. No se dibujan como flecha (irían
    # hacia atrás), pero cuentan para "a qué puede afectar" y para el resaltado.
    aristas_ojo: list = field(default_factory=list)

    def nodo(self, nid):
        for n in self.nodos:
            if n.id == nid:
                return n
        return None

    def _recorrer(self, nid, hacia_adelante, incluir_ojo):
        conexiones = list(self.aristas) + (list(self.aristas_ojo) if incluir_ojo else [])
        visto, pendientes = [], [nid]
        while pendientes:
            actual = pendientes.pop()
            for a in conexiones:
                origen, destino = (a.desde, a.hasta) if hacia_adelante else (a.hasta, a.desde)
                if origen == actual and destino not in visto and destino != nid:
                    visto.append(destino)
                    pendientes.append(destino)
        return visto

    def ancestros(self, nid):
        """Ids de todo lo que alimenta (directa o indirectamente) a `nid`, siguiendo solo
        las flechas dibujadas: de dónde viene."""
        return self._recorrer(nid, hacia_adelante=False, incluir_ojo=False)

    def descendientes(self, nid):
        """Ids de todo lo que `nid` alimenta siguiendo solo las flechas dibujadas."""
        return self._recorrer(nid, hacia_adelante=True, incluir_ojo=False)

    def alcance_hacia_adelante(self, nid):
        """A qué puede afectar `nid`: las flechas dibujadas MÁS las relaciones 'Ojo'
        (un cambio a mano en monto no alimenta a Doble, pero la deja desactualizada)."""
        return self._recorrer(nid, hacia_adelante=True, incluir_ojo=True)

    def afectada_por(self, nid):
        """Pasos que dejaron con 'Ojo' a `nid` (o a algo de lo que sale): los que
        cambiaron datos DESPUÉS de que se calculara. En orden de aparición."""
        base = set(self.ancestros(nid)) | {nid}
        fuentes = {a.desde for a in self.aristas_ojo if a.hasta in base}
        return [n.id for n in self.nodos if n.id in fuentes]

    def cuenta_por_grupo(self):
        """Para la leyenda: {'origen': n, 'pasos': n, 'calculadas': n, 'ojo': n}."""
        c = {"origen": 0, "pasos": 0, "calculadas": 0, "ojo": 0}
        for n in self.nodos:
            if n.tipo == NODO_ORIGEN:
                c["origen"] += 1
            elif n.tipo in (NODO_LIMPIEZA, NODO_MANOS):
                c["pasos"] += 1
            else:
                c["calculadas"] += 1
            if n.ojo:
                c["ojo"] += 1
        return c


def _lista_natural(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def _detalle_origen(origen):
    if origen is None:
        return ["No hay registro de dónde vinieron los datos de esta tabla (por ejemplo, si el "
                "proyecto se guardó antes de que Hadar llevara este registro)."]
    d = origen.detalle
    lineas = [_nombre_de_fuente(d)[0].upper() + _nombre_de_fuente(d)[1:]]
    fecha = fecha_legible(origen.fecha)
    primera = fecha_legible(d.get("primera_carga"))
    if d.get("actualizada") and primera and fecha:
        lineas.append(f"Se cargó por primera vez el {primera} y se actualizó por última vez el {fecha}.")
    elif d.get("actualizada") and fecha:
        lineas.append(f"Se actualizó desde la fuente el {fecha}.")
    elif fecha:
        lineas.append(f"Cargado el {fecha}.")
    else:
        lineas.append("Fecha de carga: no registrada.")
    if d.get("filas") is not None and d.get("columnas") is not None:
        lineas.append(f"Tenía {d['filas']:,} filas y {d['columnas']} columnas cuando se leyó.")
    lineas.append("Hadar trabaja con una copia tomada en ese momento: si la fuente cambia "
                  "después, no se entera sola.")
    return lineas


def _subtitulo_origen(origen):
    if origen is None:
        return "sin registro"
    d = origen.detalle
    if d.get("origen") == "sql_server":
        return d.get("tabla_sql") or d.get("base_datos") or "SQL Server"
    nombre = d.get("nombre_archivo") or "archivo"
    hojas = ([d["hoja"]] if d.get("hoja") else []) + list(d.get("hojas_unidas") or [])
    if len(hojas) == 1:
        return f"{nombre} · {hojas[0]}"
    if len(hojas) > 1:
        return f"{nombre} · {len(hojas)} hojas"
    return nombre


def _linea_paso(e):
    fecha = fecha_legible(e.fecha)
    texto = e.descripcion
    if e.tipo == TIPO_LIMPIEZA and e.detalle.get("subtipo") == "duplicado":
        texto += (f" La tabla pasó de {e.detalle.get('filas_antes', 0):,} "
                  f"a {e.detalle.get('filas_despues', 0):,} filas.")
    return f"{fecha} · {texto}" if fecha else texto


def _detalle_calculada(e, eventos, estado):
    formula = e.detalle.get("formula_texto") or expresion_a_texto_amigable(e.detalle.get("expresion", ""))
    lineas = [f"= {formula}"]
    usa = [d for d in e.depende_de if d != e.columna]
    if usa:
        etiquetas = [f"{d} (también calculada)" if evento_de_columna(eventos, d) else d for d in usa]
        lineas.append(f"Usa: {_lista_natural(etiquetas)}.")
    elif not usa_su_propio_valor_anterior(e):
        lineas.append("No usa ninguna otra columna.")
    if usa_su_propio_valor_anterior(e):
        lineas.append("También usa el valor que tenía antes esta misma columna.")
    base = resolver_columnas_base(eventos, e.columna)
    if base and any(evento_de_columna(eventos, d) for d in usa):
        lineas.append(f"En último término sale de: {_lista_natural(base)}.")
    fecha = fecha_legible(e.fecha)
    if fecha:
        lineas.append(f"Calculada el {fecha}.")
    lineas.append("Se calculó una sola vez: no se actualiza sola si cambias los datos.")
    aviso = frase_desactualizacion(estado)
    if aviso:
        lineas.append(f"Ojo: {aviso}")
    return lineas


def _descripcion_indicador(ind):
    operacion = getattr(ind, "operacion", None)
    columna = getattr(ind, "columna", None)
    formula = getattr(ind, "formula", None)
    if operacion == "formula":
        corta = "fórmula"
        larga = f"Fórmula {expresion_a_texto_amigable(formula or '')}"
    elif columna:
        frase = FRASE_OPERACION_INDICADOR.get(operacion, "cálculo sobre")
        corta = f"{frase} {columna}"
        larga = f"{frase[0].upper() + frase[1:]} {columna}"
    else:
        corta = "sin columna"
        larga = "Cálculo sin columna definida"
    return corta, larga


def construir_grafo_origen(eventos, columnas, indicadores=None):
    """Arma el grafo de la tabla activa.
    eventos: eventos de procedencia de ESA tabla (BitacoraProcedencia.eventos_de_tabla).
    columnas: nombres de las columnas que existen hoy (los eventos de columnas que ya
    no existen se ignoran).
    indicadores: objetos tipo Indicador (opcional; se usan solo por duck typing).
    Devuelve un GrafoOrigen (vacío si no hay nada que contar)."""
    indicadores = list(indicadores or [])
    columnas = {str(c) for c in columnas}
    origen = evento_de_origen(eventos)
    calculadas = [e for e in eventos if e.tipo == TIPO_COLUMNA_CALCULADA and e.columna in columnas]
    pasos = sorted(
        (e for e in eventos
         if e.tipo == TIPO_LIMPIEZA or (e.tipo == TIPO_EDICION_MANUAL and e.columna in columnas)),
        key=lambda e: e.fecha or "",
    )
    grafo = GrafoOrigen()
    if origen is None and not pasos and not calculadas and not indicadores:
        return grafo

    # ---- cadena principal: origen -> grupos de pasos
    nodo_origen = NodoOrigen(
        id="origen", tipo=NODO_ORIGEN, titulo="Origen", subtitulo=_subtitulo_origen(origen),
        detalle=_detalle_origen(origen), capa=0, fila=0,
    )
    grafo.nodos.append(nodo_origen)
    grupos = []   # [(nodo, fecha_max)]
    ultimo_tipo = None
    for e in pasos:
        tipo = NODO_LIMPIEZA if e.tipo == TIPO_LIMPIEZA else NODO_MANOS
        if tipo != ultimo_tipo:
            grupos.append({"tipo": tipo, "eventos": []})
            ultimo_tipo = tipo
        grupos[-1]["eventos"].append(e)
    cadena = ["origen"]
    fechas_max = {"origen": ""}
    grupo_de_evento = {}   # id(evento) -> id de la caja de paso que lo contiene
    for i, g in enumerate(grupos, start=1):
        evs = g["eventos"]
        if g["tipo"] == NODO_LIMPIEZA:
            titulo = "Limpieza"
            subtitulo = "1 paso" if len(evs) == 1 else f"{len(evs)} pasos"
            detalle = [_linea_paso(e) for e in evs]
        else:
            titulo = "Cambios a mano"
            if len(evs) == 1:
                subtitulo = f"{evs[0].detalle.get('cambios', 1)} en {evs[0].columna}"
            else:
                subtitulo = f"{len(evs)} columnas"
            detalle = [_linea_paso(e) for e in evs]
        nid = f"paso{i}"
        grafo.nodos.append(NodoOrigen(id=nid, tipo=g["tipo"], titulo=titulo, subtitulo=subtitulo,
                                      detalle=detalle, capa=i, fila=0))
        grafo.aristas.append(AristaOrigen(cadena[-1], nid))
        cadena.append(nid)
        fechas_max[nid] = max((e.fecha or "") for e in evs)
        for e in evs:
            grupo_de_evento[id(e)] = nid

    def ancla_para(fecha):
        """Último paso que ya existía cuando se calculó algo en `fecha`."""
        ancla = "origen"
        if not fecha:
            return cadena[-1]
        for nid in cadena[1:]:
            if fechas_max[nid] and fechas_max[nid] <= fecha:
                ancla = nid
            else:
                break
        return ancla

    # ---- columnas calculadas e indicadores: padres
    padres = {}          # id -> [ids de los que salen]
    nuevos = []          # nodos no-cadena, en orden de creación
    nombres_calc = {e.columna for e in calculadas}
    for e in calculadas:
        nid = f"col:{e.columna}"
        estado = estado_de_actualizacion(eventos, e.columna)
        usa = [d for d in e.depende_de if d != e.columna]
        padres_calc = [f"col:{d}" for d in usa if d in nombres_calc]
        padres[nid] = padres_calc or [ancla_para(e.fecha)]
        etiquetas_sub = ", ".join(usa) if usa else "sin columnas"
        nuevos.append(NodoOrigen(
            id=nid, tipo=NODO_CALCULADA, titulo=str(e.columna),
            subtitulo=f"de {etiquetas_sub}" if usa else "sin otras columnas",
            detalle=_detalle_calculada(e, eventos, estado), ojo=esta_desactualizada(estado),
        ))
    estados_calc = {e.columna: estado_de_actualizacion(eventos, e.columna) for e in calculadas}
    for e in calculadas:
        for paso in sorted({grupo_de_evento[id(m)] for m in estados_calc[e.columna]["modificaciones"]
                            if id(m) in grupo_de_evento}):
            grafo.aristas_ojo.append(AristaOrigen(paso, f"col:{e.columna}"))
    for i, ind in enumerate(indicadores):
        nid = f"ind:{i}"
        try:
            deps = sorted(ind.columnas_de_las_que_depende())
        except Exception:
            deps = []
        deps_calc = [d for d in deps if d in nombres_calc]
        padres[nid] = [f"col:{d}" for d in deps_calc] or [cadena[-1]]
        corta, larga = _descripcion_indicador(ind)
        detalle = [f"Indicador: {larga}."]
        if deps:
            detalle.append(f"Usa: {_lista_natural(deps)}.")
        con_ojo = [d for d in deps_calc if esta_desactualizada(estados_calc[d])]
        if deps_calc:
            base_txt = []
            for d in deps_calc:
                base = resolver_columnas_base(eventos, d)
                base_txt.append(f"{d} (sale de {_lista_natural(base)})" if base else d)
            detalle.append(f"Usa columnas calculadas: {_lista_natural(base_txt)}.")
        if con_ojo:
            detalle.append(
                f"Ojo: usa columnas calculadas que pueden estar desactualizadas: {_lista_natural(con_ojo)}."
            )
        detalle.append("Un indicador se calcula sobre los datos actuales cada vez que lo miras.")
        nuevos.append(NodoOrigen(
            id=nid, tipo=NODO_INDICADOR, titulo=str(getattr(ind, "nombre", "Indicador")),
            subtitulo=corta, detalle=detalle, ojo=bool(con_ojo),
        ))

    # ---- capas (columna del diagrama), con protección contra ciclos
    capa_de = {nid: i for i, nid in enumerate(cadena)}
    por_id = {n.id: n for n in nuevos}

    def capa(nid, ruta):
        if nid in capa_de:
            return capa_de[nid]
        # un padre que ya está en la ruta cerraría un ciclo (A usa B y B usa A): se descarta
        padres[nid] = [p for p in padres[nid] if p not in ruta and (p in capa_de or p in por_id)]
        if not padres[nid]:
            padres[nid] = [cadena[-1]]
        capa_de[nid] = 1 + max(capa(p, ruta | {nid}) for p in padres[nid])
        return capa_de[nid]

    for n in nuevos:
        capa(n.id, frozenset())
        n.capa = capa_de[n.id]

    # ---- filas: la cadena va en la fila 0; el resto busca su lugar (mejor, la fila de su padre)
    ocupadas = {(n.capa, 0) for n in grafo.nodos}
    fila_de = {n.id: 0 for n in grafo.nodos}
    for n in sorted(nuevos, key=lambda x: x.capa):
        preferida = fila_de.get(padres[n.id][0], 0)
        r = preferida if (n.capa, preferida) not in ocupadas else 0
        while (n.capa, r) in ocupadas:
            r += 1
        n.fila = r
        ocupadas.add((n.capa, r))
        fila_de[n.id] = r
    grafo.nodos.extend(nuevos)
    for n in nuevos:
        for p in padres[n.id]:
            grafo.aristas.append(AristaOrigen(p, n.id))
    return grafo
