"""
Procedencia de los datos: de dónde salió cada columna y qué le pasó antes
de llegar a lo que el usuario ve hoy.

OJO con el nombre: esto NO es lo mismo que la sub-pestaña "Linaje" de
Narrativa (contagio.py: explorar_linaje), que parte de UN registro y muestra
con qué registros de otras tablas se conecta. Acá es la historia de una
COLUMNA: si la calculó Hadar, con qué fórmula y a partir de qué otras
columnas. En la interfaz se llama "Origen de los datos".

Este módulo NO usa PySide6 (se puede probar suelto). Es una bitácora simple:
una lista de eventos que los módulos que transforman datos van agregando a
medida que actúan, en vez de intentar reconstruir la historia después.

Etapa 1: columnas calculadas ("Excel: Calcular y Arrastrar").
Etapa 2: de dónde vino la tabla (archivo, hoja, o base SQL Server) y cuándo
se cargó.
Etapa 3: qué se le hizo a los datos dentro de Hadar (limpieza sugerida y
cambios a mano en Datos), y el aviso de columnas calculadas que pudieron
quedar desactualizadas porque después se modificaron los datos de los que salen.

Convención: en los eventos de limpieza y de edición a mano, `depende_de` son
las columnas cuyos VALORES se modificaron.
"""
import os
import re
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime

from .formulas import extraer_columnas_formula, FormulaSeguridadError

TIPO_COLUMNA_CALCULADA = "columna_calculada"
TIPO_ARCHIVO_ORIGEN = "archivo_origen"
TIPO_LIMPIEZA = "limpieza"
TIPO_EDICION_MANUAL = "edicion_manual"
# Los eventos de estos tipos cambian VALORES de columnas ya existentes (a
# diferencia de eliminar filas duplicadas, que no altera lo que dice cada fila).
TIPOS_QUE_MODIFICAN_VALORES = (TIPO_LIMPIEZA, TIPO_EDICION_MANUAL)

# Valores de detalle["origen"] en un evento de TIPO_ARCHIVO_ORIGEN.
ORIGEN_ARCHIVO = "archivo"        # csv / xlsx / xls / parquet / .sql
ORIGEN_SQL_SERVER = "sql_server"  # conexión en vivo


@dataclass
class EventoProcedencia:
    tipo: str                    # TIPO_COLUMNA_CALCULADA | (futuro) archivo_origen | limpieza
    tabla: str                   # tabla a la que pertenece el resultado
    columna: str | None          # columna resultado (None si el evento es de toda la tabla)
    depende_de: list             # columnas de las que sale (de la misma tabla)
    descripcion: str             # frase en español, lista para mostrar
    fecha: str                   # ISO 8601, ej. "2026-09-20T14:32:05"
    detalle: dict = field(default_factory=dict)   # datos extra según el tipo


def _ahora_iso():
    return datetime.now().isoformat(timespec="seconds")


def fecha_legible(fecha_iso):
    """'2026-09-20T14:32:05' -> '20-09-2026 14:32' (mismo formato día-mes-año
    que ya usa Narrativa). Si la fecha viene rota, devuelve '' en vez de fallar."""
    try:
        return datetime.fromisoformat(fecha_iso).strftime("%d-%m-%Y %H:%M")
    except (TypeError, ValueError):
        return ""


def _descripcion_columna_calculada(columna, formula_texto):
    return f"«{columna}» se calculó con la fórmula {formula_texto}."


# Cómo se dice, en lenguaje llano, la operación de un Indicador simple. Vive acá (y no
# en indicadores.py) para que el informe y el diagrama de Origen puedan describir un
# indicador sin importar la clase Indicador ni PySide6: solo necesitan objetos con
# .nombre, .operacion, .columna, .formula y .columnas_de_las_que_depende().
FRASE_OPERACION_INDICADOR = {
    "suma": "suma de",
    "promedio": "promedio de",
    "mediana": "mediana de",
    "minimo": "valor mínimo de",
    "maximo": "valor máximo de",
    "conteo": "cantidad de datos en",
    "conteo_unico": "cantidad de valores distintos en",
}


def _n_celdas(n):
    return "1 celda" if n == 1 else f"{n:,} celdas"


def _columnas_con_conteo(celdas_por_columna, tope=5):
    """{'nombre': 8, 'ciudad': 4} -> 'nombre: 8, ciudad: 4' (las de más
    cambios primero; si son muchas, 'y N columnas más')."""
    orden = sorted(celdas_por_columna.items(), key=lambda kv: (-kv[1], kv[0]))
    txt = ", ".join(f"{c}: {n:,}" for c, n in orden[:tope])
    if len(orden) > tope:
        txt += f" y {len(orden) - tope} columnas más"
    return txt


_FRASE_LIMPIEZA = {
    "espacio_en_blanco": "Se recortaron los espacios de más en {celdas}",
    "formato_inconsistente": "Se unificó el formato (mayúsculas, minúsculas y variantes) en {celdas}",
    "caracter_especial": "Se quitaron caracteres especiales en {celdas}",
    "numero_en_texto": "Se pasaron a número los valores de {celdas} que estaban escritos como texto",
}


def _descripcion_limpieza(subtipo, detalle):
    if subtipo == "duplicado":
        n = detalle.get("filas_eliminadas", 0)
        if n == 1:
            return "Se eliminó 1 fila duplicada (se conservó la primera de cada grupo)."
        return f"Se eliminaron {n:,} filas duplicadas (se conservó la primera de cada grupo)."
    celdas = detalle.get("celdas_por_columna") or {}
    base = _FRASE_LIMPIEZA.get(subtipo, "Se corrigieron datos en {celdas}").format(
        celdas=_n_celdas(detalle.get("celdas_total", 0))
    )
    if subtipo == "caracter_especial" and detalle.get("caracteres"):
        base += " (" + " ".join(detalle["caracteres"]) + ")"
    if celdas:
        base += f" — {_columnas_con_conteo(celdas)}"
    return base + "."


def _descripcion_edicion_manual(columna, detalle):
    n = detalle.get("cambios", 0)
    veces = "1 cambio" if n == 1 else f"{n:,} cambios"
    return f"Se hicieron {veces} a mano en «{columna}»."


def _iguales(a, b):
    try:
        if a is None and b is None:
            return True
        if a != a and b != b:   # ambos NaN
            return True
        return bool(a == b)
    except Exception:
        return str(a) == str(b)


def resumir_cambios(antes, despues):
    """Compara una tabla ANTES y DESPUÉS de una corrección y devuelve lo que
    realmente cambió (no lo que se pensaba cambiar): filas antes/después y,
    por columna, cuántas celdas quedaron con otro valor. Solo compara las
    filas que siguen existiendo. Nunca lanza: si algo no se puede comparar,
    esa columna simplemente no aporta conteo."""
    resumen = {
        "filas_antes": len(antes), "filas_despues": len(despues),
        "celdas_por_columna": {},
    }
    try:
        if not (antes.index.is_unique and despues.index.is_unique
                and antes.columns.is_unique and despues.columns.is_unique
                and despues.index.isin(antes.index).all()):
            return resumen
        comunes = [c for c in antes.columns if c in despues.columns]
        a = antes.loc[despues.index, comunes]
        b = despues[comunes]
        for c in comunes:
            sa, sb = a[c], b[c]
            try:
                distintas = ~((sa == sb) | (sa.isna() & sb.isna()))
            except Exception:
                sa2, sb2 = sa.astype(str), sb.astype(str)
                distintas = sa2 != sb2
            n = int(distintas.sum())
            if n:
                resumen["celdas_por_columna"][str(c)] = n
    except Exception:
        pass
    return resumen


def _nombre_de_fuente(detalle):
    """Frase corta de la fuente, ej. 'ventas.xlsx (hoja «Enero»)' o
    'la base «Ventas» del servidor «SRV01» (tabla «Pedidos»)'."""
    if detalle.get("origen") == ORIGEN_SQL_SERVER:
        base = f"la base «{detalle.get('base_datos', '?')}»"
        if detalle.get("servidor"):
            base += f" del servidor «{detalle['servidor']}»"
        if detalle.get("tabla_sql"):
            base += f" (tabla «{detalle['tabla_sql']}»)"
        return base
    nombre = detalle.get("nombre_archivo") or "un archivo"
    hojas = ([detalle["hoja"]] if detalle.get("hoja") else []) + list(detalle.get("hojas_unidas") or [])
    if len(hojas) == 1:
        return f"{nombre} (hoja «{hojas[0]}»)"
    if len(hojas) > 1:
        return f"{nombre} (hojas " + ", ".join(f"«{h}»" for h in hojas) + ")"
    if detalle.get("tabla_sql"):
        return f"{nombre} (tabla «{detalle['tabla_sql']}»)"
    return nombre


def expresion_a_texto_amigable(expresion):
    """Convierte una expresión interna (df['Ventas'] - df['Costos']) a como la
    escribiría el usuario ([Ventas] - [Costos]). Solo para mostrar."""
    if not expresion:
        return ""
    return re.sub(
        r"df\[(['\"])((?:(?!\1).)*)\1\]",
        lambda m: f"[{m.group(2)}]",
        expresion,
    )


# ----------------------------------------------------------------------------
# Funciones sueltas sobre la lista de eventos de UNA tabla (las usa también
# narrativa.py, que solo recibe los eventos de la tabla activa).
# ----------------------------------------------------------------------------
def evento_de_columna(eventos, columna, tipo=TIPO_COLUMNA_CALCULADA):
    for e in eventos:
        if e.tipo == tipo and e.columna == columna:
            return e
    return None


def resolver_columnas_base(eventos, columna):
    """Las columnas 'de partida' de las que sale, en último término, una
    columna calculada: sigue la cadena (Margen % -> Margen -> Ventas, Costos)
    hasta llegar a columnas que Hadar NO calculó. Devuelve una lista sin
    repetidos, en orden de aparición. Se protege de ciclos (A usa B y B usa A,
    posible si se reemplaza una columna ya usada) y de una columna que usa su
    propio valor anterior (Total = [Total] * 1.19): en ambos casos ese tramo
    se ignora en vez de quedar dando vueltas."""
    base = []

    def _recorrer(col, ruta):
        ev = evento_de_columna(eventos, col)
        if ev is None:
            return
        for dep in ev.depende_de:
            if dep == col or dep in ruta:
                continue
            if evento_de_columna(eventos, dep) is None:
                if dep not in base:
                    base.append(dep)
            else:
                _recorrer(dep, ruta | {dep})

    _recorrer(columna, {columna})
    return base


def evento_de_origen(eventos):
    """El evento de origen (archivo / base) de la tabla, o None si no se
    registró ninguno."""
    for e in eventos:
        if e.tipo == TIPO_ARCHIVO_ORIGEN:
            return e
    return None


def usa_su_propio_valor_anterior(evento):
    return evento.columna is not None and evento.columna in evento.depende_de


def estado_de_actualizacion(eventos, columna, _visitando=None):
    """¿Una columna calculada pudo quedar desactualizada? Lo está si, DESPUÉS
    de calcularla, se modificaron los valores de alguna columna de la que sale
    (por limpieza o a mano), si sale de otra columna calculada que ya lo
    está, o si alguna de las columnas calculadas de las que sale se volvió a
    calcular DESPUÉS que ella. Devuelve {'modificaciones': [eventos],
    'columnas_tocadas': [...], 'arrastrada_de': [calculadas desactualizadas de
    las que sale], 'recalculadas_despues': [calculadas de las que sale y que se
    recalcularon más tarde]}. Es una advertencia, no una certeza: solo compara
    fechas de la bitácora, no recalcula nada."""
    vacio = {"modificaciones": [], "columnas_tocadas": [], "arrastrada_de": [],
             "recalculadas_despues": []}
    ev = evento_de_columna(eventos, columna)
    if ev is None or not ev.fecha:
        return vacio
    visitando = (_visitando or set()) | {columna}
    directas = [d for d in ev.depende_de if d != columna]
    modificaciones = [
        e for e in eventos
        if e.tipo in TIPOS_QUE_MODIFICAN_VALORES and e.fecha and e.fecha > ev.fecha
        and set(e.depende_de) & set(directas)
    ]
    tocadas = sorted({c for e in modificaciones for c in e.depende_de if c in directas})
    arrastrada, recalculadas = [], []
    for d in directas:
        ev_d = evento_de_columna(eventos, d)
        if d in visitando or ev_d is None:
            continue
        if ev_d.fecha and ev_d.fecha > ev.fecha:
            recalculadas.append(d)
            continue
        st = estado_de_actualizacion(eventos, d, visitando)
        if esta_desactualizada(st):
            arrastrada.append(d)
    return {"modificaciones": modificaciones, "columnas_tocadas": tocadas,
            "arrastrada_de": arrastrada, "recalculadas_despues": recalculadas}


def esta_desactualizada(estado):
    return bool(estado["modificaciones"] or estado["arrastrada_de"] or estado["recalculadas_despues"])


def frase_desactualizacion(estado, larga=True):
    """Aviso en lenguaje llano (sin emojis). '' si no hay nada que avisar."""
    if not esta_desactualizada(estado):
        return ""
    partes = []
    if estado["columnas_tocadas"]:
        partes.append(
            "Después de calcularla se modificaron datos de: "
            + ", ".join(estado["columnas_tocadas"]) + "."
        )
    if estado["recalculadas_despues"]:
        partes.append(
            "Sale de columnas calculadas que se volvieron a calcular después: "
            + ", ".join(estado["recalculadas_despues"]) + "."
        )
    if estado["arrastrada_de"]:
        partes.append(
            "Sale de columnas calculadas que ya quedaron desactualizadas: "
            + ", ".join(estado["arrastrada_de"]) + "."
        )
    partes.append("Puede estar desactualizada.")
    if larga:
        partes.append("Para ponerla al día, créala otra vez con la misma fórmula y el mismo nombre.")
    return " ".join(partes)


class BitacoraProcedencia:
    """Lista de eventos de procedencia de todas las tablas del proyecto."""

    def __init__(self, eventos=None):
        self._eventos = list(eventos or [])

    def __len__(self):
        return len(self._eventos)

    # ------------------------------------------------------------ registrar
    def registrar_columna_calculada(self, tabla, columna, formula_texto, expresion, fecha=None):
        """Anota que `columna` de `tabla` se calculó con esa fórmula.
        `formula_texto` es lo que escribió el usuario ([Ventas] × 2);
        `expresion` es la versión interna (df['Ventas'] * 2) de la que se
        leen las columnas sin ejecutar nada (formulas.extraer_columnas_formula).
        Si esa columna ya tenía un evento (se reemplazó), queda solo el nuevo."""
        try:
            depende_de = sorted(extraer_columnas_formula(expresion))
        except FormulaSeguridadError:
            depende_de = []
        # Se reemplaza el evento anterior de esa columna y también los cambios a
        # mano hechos sobre la versión anterior (esos valores ya no existen).
        self._eventos = [
            e for e in self._eventos
            if not (e.tabla == tabla and e.columna == columna
                    and e.tipo in (TIPO_COLUMNA_CALCULADA, TIPO_EDICION_MANUAL))
        ]
        evento = EventoProcedencia(
            tipo=TIPO_COLUMNA_CALCULADA,
            tabla=tabla,
            columna=columna,
            depende_de=depende_de,
            descripcion=_descripcion_columna_calculada(columna, formula_texto),
            fecha=fecha or _ahora_iso(),
            detalle={"formula_texto": formula_texto, "expresion": expresion},
        )
        self._eventos.append(evento)
        return evento

    def registrar_origen(self, tabla, origen, *, filas=None, columnas=None, fecha=None,
                         es_actualizacion=False, nombre_archivo=None, ruta=None, hoja=None,
                         tabla_sql=None, servidor=None, base_datos=None):
        """Anota de dónde salió la tabla y cuándo se leyó. `origen` es
        ORIGEN_ARCHIVO u ORIGEN_SQL_SERVER; el resto son los datos de esa
        fuente (solo se guardan los que vengan). NUNCA se guarda una
        contraseña ni el usuario de SQL Server. Reemplaza cualquier origen
        anterior de esa tabla. es_actualizacion=True: se registra al releer
        desde la fuente sin saber cuándo fue la primera carga."""
        fecha = fecha if fecha is not None else _ahora_iso()
        detalle = {"origen": origen, "filas": filas, "columnas": columnas}
        for clave, valor in (
            ("nombre_archivo", nombre_archivo), ("ruta", ruta), ("hoja", hoja),
            ("tabla_sql", tabla_sql), ("servidor", servidor), ("base_datos", base_datos),
        ):
            if valor is not None:
                detalle[clave] = valor
        if es_actualizacion:
            detalle["actualizada"] = True
        elif fecha:
            detalle["primera_carga"] = fecha
        self._eventos = [
            e for e in self._eventos
            if not (e.tipo == TIPO_ARCHIVO_ORIGEN and e.tabla == tabla)
        ]
        evento = EventoProcedencia(
            tipo=TIPO_ARCHIVO_ORIGEN, tabla=tabla, columna=None, depende_de=[],
            descripcion=f"Los datos de «{tabla}» se cargaron desde {_nombre_de_fuente(detalle)}.",
            fecha=fecha, detalle=detalle,
        )
        self._eventos.append(evento)
        return evento

    def registrar_actualizacion(self, tabla, filas=None, columnas=None, fecha=None, hoja=None):
        """La tabla se volvió a leer desde su fuente: se conserva la fuente
        y la fecha de la primera carga, y se actualiza la fecha de lectura.
        `hoja`: la hoja de Excel que realmente se leyó; solo se anota si el
        origen no tenía ninguna (ej. proyecto antiguo). Devuelve False si esa
        tabla no tenía origen registrado."""
        origen = evento_de_origen(self.eventos_de_tabla(tabla))
        if origen is None:
            return False
        origen.fecha = fecha if fecha is not None else _ahora_iso()
        if hoja and not origen.detalle.get("hoja"):
            origen.detalle["hoja"] = hoja
        origen.detalle["actualizada"] = True
        origen.detalle["filas"] = filas
        origen.detalle["columnas"] = columnas
        return True

    def registrar_hoja_unida(self, tabla, hoja, filas=None):
        """Se unió otra hoja del mismo Excel a la tabla (más filas)."""
        origen = evento_de_origen(self.eventos_de_tabla(tabla))
        if origen is None:
            return False
        unidas = origen.detalle.setdefault("hojas_unidas", [])
        if hoja not in unidas:
            unidas.append(hoja)
        origen.detalle["filas"] = filas
        return True

    def completar_origen_desde_fuentes(self, tablas, fuentes):
        """Proyectos guardados antes de llevar este registro: para cada tabla
        SIN origen anotado pero con una fuente guardada (fuentes_datos), anota
        lo que esa fuente sí dice (archivo o base y tabla), dejando la fecha
        vacía -- no se sabe cuándo fue, y no se inventa. Devuelve cuántas
        completó."""
        completadas = 0
        for tabla in tablas:
            if evento_de_origen(self.eventos_de_tabla(tabla)) is not None:
                continue
            f = (fuentes or {}).get(tabla)
            if not f:
                continue
            if f.get("tipo") == "archivo" and f.get("ruta"):
                self.registrar_origen(
                    tabla, ORIGEN_ARCHIVO, fecha="",
                    nombre_archivo=os.path.basename(f["ruta"]), ruta=f["ruta"],
                )
            elif f.get("tipo") == "sql_server":
                self.registrar_origen(
                    tabla, ORIGEN_SQL_SERVER, fecha="",
                    servidor=f.get("servidor"), base_datos=f.get("base_datos"),
                    tabla_sql=f.get("tabla"),
                )
            else:
                continue
            completadas += 1
        return completadas

    def registrar_limpieza(self, tabla, subtipo, df_antes, df_despues, caracteres=None, fecha=None):
        """Anota una corrección de la Limpieza sugerida. Lo que se guarda sale
        de comparar la tabla antes y después (cuántas filas se eliminaron,
        cuántas celdas cambiaron y en qué columnas), no de lo que se esperaba.
        Si la corrección no cambió nada, no se anota. Devuelve el evento o None."""
        resumen = resumir_cambios(df_antes, df_despues)
        eliminadas = resumen["filas_antes"] - resumen["filas_despues"]
        celdas = resumen["celdas_por_columna"]
        if eliminadas <= 0 and not celdas:
            return None
        detalle = {
            "subtipo": subtipo,
            "filas_antes": resumen["filas_antes"], "filas_despues": resumen["filas_despues"],
            "filas_eliminadas": max(eliminadas, 0),
            "celdas_por_columna": celdas, "celdas_total": sum(celdas.values()),
        }
        if caracteres:
            detalle["caracteres"] = list(caracteres)
        evento = EventoProcedencia(
            tipo=TIPO_LIMPIEZA, tabla=tabla, columna=None, depende_de=sorted(celdas),
            descripcion=_descripcion_limpieza(subtipo, detalle),
            fecha=fecha or _ahora_iso(), detalle=detalle,
        )
        self._eventos.append(evento)
        return evento

    def registrar_edicion_manual(self, tabla, columna, fecha=None):
        """Anota un cambio hecho a mano en una celda de Datos. Los cambios a
        una misma columna se acumulan en UN solo evento (con la fecha del
        último), para no llenar la bitácora con una línea por celda."""
        fecha = fecha or _ahora_iso()
        for e in self._eventos:
            if e.tipo == TIPO_EDICION_MANUAL and e.tabla == tabla and e.columna == columna:
                e.detalle["cambios"] = e.detalle.get("cambios", 0) + 1
                e.fecha = fecha
                e.descripcion = _descripcion_edicion_manual(columna, e.detalle)
                return e
        detalle = {"cambios": 1, "primera_edicion": fecha}
        evento = EventoProcedencia(
            tipo=TIPO_EDICION_MANUAL, tabla=tabla, columna=columna, depende_de=[columna],
            descripcion=_descripcion_edicion_manual(columna, detalle), fecha=fecha, detalle=detalle,
        )
        self._eventos.append(evento)
        return evento

    # ------------------------------------------------------------- consultar
    def eventos_de_tabla(self, tabla, tipo=None):
        return [
            e for e in self._eventos
            if e.tabla == tabla and (tipo is None or e.tipo == tipo)
        ]

    def dependientes_de(self, tabla, columna):
        """Columnas calculadas de `tabla` que usan `columna` directamente
        (sin contar a la propia columna). Sirve para avisar antes de
        reemplazar una columna de la que otras salieron."""
        return [
            e.columna for e in self.eventos_de_tabla(tabla, TIPO_COLUMNA_CALCULADA)
            if columna in e.depende_de and e.columna != columna
        ]

    # ---------------------------------------------------------- mantenimiento
    def renombrar_columna(self, tabla, viejo, nuevo):
        """Sigue el renombre de una columna en Datos: el resultado, las
        dependencias y el texto de la fórmula pasan al nombre nuevo."""
        patron = re.compile(r"\[" + re.escape(viejo) + r"\]")
        for e in self.eventos_de_tabla(tabla):
            if e.columna == viejo:
                e.columna = nuevo
            e.depende_de = [nuevo if d == viejo else d for d in e.depende_de]
            if e.tipo == TIPO_LIMPIEZA:
                por_col = e.detalle.get("celdas_por_columna") or {}
                if viejo in por_col:
                    e.detalle["celdas_por_columna"] = {
                        (nuevo if k == viejo else k): v for k, v in por_col.items()
                    }
                e.descripcion = _descripcion_limpieza(e.detalle.get("subtipo", ""), e.detalle)
            elif e.tipo == TIPO_EDICION_MANUAL:
                e.descripcion = _descripcion_edicion_manual(e.columna, e.detalle)
            if e.tipo == TIPO_COLUMNA_CALCULADA:
                texto = e.detalle.get("formula_texto")
                if texto:
                    e.detalle["formula_texto"] = patron.sub(lambda _m: f"[{nuevo}]", texto)
                expr = e.detalle.get("expresion")
                if expr:
                    e.detalle["expresion"] = expr.replace(f"df[{viejo!r}]", f"df[{nuevo!r}]")
                e.descripcion = _descripcion_columna_calculada(
                    e.columna, e.detalle.get("formula_texto", "")
                )

    def olvidar_tabla(self, tabla, conservar=()):
        """La tabla se volvió a cargar desde cero (otra hoja, actualizar desde
        la fuente): lo que se anotó de la versión anterior ya no aplica.
        `conservar`: tipos de evento que se mantienen (ej. el origen, cuando
        se actualiza desde la MISMA fuente)."""
        self._eventos = [
            e for e in self._eventos
            if e.tabla != tabla or e.tipo in conservar
        ]

    def podar(self, tablas):
        """Descarta eventos cuya tabla o columna resultado ya no existe.
        `tablas` es el dict {nombre: DataFrame}. Devuelve cuántos quitó."""
        antes = len(self._eventos)
        vigentes = []
        for e in self._eventos:
            df = tablas.get(e.tabla)
            if df is None:
                continue
            if e.columna is not None and e.columna not in df.columns:
                continue
            vigentes.append(e)
        self._eventos = vigentes
        return antes - len(vigentes)

    # ------------------------------------------------------------ guardado
    def a_lista(self):
        """Lista de dicts lista para json (va dentro del .hadarproy)."""
        return [asdict(e) for e in self._eventos]

    @classmethod
    def desde_lista(cls, datos):
        """Reconstruye desde lo guardado. Tolerante: ignora claves
        desconocidas y descarta entradas rotas en vez de impedir abrir el
        proyecto (un proyecto viejo, sin bitácora, entrega una lista vacía)."""
        conocidos = {f.name for f in fields(EventoProcedencia)}
        eventos = []
        for d in datos or []:
            try:
                limpio = {k: v for k, v in d.items() if k in conocidos}
                limpio["depende_de"] = list(limpio.get("depende_de") or [])
                limpio["detalle"] = dict(limpio.get("detalle") or {})
                eventos.append(EventoProcedencia(**limpio))
            except (TypeError, AttributeError):
                continue
        return cls(eventos)