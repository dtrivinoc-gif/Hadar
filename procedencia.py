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
El tipo "limpieza" está previsto para la etapa 3; el formato guardado ya lo
admite sin cambios.
"""
import os
import re
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime

from .formulas import extraer_columnas_formula, FormulaSeguridadError

TIPO_COLUMNA_CALCULADA = "columna_calculada"
TIPO_ARCHIVO_ORIGEN = "archivo_origen"
# Previsto para la etapa 3 (todavía no se registra):
TIPO_LIMPIEZA = "limpieza"

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
        self._eventos = [
            e for e in self._eventos
            if not (e.tipo == TIPO_COLUMNA_CALCULADA and e.tabla == tabla and e.columna == columna)
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