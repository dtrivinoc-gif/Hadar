"""
Generación del informe Narrativo: el asistente de preguntas sugeridas
(QuestionAssistant), los helpers para codificar/graficar anomalías como
imágenes embebidas en HTML, la biblioteca de impacto + recomendaciones
por anomalía, el generador del informe tipo libro (NarrativeGenerator) y
el generador de mapa causal (todavía en una etapa temprana).
"""
import base64
import html
import json

import numpy as np
import pandas as pd
import pyqtgraph as pg

from .anomalias import detectar_columnas_fecha, _sin_acentos_narrativa, buscar_coincidencias
from .contagio import calcular_indice_contagio, resumen_en_texto, columna_clave_de_tabla
from .correlaciones import detectar_correlaciones_significativas
from .cooccurrencia import detectar_coocurrencia_de_anomalias, columnas_con_anomalias_suficientes
from .regimen import detectar_quiebres_de_comportamiento
from .procedencia import (
    TIPO_COLUMNA_CALCULADA, evento_de_columna, evento_de_origen, resolver_columnas_base,
    usa_su_propio_valor_anterior, expresion_a_texto_amigable, fecha_legible, _nombre_de_fuente,
    TIPO_LIMPIEZA, TIPO_EDICION_MANUAL, TIPO_UNION_HOJA, estado_de_actualizacion, frase_desactualizacion,
    FRASE_OPERACION_INDICADOR as _FRASE_OPERACION_INDICADOR,
    aviso_de_union, impacto_de_columnas, etiqueta_corta_de_paso,
)

class QuestionAssistant:
    """Sugiere preguntas a partir de lo detectado, sin asumir el dominio del
    dataset. Nunca referencia una columna que no exista realmente en `df`."""

    def __init__(self, df, anomalias, max_preguntas=8):
        self.df = df
        self.anomalias = anomalias
        self.max_preguntas = max_preguntas

    def suggest(self):
        preguntas = []

        for a in self.anomalias:
            if a["gravedad"] != "alta":
                continue
            cols = ", ".join(f"'{c}'" for c in a["columnas"]) if a["columnas"] else "estos datos"
            if a["tipo"] == "temporal":
                preguntas.append(f"¿Qué está pasando en el proceso que registra {cols}?")
            elif a["tipo"] == "quiebre_patron":
                preguntas.append(f"¿Por qué {cols} se salió tanto de su comportamiento habitual?")
            elif a["tipo"] == "valores_nulos":
                preguntas.append(f"¿Por qué {cols} tiene tantos datos faltantes?")
            elif a["tipo"] == "duplicados":
                preguntas.append("¿Por qué hay filas duplicadas en los datos?")
            elif a["tipo"] == "deriva_historica":
                preguntas.append(f"¿Qué cambió de fondo en {cols} respecto de lo que veníamos viendo?")
            elif a["tipo"] == "patron_multivariado":
                preguntas.append(f"¿Qué explica que {cols} aparezcan combinadas así en ese registro?")
            else:
                preguntas.append(f"¿Por qué no se está cumpliendo la regla sobre {cols}?")

        proporcion_nulos = self.df.isna().mean()
        for col, prop in proporcion_nulos.items():
            if prop > 0.3:
                preguntas.append(f"¿Por qué '{col}' tiene un {prop * 100:.0f}% de datos faltantes?")

        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            n_unicos = self.df[col].nunique(dropna=True)
            if len(self.df) > 0 and 0 < n_unicos <= len(self.df) * 0.02 and n_unicos > 1:
                # Muy pocas categorías respecto al total de filas: candidata a
                # segmentación interesante ("¿cómo se ve esto por X?").
                preguntas.append(f"¿Cómo cambian los resultados según '{col}'?")

        vistas = set()
        preguntas_unicas = []
        for p in preguntas:
            if p not in vistas:
                vistas.add(p)
                preguntas_unicas.append(p)
        return preguntas_unicas[: self.max_preguntas]


# ----------------------------------------------------------------------------
# Impacto + Recomendación: reemplaza al capítulo "Escenarios alternativos
# (contrafactual)" que era un stub de v2. En vez de simular qué hubiera
# pasado (que exigía elegir una métrica objetivo genérica y mostrar un
# rango de incertidumbre que el usuario tendría que saber interpretar), se
# muestra algo más simple y igual de sofisticado por dentro: qué tan grave
# fue esto en términos llanos (severidad + % vs. lo típico, de
# anomalias.py), cuánto ha costado acumulado si es recurrente (de
# memoria.py) y una sugerencia concreta según el tipo de anomalía y las
# palabras del nombre de columna (mismo patrón que TOKENS_TEMPORALES_PARES
# en anomalias.py). Una sola frase por anomalía, sin estadística que
# explicar.
# ----------------------------------------------------------------------------
_CLASE_SEVERIDAD = {"leve": "leve", "moderado": "moderado", "grave": "grave", "crítico": "critico"}

_RECOMENDACIONES_POR_PALABRA_CLAVE = [
    (["stock", "inventario", "existencia"], "Revisar el nivel de stock/inventario de este producto."),
    (["entrega", "envio", "despacho", "delivery", "logistica"], "Revisar el proceso de despacho o logística."),
    (["pago", "cobro", "factura", "boleta", "transaccion"], "Revisar el procesamiento de pagos o facturación."),
    (["cliente", "comprador", "usuario"], "Revisar el historial de este cliente en particular."),
    (["proveedor"], "Contactar al proveedor asociado a este registro."),
    (["empleado", "vendedor", "cajero", "encargado"], "Revisar con la persona responsable de este registro."),
    (["precio", "monto", "valor", "venta", "ingreso"], "Revisar si hubo un error de carga de datos o un cambio real."),
]

_RECOMENDACIONES_BASE_POR_TIPO = {
    "valores_nulos": "Revisar por qué no se está registrando este dato -- puede ser un problema de carga o de proceso.",
    "duplicados": "Revisar si estas filas duplicadas son un error de carga o una operación real repetida.",
    "regla_negocio": "Revisar los casos puntuales que no cumplen esta regla.",
    "temporal": "Revisar el proceso de captura de fechas en este paso.",
    "quiebre_patron": "Revisar qué ocurrió puntualmente en este registro.",
    "deriva_historica": "Revisar si algo cambió de fondo en este dato (proceso, fuente, medición) desde la última vez.",
    "patron_multivariado": "Revisar la combinación de columnas señalada -- por separado cada una se ve normal, pero juntas no.",
}


def sugerir_recomendacion(anomalia):
    """Sugerencia concreta para una anomalía puntual, según su tipo y las
    palabras del nombre de columna. Biblioteca chica y ampliable a
    propósito -- no pretende cubrir todos los rubros posibles desde el
    día uno, y crece agregando líneas a las listas de arriba, sin tocar
    el resto del sistema."""
    columnas = anomalia.get("columnas") or []
    texto_columnas = " ".join(_sin_acentos_narrativa(str(c)) for c in columnas)
    for palabras, sugerencia in _RECOMENDACIONES_POR_PALABRA_CLAVE:
        if any(_sin_acentos_narrativa(p) in texto_columnas for p in palabras):
            return sugerencia
    return _RECOMENDACIONES_BASE_POR_TIPO.get(
        anomalia.get("tipo"), "Revisar este hallazgo con más detalle."
    )


_ORDEN_SEVERIDAD = {"leve": 0, "moderado": 1, "grave": 2, "crítico": 3}


def _agrupar_anomalias_por_columna(anomalias):
    """Junta las anomalías de tipo 'quiebre_patron' que comparten columna en
    un solo grupo, para no repetir el mismo párrafo de recurrencia/memoria
    una vez por cada fila detectada (ej. 'consumo_kwh' con 2 filas atípicas
    generaba 2 bloques casi idénticos). El resto de los tipos, que ya vienen
    consolidados en una sola anomalía desde anomalias.py, se deja igual --
    una entrada por hallazgo, en el orden en que aparecieron."""
    grupos = {}
    entradas = []
    for a in anomalias:
        if a["tipo"] == "quiebre_patron" and a.get("columnas") and a.get("fila_indice") is not None:
            col = a["columnas"][0]
            if col not in grupos:
                grupos[col] = []
                entradas.append(("quiebre_grupo", col))
            grupos[col].append(a)
        else:
            entradas.append(("individual", a))

    resultado = []
    for tipo_entrada, valor in entradas:
        if tipo_entrada == "quiebre_grupo":
            if valor not in grupos:
                continue  # ya se resolvió la primera vez que apareció esta columna
            resultado.append(("quiebre_grupo", grupos.pop(valor)))
        else:
            resultado.append(("individual", valor))
    return resultado


def _entrada_es_gravedad_alta(entrada):
    _, valor = entrada
    items = valor if isinstance(valor, list) else [valor]
    return any(a["gravedad"] == "alta" for a in items)


def _frase_valor_quiebre(anomalia, valor_tipico):
    """Una fila puntual de un grupo de quiebre_patron, redactada como parte
    de una oración ('la fila 67 llegó a 196,14 (31% sobre lo típico)') en
    vez de la frase completa e independiente que arma anomalias.py -- acá
    solo se necesita el fragmento para insertarlo en la frase del grupo."""
    fila = anomalia.get("fila_indice")
    diferencia_absoluta = anomalia.get("diferencia_absoluta")
    diferencia_porcentual = anomalia.get("diferencia_porcentual")
    if fila is None or diferencia_absoluta is None or valor_tipico is None:
        return anomalia["descripcion"]
    valor = valor_tipico + diferencia_absoluta
    direccion = "sobre" if diferencia_absoluta > 0 else "bajo"
    if diferencia_porcentual is not None:
        return f"la fila {fila + 1} llegó a {valor:,.2f} ({abs(diferencia_porcentual):.0f}% {direccion} lo típico)"
    return f"la fila {fila + 1} llegó a {valor:,.2f}"


def _bloque_quiebre_grupo(anomalias_col, df, modo_impresion, indicadores=None, eventos=None):
    """Arma un único bloque HTML para todas las anomalías de quiebre_patron
    de una misma columna: una frase que resume todas las filas afectadas, y
    la severidad/recurrencia/sugerencia mostradas UNA sola vez (tomando la
    peor de las filas del grupo como representativa), no una vez por fila."""
    anomalias_col = sorted(anomalias_col, key=lambda a: a.get("fila_indice", 0))
    columna = anomalias_col[0]["columnas"][0]
    valor_tipico = anomalias_col[0].get("valor_tipico")
    n = len(anomalias_col)

    frases_valor = [_frase_valor_quiebre(a, valor_tipico) for a in anomalias_col]
    if n == 1:
        base = f"En '{columna}', {frases_valor[0]}."
    else:
        tipico_txt = f" (lo típico: {valor_tipico:,.2f})" if valor_tipico is not None else ""
        listado = ", ".join(frases_valor[:-1]) + " y " + frases_valor[-1]
        base = f"'{columna}' se salió de lo típico en {n} filas{tipico_txt}: {listado}."

    peor = max(
        anomalias_col,
        key=lambda a: _ORDEN_SEVERIDAD.get(a.get("severidad_impacto"), 0),
    )
    impacto_html = _linea_impacto_html(peor)

    frase_memoria = peor.get("frase_memoria")
    frase_html = f"<div class='frase-memoria'>{frase_memoria}</div>" if frase_memoria else ""

    frase_acumulado = peor.get("frase_impacto_acumulado")
    frase_acumulado_html = (
        f"<div class='frase-memoria'>{frase_acumulado}</div>" if frase_acumulado else ""
    )

    # Coincidencias sí se listan por fila (cada una puede coincidir con algo
    # distinto), pero sin repetir la frase introductoria de recurrencia.
    coincidencias_html = "".join(_linea_coincidencias_html(a, df) for a in anomalias_col)

    indicadores_html = _linea_indicadores_html([columna], indicadores)
    indicadores_html += _lineas_procedencia_anomalia_html([columna], eventos)

    sugerencia_html = f"<div class='sugerencia'>💡 {sugerir_recomendacion(peor)}</div>"
    grafico = _grafico_anomalia_narrativa(peor, df)

    # El enlace "Marcar como resuelta" queda pendiente para grupos: el
    # sistema actual resuelve una anomalía a la vez (ver memoria.py), y acá
    # hay varias filas fusionadas en un solo bloque -- se omite en vez de
    # marcar solo una de las filas sin que el usuario sepa cuál.
    clase = "anomalia" if any(a["gravedad"] == "alta" for a in anomalias_col) else "anomalia media"
    cuerpo = (
        f"{base}{impacto_html}{frase_html}{frase_acumulado_html}{coincidencias_html}"
        f"{indicadores_html}{sugerencia_html}{grafico}"
    )
    return clase, cuerpo


def _indicadores_que_dependen_de(columnas, indicadores):
    """De la lista de indicadores del usuario, cuáles usan alguna de las
    columnas dadas -- para avisar, en una anomalía, qué indicadores quedan
    afectados. `columnas` suele venir de una sola anomalía (a veces son
    varias, ej. una regla de negocio entre dos columnas). Nunca revienta
    por un indicador mal armado (ej. una fórmula con error de sintaxis):
    ese indicador simplemente no aporta columnas conocidas."""
    columnas = set(columnas or [])
    if not columnas:
        return []
    afectados = []
    for ind in indicadores or []:
        try:
            depende_de = ind.columnas_de_las_que_depende()
        except Exception:
            continue
        if depende_de & columnas:
            afectados.append(ind.nombre)
    return afectados


def _linea_indicadores_html(columnas, indicadores):
    afectados = _indicadores_que_dependen_de(columnas, indicadores)
    if not afectados:
        return ""
    lista = ", ".join(f"'{n}'" for n in afectados)
    return f"<div class='frase-memoria'>Esto afecta a tus indicadores: {lista}.</div>"


def _lineas_procedencia_anomalia_html(columnas, eventos):
    """Cruza una anomalía con la historia de sus columnas (procedencia.py): si la columna
    la calculó Hadar (y de dónde puede venir el problema), por qué pasos ya pasaron esos
    datos, y qué columnas calculadas salen de ella (habría que volver a crearlas si se
    corrige). Solo dice lo que está anotado; sin eventos no muestra nada."""
    if not columnas or not eventos:
        return ""
    imp = impacto_de_columnas(eventos, columnas)
    lineas = []
    for col, formula, base in imp["son_calculadas"]:
        desde = f" Si el dato está mal, puede venir de: {', '.join(base)}." if base else ""
        lineas.append(f"«{col}» es una columna que calculó Hadar (= {formula}).{desde}")
    if imp["pasos_previos"]:
        etiquetas = []
        for e in imp["pasos_previos"]:
            et = etiqueta_corta_de_paso(e)
            if et not in etiquetas:
                etiquetas.append(et)
        lineas.append("Antes de esto, esos datos pasaron por: " + ", ".join(etiquetas) + ".")
    if imp["calculadas_afectadas"]:
        lista = ", ".join(f"«{c}»" for c in imp["calculadas_afectadas"])
        lineas.append(f"Hay columnas calculadas que salen de esta: {lista}. "
                      f"Si la corriges, conviene volver a crearlas.")
    return "".join(f"<div class='frase-memoria'>{html.escape(l)}</div>" for l in lineas)


def _linea_impacto_html(anomalia):
    """Solo las anomalías de tipo 'quiebre_patron' traen severidad_impacto
    (viene de anomalias.py) -- las demás no tienen un 'valor típico'
    numérico con el que compararse, así que no muestran esta línea."""
    severidad = anomalia.get("severidad_impacto")
    if not severidad:
        return ""
    porcentaje = anomalia.get("diferencia_porcentual")
    texto_porcentaje = ""
    if porcentaje is not None:
        direccion = "sobre" if porcentaje > 0 else "bajo"
        texto_porcentaje = f" ({abs(porcentaje):.0f}% {direccion} lo típico)"
    clase = _CLASE_SEVERIDAD.get(severidad, "leve")
    return f"<div class='impacto impacto-{clase}'>Impacto: {severidad.upper()}{texto_porcentaje}</div>"


def _linea_coincidencias_html(anomalia, df):
    """Si otra columna también tenía un valor fuera de lo común en las
    mismas filas, se menciona como pista -- SIEMPRE con lenguaje de
    coincidencia, nunca de causa (ver buscar_coincidencias en anomalias.py,
    que es quien decide qué cuenta como "fuera de lo común"; acá solo se
    redacta con el cuidado que corresponde)."""
    try:
        coincidencias = buscar_coincidencias(df, anomalia)
    except Exception:
        return ""
    if not coincidencias:
        return ""
    partes = []
    for c in coincidencias:
        valor = c["valor"]
        valor_txt = f"{valor:,.2f}" if isinstance(valor, (int, float)) else f"'{valor}'"
        partes.append(f"'{c['columna']}' (valor: {valor_txt})")
    return (
        f"<div class='coincidencia'>🔗 Esto coincidió con un valor fuera de lo común en "
        f"{', '.join(partes)}. No necesariamente es la causa, pero vale la pena revisarlo.</div>"
    )


# CSS pensado para QTextBrowser (subconjunto de CSS 2.1 que soporta Qt Rich
# Text): sin box-shadow ni border-radius, que Qt ignora igual sin romper nada.
# ----------------------------------------------------------------------------
# Mini-gráficos para el informe de Narrativa: se renderizan con PyQtGraph (el
# mismo motor que usa el resto de la app) sin necesidad de mostrarlos en
# pantalla, y se insertan en el HTML como imágenes PNG en base64 -- QTextBrowser
# (el widget que muestra el informe) sí sabe interpretar "data:image/..." en
# un <img src="...">. Si algo falla (entorno sin soporte gráfico, etc.) se
# devuelve None y el informe simplemente no muestra esa imagen -- nunca rompe
# el texto, que sigue siendo la fuente de verdad.
# ----------------------------------------------------------------------------
def _codificar_identidad_anomalia(anomalia):
    """Empaqueta tipo+columnas+contexto de una anomalía en un código corto
    para meterlo en un href (`resolver:<código>` / `falso_positivo:<código>`)
    del informe. Así el clic en "Marcar como resuelta" o "Marcar como no es
    una anomalía" puede viajar hasta MemoriaHadar sin depender de que el
    texto de la descripción se mantenga igual. El contexto (ej.
    "Producto=Jamón") distingue una categoría puntual de otra cuando la
    anomalía viene de un chequeo segmentado -- ver anomalias.py."""
    try:
        payload = json.dumps({
            "tipo": anomalia.get("tipo"),
            "columnas": anomalia.get("columnas") or [],
            "contexto": anomalia.get("contexto"),
        })
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    except Exception:
        return None


def _decodificar_identidad_anomalia(codigo):
    try:
        relleno = "=" * (-len(codigo) % 4)
        payload = base64.urlsafe_b64decode((codigo + relleno).encode("ascii")).decode("utf-8")
        return json.loads(payload)
    except Exception:
        return None


def _imagen_qt_a_data_uri(imagen_qimage):
    from PySide6.QtCore import QBuffer, QIODevice
    buffer = QBuffer()
    buffer.open(QIODevice.WriteOnly)
    imagen_qimage.save(buffer, "PNG")
    datos = bytes(buffer.data())
    buffer.close()
    return "data:image/png;base64," + base64.b64encode(datos).decode("ascii")


def _grafico_narrativa_serie(valores, x=None, resaltar=None, titulo="", ancho=560, alto=190):
    """Línea con los valores de una columna a lo largo de las filas, con los
    puntos marcados en `resaltar` (lista de posiciones, no de índices de
    pandas) pintados en rojo encima. Pensado para 'quiebre_patron'."""
    try:
        import pyqtgraph.exporters as pg_exporters
        x = list(x) if x is not None else list(range(len(valores)))
        y = list(valores)

        pw = pg.PlotWidget()
        pw.resize(ancho, alto)
        pw.setBackground('w')
        pw.showGrid(x=False, y=True, alpha=0.15)
        pw.getAxis('bottom').setPen(pg.mkPen('#9ca3af'))
        pw.getAxis('bottom').setStyle(showValues=False)  # la posición de la fila no le dice nada al lector
        pw.getAxis('left').setPen(pg.mkPen('#9ca3af'))
        pw.plot(x, y, pen=pg.mkPen('#9ca3af', width=1.3))

        if resaltar:
            xs_r = [x[i] for i in resaltar if 0 <= i < len(x)]
            ys_r = [y[i] for i in resaltar if 0 <= i < len(y)]
            if xs_r:
                pw.addItem(pg.ScatterPlotItem(xs_r, ys_r, brush=pg.mkBrush('#dc2626'),
                                               pen=pg.mkPen('#dc2626'), size=8))
        if titulo:
            pw.setTitle(titulo, color='#374151', size='8pt')

        exportador = pg_exporters.ImageExporter(pw.getPlotItem())
        exportador.parameters()['width'] = ancho
        return _imagen_qt_a_data_uri(exportador.export(toBytes=True))
    except Exception:
        return None


def _grafico_narrativa_barras(etiquetas, valores, colores=None, titulo="", ancho=420, alto=180):
    """Barras verticales simples (nulos vs completos, únicas vs duplicadas,
    etc.). `colores` es opcional, una lista de colores hex del mismo largo
    que `valores`."""
    try:
        import pyqtgraph.exporters as pg_exporters
        pw = pg.PlotWidget()
        pw.resize(ancho, alto)
        pw.setBackground('w')
        pw.showGrid(x=False, y=True, alpha=0.15)
        pw.getAxis('bottom').setTicks([[(i, etq) for i, etq in enumerate(etiquetas)]])
        pw.getAxis('bottom').setPen(pg.mkPen('#9ca3af'))
        pw.getAxis('left').setPen(pg.mkPen('#9ca3af'))

        colores = colores or (["#9ca3af"] * len(valores))
        for i, (valor, color) in enumerate(zip(valores, colores)):
            barra = pg.BarGraphItem(x=[i], height=[valor], width=0.6, brush=pg.mkBrush(color))
            pw.addItem(barra)
        pw.setXRange(-0.6, len(valores) - 0.4)

        if titulo:
            pw.setTitle(titulo, color='#374151', size='8pt')

        exportador = pg_exporters.ImageExporter(pw.getPlotItem())
        exportador.parameters()['width'] = ancho
        return _imagen_qt_a_data_uri(exportador.export(toBytes=True))
    except Exception:
        return None


def _grafico_narrativa_dispersion(x, y, x_resaltar=None, y_resaltar=None, titulo="", ancho=420, alto=300):
    """Dispersión de dos columnas numéricas, con la fila atípica marcada en
    rojo -- pensado para 'patron_multivariado', donde lo raro es la
    COMBINACIÓN de dos valores, no cada uno por separado."""
    try:
        import pyqtgraph.exporters as pg_exporters
        pw = pg.PlotWidget()
        pw.resize(ancho, alto)
        pw.setBackground('w')
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.getAxis('bottom').setPen(pg.mkPen('#9ca3af'))
        pw.getAxis('left').setPen(pg.mkPen('#9ca3af'))
        pw.addItem(pg.ScatterPlotItem(list(x), list(y), brush=pg.mkBrush('#9ca3af'),
                                       pen=None, size=6))
        if x_resaltar is not None and y_resaltar is not None:
            pw.addItem(pg.ScatterPlotItem([x_resaltar], [y_resaltar], brush=pg.mkBrush('#dc2626'),
                                           pen=pg.mkPen('#dc2626'), size=10))
        if titulo:
            pw.setTitle(titulo, color='#374151', size='8pt')
        exportador = pg_exporters.ImageExporter(pw.getPlotItem())
        exportador.parameters()['width'] = ancho
        return _imagen_qt_a_data_uri(exportador.export(toBytes=True))
    except Exception:
        return None


def _grafico_anomalia_narrativa(anomalia, df):
    """Arma el gráfico que corresponde según el tipo de anomalía, usando los
    datos reales del df. Devuelve HTML (un <img> o cadena vacía) -- nunca
    lanza excepción hacia quien la llama."""
    try:
        tipo = anomalia.get("tipo")
        columnas = anomalia.get("columnas") or []

        if tipo == "quiebre_patron" and columnas:
            col = columnas[0]
            serie_completa = pd.to_numeric(df[col], errors="coerce")
            posiciones = list(range(len(serie_completa)))
            indices_atipicos = anomalia.get("indices_atipicos") or []
            pos_atipicos = [df.index.get_loc(i) for i in indices_atipicos if i in df.index]
            uri = _grafico_narrativa_serie(
                serie_completa.fillna(serie_completa.mean()).tolist(),
                x=posiciones, resaltar=pos_atipicos, titulo=f"'{col}' por fila",
            )
        elif tipo == "temporal" and len(columnas) == 2:
            col_anterior, col_posterior = columnas
            f_ant = pd.to_datetime(df[col_anterior], errors="coerce", format="mixed")
            f_pos = pd.to_datetime(df[col_posterior], errors="coerce", format="mixed")
            dias = (f_pos - f_ant).dt.days
            posiciones = list(range(len(dias)))
            resaltar = [i for i, v in enumerate(dias) if pd.notna(v) and v < 0]
            uri = _grafico_narrativa_serie(
                dias.fillna(0).tolist(), x=posiciones, resaltar=resaltar,
                titulo=f"Días entre '{col_anterior}' y '{col_posterior}' (rojo = orden invertido)",
            )
        elif tipo == "valores_nulos" and columnas:
            col = columnas[0]
            n_nulos = int(df[col].isna().sum())
            n_completos = len(df) - n_nulos
            uri = _grafico_narrativa_barras(
                ["Completos", "Faltantes"], [n_completos, n_nulos],
                colores=["#9ca3af", "#dc2626"], titulo=f"'{col}': completos vs. faltantes",
            )
        elif tipo == "duplicados":
            n_dup = anomalia.get("filas_afectadas", 0)
            n_unicas = len(df) - n_dup
            uri = _grafico_narrativa_barras(
                ["Únicas", "Duplicadas"], [n_unicas, n_dup],
                colores=["#9ca3af", "#dc2626"], titulo="Filas únicas vs. duplicadas",
            )
        elif tipo == "patron_multivariado" and len(columnas) >= 2 and anomalia.get("fila_indice") is not None:
            col_x, col_y = columnas[0], columnas[1]
            serie_x = pd.to_numeric(df[col_x], errors="coerce")
            serie_y = pd.to_numeric(df[col_y], errors="coerce")
            fila = anomalia["fila_indice"]
            uri = _grafico_narrativa_dispersion(
                serie_x.dropna(), serie_y.dropna(),
                x_resaltar=serie_x.get(fila), y_resaltar=serie_y.get(fila),
                titulo=f"'{col_x}' vs '{col_y}'",
            )
        else:
            uri = None

        if not uri:
            return ""
        return f'<div><img src="{uri}" width="{420 if tipo in ("valores_nulos", "duplicados") else 520}"></div>'
    except Exception:
        return ""


_CSS_LIBRO = """
<style>
  body { font-family: Georgia, 'Times New Roman', serif; background:#f7f5f0; color:#1a1a1a;
         margin:0; padding:16px; }
  .libro { margin: 0 auto; }
  .capitulo { background:#ffffff; border:1px solid #e2ddd0;
              padding:16px 20px; margin-bottom:14px; }
  .capitulo h2 { margin-top:0; font-size:15px; letter-spacing:0.03em; color:#4b3b2a;
                 text-transform:uppercase; border-bottom:1px solid #e2ddd0; padding-bottom:6px;}
  .capitulo p { line-height:1.5; font-size:13.5px; }
  .anomalia { margin:8px 0; padding:8px 10px; border-left:3px solid #b45309; background:#fff7ed;}
  .anomalia.media { border-left-color:#a8a29e; background:#fafaf9; }
  .impacto { font-weight:bold; font-size:12px; margin-top:4px; letter-spacing:0.02em; }
  .impacto-leve { color:#78716c; }
  .impacto-moderado { color:#b45309; }
  .impacto-grave { color:#c2410c; }
  .impacto-critico { color:#b91c1c; }
  .sugerencia { font-size:12.5px; margin-top:4px; color:#374151; }
  .coincidencia { font-size:12px; margin-top:4px; color:#57534e; }
  .frase-memoria { color:#92672f; font-style:italic; font-size:12px; margin-top:4px; }
  .resolver-link { margin-top:6px; }
  .resolver-link a { color:#78716c; font-size:11.5px; text-decoration:none; border-bottom:1px dotted #a8a29e; }
  .resolver-link a:hover { color:#4b3b2a; }
  .pendiente { color:#78716c; font-style:italic; }
  .capitulo h3 { font-size:13px; color:#4b3b2a; margin:14px 0 4px 0; }
  .origen { margin:8px 0; padding:8px 10px; border-left:3px solid #4b3b2a; background:#faf8f3; }
  .origen-titulo { font-weight:bold; font-size:13.5px; }
  .origen-formula { font-family: Consolas, 'Courier New', monospace; font-size:12.5px; margin-top:3px; }
  .origen-detalle { font-size:12.5px; margin-top:3px; color:#374151; }
  .pregunta { padding:4px 0; }
  ul { padding-left:18px; }
</style>
"""


class NarrativeGenerator:
    """Arma el informe HTML tipo libro a partir de datos reales del df y de
    las anomalías ya detectadas. El capítulo de Relaciones entre variables
    se muestra como "próximamente" hasta que exista una v2 real (ver stub
    más abajo) -- nunca se inventa un número para rellenarlo. El capítulo
    de "Escenarios alternativos" que existía antes (simular qué hubiera
    pasado) se reemplazó por algo más simple y verificable: cada anomalía
    ya trae su impacto (severidad + % vs. lo típico), su impacto acumulado
    si es recurrente, y una sugerencia concreta -- ver Capítulo "Lo que no
    encaja" más abajo, y sugerir_recomendacion()/_linea_impacto_html() más
    arriba en este archivo.

    Base de datos relacional (opcional): si se entrega `tablas_relacionadas`
    (las OTRAS tablas cargadas junto a ésta, cada una con sus propias
    anomalías ya detectadas) y `relaciones` (el esquema sugerido/confirmado
    entre todas las tablas, de `ontologia.inferir_relaciones` o del diálogo
    "Ver esquema"), el informe agrega 2 capítulos más: un resumen de las
    otras tablas, y un cruce real de anomalías a través de las relaciones
    (ej. "el cliente X concentra 3 pedidos con anomalías"). Sin esos
    argumentos, el informe queda exactamente igual que con una sola tabla."""

    def __init__(self, df, anomalias, nombre_tabla=None, tablas_relacionadas=None, relaciones=None,
                 indicadores=None, modo_impresion=False, eventos_procedencia=None):
        self.df = df
        self.anomalias = anomalias
        self.nombre_tabla = nombre_tabla
        self.tablas_relacionadas = tablas_relacionadas or {}
        self.relaciones = relaciones or []
        # Los indicadores que el usuario armó en la pestaña Indicadores (o
        # que Hadar le sugirió) -- se usan solo para avisar, en cada
        # anomalía, cuáles quedan afectados (ver _linea_indicadores_html).
        # Duck typing a propósito: cualquier objeto con .nombre y
        # .columnas_de_las_que_depende() sirve, sin importar Indicador acá.
        self.indicadores = indicadores or []
        # Eventos de procedencia (procedencia.py) de la tabla que se está
        # narrando: hoy, las columnas que Hadar calculó y con qué fórmula.
        # Alimentan el capítulo "Origen de los datos".
        self.eventos_procedencia = list(eventos_procedencia or [])
        # True cuando el HTML es para exportar a PDF / imprimir: se omiten
        # los elementos que solo tienen sentido dentro de la app (el enlace
        # "Marcar como resuelta" no sirve de nada en un documento estático,
        # y quedaría como texto suelto pegado a la descripción).
        self.modo_impresion = modo_impresion

    def generar_html(self, nombre_dataset="tu dataset"):
        n = 1
        capitulos = []

        capitulos.append(self._capitulo_panorama(nombre_dataset, n)); n += 1
        capitulo_origen = self._capitulo_origen(n)
        if capitulo_origen:
            capitulos.append(capitulo_origen); n += 1
        capitulos.append(self._capitulo_anomalias(n)); n += 1

        if self.tablas_relacionadas:
            capitulos.append(self._capitulo_otras_tablas(n)); n += 1
            capitulos.append(self._capitulo_cruces(n)); n += 1
            capitulo_contagio = self._capitulo_contagio(n)
            if capitulo_contagio:
                capitulos.append(capitulo_contagio); n += 1

        capitulo_correlaciones = self._capitulo_correlaciones(n)
        if capitulo_correlaciones:
            capitulos.append(capitulo_correlaciones); n += 1
        capitulos.append(self._capitulo_preguntas(n)); n += 1
        capitulos.append(self._capitulo_recomendaciones(n)); n += 1

        cuerpo = "\n".join(capitulos)
        return f"<html><head><meta charset='utf-8'>{_CSS_LIBRO}</head><body><div class='libro'>{cuerpo}</div></body></html>"

    def _capitulo_panorama(self, nombre_dataset, n):
        n_filas, n_cols = self.df.shape
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_fecha = detectar_columnas_fecha(self.df)

        detalle_fecha = ""
        if columnas_fecha:
            col = columnas_fecha[0]
            fechas = pd.to_datetime(self.df[col], errors="coerce", format="mixed").dropna()
            if not fechas.empty:
                detalle_fecha = (
                    f" Los datos cubren desde {fechas.min():%d-%m-%Y} hasta "
                    f"{fechas.max():%d-%m-%Y} (según '{col}')."
                )

        detalle_relacional = ""
        if self.tablas_relacionadas:
            total_tablas = len(self.tablas_relacionadas) + 1
            detalle_relacional = (
                f" Esta tabla forma parte de una base relacional de {total_tablas} tablas "
                f"en total, conectadas por {len(self.relaciones)} relación(es) sugerida(s)."
            )

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · El panorama general</h2>
          <p>{nombre_dataset} tiene {n_filas:,} filas y {n_cols} columnas,
             de las cuales {len(num_cols)} son numéricas.{detalle_fecha}{detalle_relacional}</p>
        </div>"""

    # ------------------------------------------------------------------
    # Origen de los datos (procedencia.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _unir_con_y(items):
        """['A', 'B', 'C'] -> 'A, B y C' (español natural)."""
        items = list(items)
        if len(items) <= 1:
            return "".join(items)
        return ", ".join(items[:-1]) + " y " + items[-1]

    @staticmethod
    def _origen_fuente_html(origen):
        """Bloque 'De dónde vinieron los datos': archivo (y hoja) o base SQL
        Server, y cuándo se leyó. Solo se muestra el NOMBRE del archivo, no
        la ruta completa (el informe se puede exportar y compartir)."""
        esc = html.escape
        if origen is None:
            return (
                "<h3>De dónde vinieron los datos</h3>"
                "<p class='pendiente'>No hay registro de dónde vinieron los datos de esta tabla "
                "(por ejemplo, si el proyecto se guardó antes de que Hadar llevara este registro).</p>"
            )
        d = origen.detalle
        lineas = []
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
        lineas.append(
            "<i>Hadar trabaja con una copia tomada en ese momento: si la fuente cambia "
            "después, no se entera sola.</i>"
        )
        detalle_html = "".join(
            f"<div class='origen-detalle'>{l if l.startswith('<i>') else esc(l)}</div>" for l in lineas
        )
        return (
            "<h3>De dónde vinieron los datos</h3>"
            f"<div class='origen'><div class='origen-titulo'>{esc(_nombre_de_fuente(d))}</div>"
            f"{detalle_html}</div>"
        )

    def _diagrama_origen_html(self):
        """El mismo diagrama de la sub-pestaña «Origen», como imagen (así también sale
        en el PDF). Solo si tiene al menos 3 cajas (con menos no aporta nada). Si algo
        falla (sin soporte gráfico, etc.) simplemente no se muestra: el texto de abajo
        sigue contando lo mismo."""
        try:
            from .config import THEMES
            from .origen_grafo import construir_grafo_origen
            from .origen_ui import render_imagen_grafo
            grafo = construir_grafo_origen(
                self.eventos_procedencia, [str(c) for c in self.df.columns], self.indicadores
            )
            if len(grafo.nodos) < 3:
                return ""
            resultado = render_imagen_grafo(grafo, THEMES["light"])
            if resultado is None:
                return ""
            imagen, ancho, _alto = resultado
            mostrado = int(min(ancho, 520 if self.modo_impresion else 640))
            pie = (
                "<p class='pendiente'>Cada flecha va de lo que alimenta a lo que sale de ello. "
                "Gris: de dónde vinieron los datos. Verde azulado: lo que se les hizo. "
                "Morado: columnas calculadas e indicadores. «Ojo»: conviene revisar.</p>"
            )
            if not self.modo_impresion:
                pie += (
                    "<p class='pendiente'>Para ver el detalle de cada caja y a qué puede afectar, "
                    "usa la sub-pestaña «Origen».</p>"
                )
            return f"<div><img src=\"{_imagen_qt_a_data_uri(imagen)}\" width=\"{mostrado}\"></div>{pie}"
        except Exception:
            return ""

    @staticmethod
    def _origen_hechos_html(hechos, eventos=()):
        """'Qué se le hizo a los datos': las correcciones de Limpieza sugerida y
        los cambios a mano, en orden. Cada línea sale de lo que se anotó al
        aplicarla (comparando antes y después), no de una suposición."""
        esc = html.escape
        items = []
        for e in hechos:
            fecha = fecha_legible(e.fecha)
            extra = ""
            if e.tipo == TIPO_LIMPIEZA and e.detalle.get("subtipo") == "duplicado":
                extra = (
                    f" La tabla pasó de {e.detalle.get('filas_antes', 0):,} "
                    f"a {e.detalle.get('filas_despues', 0):,} filas."
                )
            elif e.tipo == TIPO_UNION_HOJA:
                aviso = aviso_de_union(e, eventos)
                extra = f" Ojo: {aviso}" if aviso else ""
            prefijo = f"<b>{esc(fecha)}</b> · " if fecha else ""
            items.append(f"<li>{prefijo}{esc(e.descripcion)}{esc(extra)}</li>")
        return (
            "<h3>Qué se le hizo a los datos</h3>"
            "<p class='pendiente'>Correcciones hechas dentro de Hadar, en orden. "
            "El archivo original no se modificó.</p>"
            "<ul>" + "".join(items) + "</ul>"
        )

    def _capitulo_origen(self, n):
        """Origen de los datos: de dónde sale cada cifra que NO viene tal cual
        de la carga -- las columnas que Hadar calculó (con su fórmula y las
        columnas de las que salen, siguiendo la cadena hasta las de partida)
        y los indicadores del usuario. Todo sale de lo que se anotó al
        momento de calcular; no se reconstruye ni se supone nada. Devuelve
        None (sin gastar número de capítulo) si no hay nada que contar."""
        calculadas = [
            e for e in self.eventos_procedencia
            if e.tipo == TIPO_COLUMNA_CALCULADA and e.columna in self.df.columns
        ]
        origen = evento_de_origen(self.eventos_procedencia)
        # Limpiezas y cambios a mano, en orden (una edición a mano solo cuenta si su
        # columna sigue existiendo).
        hechos = sorted(
            (
                e for e in self.eventos_procedencia
                if e.tipo in (TIPO_LIMPIEZA, TIPO_UNION_HOJA)
                or (e.tipo == TIPO_EDICION_MANUAL and e.columna in self.df.columns)
            ),
            key=lambda e: e.fecha or "",
        )
        if origen is None and not hechos and not calculadas and not self.indicadores:
            return None

        esc = html.escape
        partes = [self._diagrama_origen_html(), self._origen_fuente_html(origen)]
        if hechos:
            partes.append(self._origen_hechos_html(hechos, self.eventos_procedencia))

        if calculadas:
            bloques = []
            for e in calculadas:
                formula = esc(e.detalle.get("formula_texto") or expresion_a_texto_amigable(
                    e.detalle.get("expresion", "")))
                usa = [d for d in e.depende_de if d != e.columna]
                detalle = []
                if usa:
                    etiquetas = [
                        f"{esc(d)} (también calculada)" if evento_de_columna(self.eventos_procedencia, d)
                        else esc(d)
                        for d in usa
                    ]
                    detalle.append(f"Usa: {self._unir_con_y(etiquetas)}.")
                elif not usa_su_propio_valor_anterior(e):
                    detalle.append("No usa ninguna otra columna.")
                if usa_su_propio_valor_anterior(e):
                    detalle.append("También usa el valor que tenía antes esta misma columna.")
                base = resolver_columnas_base(self.eventos_procedencia, e.columna)
                if base and any(evento_de_columna(self.eventos_procedencia, d) for d in usa):
                    detalle.append(
                        f"En último término sale de: {self._unir_con_y([esc(b) for b in base])}."
                    )
                fecha = fecha_legible(e.fecha)
                if fecha:
                    detalle.append(f"Calculada el {fecha}.")
                aviso = frase_desactualizacion(
                    estado_de_actualizacion(self.eventos_procedencia, e.columna)
                )
                if aviso:
                    detalle.append(f"<b>Ojo:</b> {esc(aviso)}")
                detalle_html = "".join(f"<div class='origen-detalle'>{d}</div>" for d in detalle)
                bloques.append(
                    f"<div class='origen'><div class='origen-titulo'>{esc(str(e.columna))}</div>"
                    f"<div class='origen-formula'>= {formula}</div>{detalle_html}</div>"
                )
            partes.append(
                "<h3>Columnas que Hadar calculó</h3>"
                "<p class='pendiente'>Se calcularon una sola vez, con los datos que había en ese "
                "momento. Si después corriges datos de las columnas de las que salen, Hadar no "
                "las recalcula sola.</p>" + "".join(bloques)
            )

        if self.indicadores:
            items = []
            for ind in self.indicadores:
                nombre = esc(str(getattr(ind, "nombre", "Indicador")))
                operacion = getattr(ind, "operacion", None)
                columna = getattr(ind, "columna", None)
                formula = getattr(ind, "formula", None)
                if operacion == "formula":
                    como = f"fórmula <span style='font-family:Consolas, monospace'>{esc(expresion_a_texto_amigable(formula or ''))}</span>"
                elif columna:
                    como = f"{_FRASE_OPERACION_INDICADOR.get(operacion, 'cálculo sobre')} {esc(str(columna))}"
                else:
                    como = "cálculo sin columna definida"
                try:
                    depende_de = sorted(ind.columnas_de_las_que_depende())
                except Exception:
                    depende_de = []
                extra = ""
                calculadas_usadas = [
                    d for d in depende_de if evento_de_columna(self.eventos_procedencia, d)
                ]
                if calculadas_usadas:
                    frases = []
                    for d in calculadas_usadas:
                        base = resolver_columnas_base(self.eventos_procedencia, d)
                        sufijo = f" (sale de {self._unir_con_y([esc(b) for b in base])})" if base else ""
                        frases.append(f"{esc(d)}{sufijo}")
                    extra = f" Usa columnas calculadas: {self._unir_con_y(frases)}."
                items.append(f"<li><b>{nombre}</b>: {como}.{extra}</li>")
            partes.append("<h3>Tus indicadores</h3><ul>" + "".join(items) + "</ul>")

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Origen de los datos</h2>
          {"".join(partes)}
        </div>"""

    def _capitulo_anomalias(self, n):
        if not self.anomalias:
            cuerpo = "<p>No se detectaron anomalías con las reglas actuales.</p>"
        else:
            entradas = _agrupar_anomalias_por_columna(self.anomalias)
            entradas.sort(key=lambda e: not _entrada_es_gravedad_alta(e))

            items = []
            for tipo_entrada, valor in entradas:
                if tipo_entrada == "quiebre_grupo":
                    clase, cuerpo_html = _bloque_quiebre_grupo(
                        valor, self.df, modo_impresion=self.modo_impresion,
                        indicadores=self.indicadores, eventos=self.eventos_procedencia,
                    )
                    items.append(f"<div class='{clase}'>{cuerpo_html}</div>")
                    continue

                a = valor
                clase = "anomalia" if a["gravedad"] == "alta" else "anomalia media"
                grafico = _grafico_anomalia_narrativa(a, self.df)

                impacto_html = _linea_impacto_html(a)

                frase_memoria = a.get("frase_memoria")
                frase_html = f"<div class='frase-memoria'>{frase_memoria}</div>" if frase_memoria else ""

                frase_acumulado = a.get("frase_impacto_acumulado")
                frase_acumulado_html = (
                    f"<div class='frase-memoria'>{frase_acumulado}</div>" if frase_acumulado else ""
                )

                coincidencia_html = _linea_coincidencias_html(a, self.df)

                indicadores_html = _linea_indicadores_html(a.get("columnas"), self.indicadores)
                indicadores_html += _lineas_procedencia_anomalia_html(a.get("columnas"), self.eventos_procedencia)

                sugerencia_html = f"<div class='sugerencia'>💡 {sugerir_recomendacion(a)}</div>"

                codigo = _codificar_identidad_anomalia(a)
                enlace_html = (
                    f"<div class='resolver-link'>"
                    f"<a href='resolver:{codigo}'>Marcar como resuelta</a>"
                    f" · <a href='falso_positivo:{codigo}'>Marcar como no es una anomalía</a>"
                    f"</div>"
                    if (codigo and not self.modo_impresion) else ""
                )

                items.append(
                    f"<div class='{clase}'>{a['descripcion']}{impacto_html}{frase_html}"
                    f"{frase_acumulado_html}{coincidencia_html}{indicadores_html}{sugerencia_html}{grafico}{enlace_html}</div>"
                )
            cuerpo = "".join(items)

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Lo que no encaja</h2>
          {cuerpo}
        </div>"""

    def _capitulo_otras_tablas(self, n):
        """Resumen corto de anomalías en las DEMÁS tablas de la base
        relacional (el detalle fila a fila de cada una vive en la pestaña
        Datos, cambiando el selector de tabla activa; acá es solo un
        panorama para no repetir el informe completo N veces)."""
        items = []
        for nombre_tabla, info in self.tablas_relacionadas.items():
            df_t = info["df"]
            anomalias_t = info["anomalias"]
            n_filas = len(df_t)
            if not anomalias_t:
                items.append(
                    f"<p><b>{nombre_tabla}</b> ({n_filas:,} filas): no se detectaron anomalías.</p>"
                )
                continue
            n_altas = sum(1 for a in anomalias_t if a["gravedad"] == "alta")
            n_medias = len(anomalias_t) - n_altas
            detalle_gravedad = []
            if n_altas:
                detalle_gravedad.append(f"{n_altas} de gravedad alta")
            if n_medias:
                detalle_gravedad.append(f"{n_medias} de gravedad media")
            items.append(
                f"<p><b>{nombre_tabla}</b> ({n_filas:,} filas): {len(anomalias_t)} anomalía(s) "
                f"({', '.join(detalle_gravedad)}).</p>"
            )

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Tus otras tablas</h2>
          {"".join(items)}
        </div>"""

    def _capitulo_cruces(self, n):
        """El corazón del análisis relacional: usa las relaciones del
        esquema para ver si las filas anómalas de una tabla 'hija' se
        concentran en ciertos valores de la tabla 'padre' (ej. ciertos
        clientes concentran más pedidos anómalos que otros). Es conteo
        real sobre los datos, nunca una inferencia de causalidad."""
        if not self.relaciones:
            cuerpo = (
                "<p class='pendiente'>Hadar no tiene relaciones confirmadas entre tus tablas "
                "todavía. Abre \"Ver esquema\" en la pestaña Datos para revisarlas.</p>"
            )
            return f"""
            <div class="capitulo">
              <h2>Capítulo {n} · Cómo se cruzan tus tablas</h2>
              {cuerpo}
            </div>"""

        todas_las_tablas = dict(self.tablas_relacionadas)
        if self.nombre_tabla:
            todas_las_tablas[self.nombre_tabla] = {"df": self.df, "anomalias": self.anomalias}

        hallazgos = []
        for r in self.relaciones:
            principal = getattr(r, "tabla_principal", None)
            if not principal:
                continue
            if principal == r.tabla_origen:
                tabla_padre, col_padre = r.tabla_origen, r.columna_origen
                tabla_hija, col_hija = r.tabla_destino, r.columna_destino
            else:
                tabla_padre, col_padre = r.tabla_destino, r.columna_destino
                tabla_hija, col_hija = r.tabla_origen, r.columna_origen

            if tabla_padre not in todas_las_tablas or tabla_hija not in todas_las_tablas:
                continue

            df_padre = todas_las_tablas[tabla_padre]["df"]
            df_hija = todas_las_tablas[tabla_hija]["df"]
            anomalias_hija = todas_las_tablas[tabla_hija]["anomalias"]
            if col_hija not in df_hija.columns or col_padre not in df_padre.columns:
                continue

            conteo_por_clave = {}
            for a in anomalias_hija:
                filas = a.get("indices_atipicos") or []
                for fila in filas:
                    if fila not in df_hija.index:
                        continue
                    clave = df_hija.loc[fila, col_hija]
                    if pd.isna(clave):
                        continue
                    conteo_por_clave[clave] = conteo_por_clave.get(clave, 0) + 1

            if not conteo_por_clave:
                continue

            top = sorted(conteo_por_clave.items(), key=lambda kv: kv[1], reverse=True)[:3]
            col_nombre_padre = self._columna_nombre_amigable(df_padre, col_padre)

            lineas = []
            for clave, cuenta in top:
                etiqueta = self._etiqueta_amigable(df_padre, col_padre, clave, col_nombre_padre)
                lineas.append(
                    f"<li>{etiqueta}: {cuenta} fila(s) con anomalías en '{tabla_hija}'.</li>"
                )

            hallazgos.append(f"""
            <div class="anomalia media">
              <b>{tabla_padre}</b> <-> <b>{tabla_hija}</b> (por '{col_padre}' <-> '{col_hija}'):
              <ul>{"".join(lineas)}</ul>
            </div>""")

        if not hallazgos:
            cuerpo = (
                "<p>No se encontraron filas anómalas que se concentren en algún valor "
                "en particular a través de las relaciones detectadas.</p>"
            )
        else:
            cuerpo = "".join(hallazgos)

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Cómo se cruzan tus tablas</h2>
          {cuerpo}
        </div>"""

    def _capitulo_contagio(self, n):
        """Índice de Contagio: a diferencia del capítulo anterior (que
        agrupa anomalías de una tabla HIJA por su tabla padre), acá el
        punto de partida es cada anomalía real y se sigue la relación
        hacia ADELANTE (de la tabla principal hacia sus tablas hijas) para
        ver cuántas filas dependientes quedan infectadas por cascada.
        Responde "si arreglo esta fila, ¿cuánto sano aguas abajo?". Es
        puro recorrido de relaciones ya confirmadas + conteo real de
        filas -- no hay nada probabilístico ni inventado. Devuelve None
        (sin capítulo, sin gastar número) si no hay relaciones confirmadas
        -- ese caso ya se explica en el capítulo de cruces."""
        if not self.relaciones:
            return None

        tablas_df = {nombre: info["df"] for nombre, info in self.tablas_relacionadas.items()}
        anomalias_por_tabla = {nombre: info["anomalias"] for nombre, info in self.tablas_relacionadas.items()}
        if self.nombre_tabla:
            tablas_df[self.nombre_tabla] = self.df
            anomalias_por_tabla[self.nombre_tabla] = self.anomalias

        hallazgos = []
        for tabla_origen, anomalias_tabla in anomalias_por_tabla.items():
            columna_clave = columna_clave_de_tabla(tabla_origen, self.relaciones)
            if not columna_clave:
                continue  # esta tabla no es "principal" de ninguna relación: no contagia a nadie
            for a in anomalias_tabla:
                indices = a.get("indices_atipicos") or []
                if not indices:
                    continue
                resultado = calcular_indice_contagio(
                    tabla_origen=tabla_origen,
                    indices_afectados=indices,
                    columna_clave=columna_clave,
                    tablas=tablas_df,
                    relaciones=self.relaciones,
                )
                if resultado:
                    hallazgos.append((tabla_origen, a, resultado))

        if not hallazgos:
            cuerpo = (
                "<p>Ninguna de las anomalías detectadas se propaga hacia otras tablas "
                "a través de las relaciones confirmadas.</p>"
            )
        else:
            items = []
            for tabla_origen, a, resultado in hallazgos:
                resumen = resumen_en_texto(resultado)
                items.append(
                    f"<div class='anomalia media'>"
                    f"<b>{tabla_origen}</b>: {a['descripcion']}"
                    f"<div class='frase-memoria'>{resumen}</div>"
                    f"</div>"
                )
            cuerpo = "".join(items)

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Índice de Contagio</h2>
          <p>Cuando una fila con un error está conectada a otras tablas, ese error no se queda ahí:
             se hereda en cada fila que depende de ella. Esto muestra, para cada anomalía real,
             cuántas filas de otras tablas quedarían resueltas si se corrige la fuente del error.</p>
          {cuerpo}
        </div>"""

    def _capitulo_correlaciones(self, n):
        """Relaciones entre variables. Se arma en secciones, cada una una
        mirada distinta y complementaria sobre la misma pregunta -- ninguna
        de las tres dice nunca "causa", todas dejan ese límite explícito:

        1. Correlaciones numéricas (Fase 1): ¿qué columnas se mueven juntas?
        2. Fallas que ocurren juntas (Idea 1): ¿qué anomalías coinciden más
           de lo que el azar explicaría?
        3. Cambios de comportamiento (Idea 2): ¿alguna columna cambió su
           comportamiento típico en algún punto del tiempo? (requiere una
           columna de fecha; si no hay ninguna, esta sección no aparece)

        El capítulo entero se salta (sin gastar número) solo si NINGUNA de
        las tres secciones tiene siquiera con qué intentar calcular algo.
        Si al menos una sí puede calcularse, el capítulo aparece, y cada
        sección por separado muestra sus hallazgos o un mensaje honesto de
        que no encontró nada -- nunca se inventa ni se fuerza un resultado."""
        columnas_numericas = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_con_anomalias = columnas_con_anomalias_suficientes(self.anomalias)
        columnas_fecha = detectar_columnas_fecha(self.df)

        puede_correlaciones = len(columnas_numericas) >= 2
        puede_coocurrencia = len(columnas_con_anomalias) >= 2
        puede_regimen = bool(columnas_fecha) and len(columnas_numericas) >= 1

        if not (puede_correlaciones or puede_coocurrencia or puede_regimen):
            return None

        secciones = []

        if puede_correlaciones:
            resultados = detectar_correlaciones_significativas(self.df)
            if resultados:
                cuerpo = "<ul>" + "".join(f"<li>{c.como_texto()}</li>" for c in resultados) + "</ul>"
            else:
                cuerpo = (
                    "<p>No se encontraron correlaciones lo bastante fuertes y "
                    "estadísticamente significativas entre las columnas numéricas.</p>"
                )
            secciones.append(f"""
              <h3>Correlaciones numéricas</h3>
              <p>Qué columnas numéricas tienden a moverse juntas, con respaldo estadístico
                 real -- el umbral se ajusta según cuántos pares se comparan, para no
                 confundir azar con patrón.</p>
              {cuerpo}
            """)

        if puede_coocurrencia:
            resultados_co = detectar_coocurrencia_de_anomalias(len(self.df), self.anomalias)
            if resultados_co:
                cuerpo_co = "<ul>" + "".join(f"<li>{c.como_texto()}</li>" for c in resultados_co) + "</ul>"
            else:
                cuerpo_co = (
                    "<p>Las anomalías detectadas en esta tabla no coinciden entre sí más "
                    "de lo que el azar explicaría.</p>"
                )
            secciones.append(f"""
              <h3>Fallas que ocurren juntas</h3>
              <p>Esto no mira los valores normales, sino las filas marcadas como atípicas:
                 ¿hay columnas cuyas fallas coinciden en las mismas filas más seguido de lo
                 esperable? Son focos de colapso conjunto, no columnas parecidas.</p>
              {cuerpo_co}
            """)

        if puede_regimen:
            columna_fecha = columnas_fecha[0]
            quiebres = detectar_quiebres_de_comportamiento(self.df, columna_fecha)
            if quiebres:
                cuerpo_q = "<ul>" + "".join(f"<li>{q.como_texto()}</li>" for q in quiebres) + "</ul>"
            else:
                cuerpo_q = (
                    f"<p>Ninguna columna numérica cambió su comportamiento típico de forma "
                    f"estadísticamente significativa a lo largo de '{columna_fecha}'.</p>"
                )
            secciones.append(f"""
              <h3>Cambios de comportamiento en el tiempo</h3>
              <p>Ordenando por '{columna_fecha}', esto busca el punto donde el comportamiento
                 típico de una columna cambió de forma más marcada -- no una tendencia
                 gradual, sino un quiebre real.</p>
              {cuerpo_q}
            """)

        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Relaciones entre variables</h2>
          <p class="pendiente">Ninguna sección de este capítulo afirma causalidad: que dos
             cosas se muevan juntas, fallen juntas, o cambien en el tiempo no dice que una
             provoque a la otra, ni descarta que ambas respondan a un tercer factor que no
             está en esta tabla.</p>
          {"".join(secciones)}
        </div>"""

    @staticmethod
    def _columna_nombre_amigable(df_padre, col_padre):
        """Busca una columna de texto tipo 'nombre' en la tabla padre, para
        mostrar 'Carla' en vez de solo el id 3. Si no encuentra ninguna,
        devuelve None (se muestra el id tal cual)."""
        for col in df_padre.columns:
            if col == col_padre:
                continue
            n = str(col).strip().lower()
            if "nombre" in n or n == "name":
                return col
        return None

    @staticmethod
    def _etiqueta_amigable(df_padre, col_padre, clave, col_nombre_padre):
        base = f"{col_padre} = {clave}"
        if col_nombre_padre is None:
            return base
        try:
            coincidencias = df_padre.loc[df_padre[col_padre] == clave, col_nombre_padre]
            if not coincidencias.empty and pd.notna(coincidencias.iloc[0]):
                return f"{coincidencias.iloc[0]} ({base})"
        except Exception:
            pass
        return base

    def _capitulo_pendiente(self, titulo, mensaje):
        return f"""
        <div class="capitulo">
          <h2>{titulo}</h2>
          <p class="pendiente">{mensaje}</p>
        </div>"""

    def _capitulo_preguntas(self, n):
        preguntas = QuestionAssistant(self.df, self.anomalias).suggest()
        if not preguntas:
            cuerpo = "<p>No hay preguntas sugeridas por ahora.</p>"
        else:
            cuerpo = "<ul>" + "".join(f"<li class='pregunta'>{p}</li>" for p in preguntas) + "</ul>"
        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Preguntas que deberías hacerte</h2>
          {cuerpo}
        </div>"""

    def _capitulo_recomendaciones(self, n):
        tipos_presentes = {a["tipo"] for a in self.anomalias}
        recomendaciones = []
        if "temporal" in tipos_presentes:
            recomendaciones.append("Revisar el proceso de captura de fechas donde se detectó inconsistencia.")
        if "quiebre_patron" in tipos_presentes:
            recomendaciones.append("Investigar qué ocurrió en los registros más recientes marcados como atípicos.")
        if "regla_negocio" in tipos_presentes:
            recomendaciones.append("Revisar los casos que no cumplen las reglas de negocio definidas.")
        if "deriva_historica" in tipos_presentes:
            recomendaciones.append(
                "Este proyecto tiene aprendizaje continuo activado: algo se salió del comportamiento "
                "histórico propio de estos datos, no solo de un umbral genérico."
            )
        if "patron_multivariado" in tipos_presentes:
            recomendaciones.append(
                "Algunas filas fueron marcadas por una combinación poco común de columnas, "
                "no por un valor extremo en una sola -- revísalas con ese criterio en mente."
            )
        if not recomendaciones:
            recomendaciones.append("No se detectaron problemas relevantes con las reglas actuales.")

        cuerpo = "<ul>" + "".join(f"<li>{r}</li>" for r in recomendaciones) + "</ul>"
        return f"""
        <div class="capitulo">
          <h2>Capítulo {n} · Recomendaciones</h2>
          {cuerpo}
        </div>"""


# ----------------------------------------------------------------------------
# STUB explícito de v2 (interfaz definida, implementación pendiente a
# propósito -- no se improvisa un resultado con menos rigor del necesario).
#
# CounterfactualGenerator (simular "qué hubiera pasado") se sacó de acá:
# exigía elegir una métrica objetivo genérica y mostrar un rango de
# incertidumbre (Monte Carlo) que el usuario tendría que saber interpretar
# -- decisión consciente de reemplazarlo por algo más simple y verificable
# (severidad + impacto acumulado + sugerencia, ver más arriba en este
# archivo y el Capítulo "Lo que no encaja" en NarrativeGenerator).
# ----------------------------------------------------------------------------
class CausalMapGenerator:
    """STUB v2. Pendiente: detección de columna de fecha real, chequeo de
    estacionariedad, y usar lenguaje de precedencia temporal en vez de
    'causalidad' mientras no haya el respaldo estadístico correspondiente."""

    def __init__(self, df):
        self.df = df

    def build(self):
        raise NotImplementedError(
            "CausalMapGenerator es un stub de v2: pendiente detección de fecha, "
            "chequeo de estacionariedad y decidir el lenguaje correcto para no "
            "implicar causalidad real."
        )