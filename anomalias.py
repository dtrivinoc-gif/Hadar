"""
Detector semántico de anomalías (SemanticAnomalyDetector): valores nulos,
duplicados, incumplimiento de reglas de negocio, inconsistencias de orden
temporal y rupturas de patrón (outliers). También el helper que traduce
las anomalías detectadas en notas por celda sobre la tabla de datos.
"""
import numpy as np
import pandas as pd

PROPORCION_MINIMA_FECHA_NARRATIVA = 0.8   # % de valores parseables como fecha
TAMANO_MUESTRA_DETECCION_FECHA = 3000     # ver detectar_columnas_fecha
                                            # para considerar una columna "de fecha"

# Pares de tokens que sugieren un orden temporal esperado (anterior -> posterior).
# Se buscan como substring del nombre de columna, sin importar mayúsculas ni
# el resto del nombre (ej. "Fecha de Pedido" / "fecha_entrega_real" calzan).
TOKENS_TEMPORALES_PARES = [
    ("pedido", "entrega", "pedido → entrega"),
    ("compra", "pago", "compra → pago"),
    ("solicitud", "aprobacion", "solicitud → aprobación"),
    ("emision", "vencimiento", "emisión → vencimiento"),
    ("inicio", "fin", "inicio → fin"),
    ("creacion", "cierre", "creación → cierre"),
    # Pares neutros al dominio, pensados para mantenimiento preventivo,
    # operaciones e inventario (Hadar no es solo para ventas):
    ("mantencion", "falla", "mantención → falla"),
    ("apertura", "cierre", "apertura → cierre"),
    ("ingreso", "egreso", "ingreso → egreso"),
    ("deteccion", "resolucion", "detección → resolución"),
    ("alta", "baja", "alta → baja"),
    ("instalacion", "reemplazo", "instalación → reemplazo"),
]


def _sin_acentos_narrativa(texto):
    reemplazos = str.maketrans("áéíóúñ", "aeioun")
    return texto.lower().translate(reemplazos)


def detectar_columnas_fecha(df, proporcion_minima=PROPORCION_MINIMA_FECHA_NARRATIVA):
    """Devuelve los nombres de columna que, al intentar convertirse a fecha,
    logran un mínimo de valores válidos. No asume ningún nombre en particular.

    Adivina con una MUESTRA, no con la columna completa: con 7+ millones de
    filas, intentar parsear cada valor (format="mixed" es el modo más lento
    de pandas, prueba formato por valor) se puede ir a decenas de segundos
    por columna. Una columna de fecha real es consistente de principio a
    fin, así que 3.000 valores alcanzan para la misma conclusión que
    escanear todo -- el ahorro es real y el riesgo de equivocarse, mínimo.
    """
    columnas_fecha = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            columnas_fecha.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # evita falsos positivos con columnas numéricas
        muestra = df[col].dropna()
        if muestra.empty:
            continue
        if len(muestra) > TAMANO_MUESTRA_DETECCION_FECHA:
            muestra = muestra.sample(TAMANO_MUESTRA_DETECCION_FECHA, random_state=0)
        parseado = pd.to_datetime(muestra, errors="coerce", format="mixed")
        if parseado.notna().mean() >= proporcion_minima:
            columnas_fecha.append(col)
    return columnas_fecha


def _buscar_columna_por_token_narrativa(columnas, token):
    """Busca, entre `columnas`, la primera cuyo nombre contenga `token`
    (comparación sin acentos ni mayúsculas)."""
    token_norm = _sin_acentos_narrativa(token)
    for col in columnas:
        if token_norm in _sin_acentos_narrativa(str(col)):
            return col
    return None


def _clasificar_severidad_impacto(z_abs, umbral_z):
    """Traduce qué tan lejos está un valor de lo típico (en unidades de
    z robusto) a una palabra simple -- el usuario nunca ve el número,
    solo la palabra. Los multiplicadores son relativos al umbral que ya
    configuraste (umbral_z), no un número fijo, para que se sigan viendo
    coherentes si en algún momento cambia lo estricto que es el detector."""
    if z_abs > umbral_z * 3:
        return "crítico"
    if z_abs > umbral_z * 2:
        return "grave"
    if z_abs > umbral_z * 1.3:
        return "moderado"
    return "leve"


class SemanticAnomalyDetector:
    """Detecta anomalías sin asumir el dominio del dataset:
      1. Reglas de negocio: motor listo, vacío por defecto (se completan desde
         la UI en una v2 -- aquí solo se deja la interfaz funcionando).
      2. Inconsistencias temporales: por tokens de nombre de columna + fechas.
      3. Quiebres de patrón: z-score de CADA valor de cada columna numérica
         contra el promedio y desviación estándar de toda la columna (no solo
         de las últimas filas -- una anomalía puede estar en cualquier parte
         del dataset).
      4. Valores nulos: columnas con una proporción relevante de datos
         faltantes (por encima de `umbral_minimo_nulos`, para no reportar
         un par de nulos sueltos como si fuera un problema).
      5. Duplicados: filas idénticas en todas sus columnas.
    """

    def __init__(self, df, umbral_z=3.0, minimo_muestras=10,
                 max_anomalias_individuales=5, reglas_negocio=None,
                 pares_temporales_personalizados=None, umbral_minimo_nulos=0.01):
        self.df = df
        self.umbral_z = umbral_z
        # Con menos muestras que esto, la desviación estándar no es confiable
        # y no se revisa esa columna (evita falsos positivos con pocos datos).
        self.minimo_muestras = minimo_muestras
        # Si una columna tiene más atípicos que esto, se agrupan en una sola
        # anomalía resumida en vez de listar cada fila (informe legible).
        self.max_anomalias_individuales = max_anomalias_individuales
        # Cada regla: {"nombre": str, "condicion": callable(df) -> Serie booleana,
        #              "descripcion": str}. Vacío por defecto a propósito.
        self.reglas_negocio = reglas_negocio or []
        # Relaciones "esta columna debe ir antes que esta otra" definidas a
        # mano por el usuario, con nombres de columna REALES (no tokens).
        # Complementan -- no reemplazan -- los pares automáticos por token de
        # TOKENS_TEMPORALES_PARES. Cada dict: {"col_anterior", "col_posterior",
        # "etiqueta"}.
        self.pares_temporales_personalizados = pares_temporales_personalizados or []
        # Por debajo de este % de nulos en una columna no se reporta nada:
        # uno o dos valores faltantes sueltos son normales, no una anomalía.
        self.umbral_minimo_nulos = umbral_minimo_nulos

    def detect_all(self):
        anomalias = []
        anomalias += self._check_business_rules()
        anomalias += self._check_temporal_logic()
        anomalias += self._check_pattern_breaks()
        anomalias += self._check_valores_nulos()
        anomalias += self._check_duplicados()
        return anomalias

    def _check_valores_nulos(self):
        anomalias = []
        total_filas = len(self.df)
        if total_filas == 0:
            return anomalias
        for col in self.df.columns:
            mascara_nulos = self.df[col].isna()
            n_nulos = int(mascara_nulos.sum())
            if n_nulos == 0:
                continue
            proporcion = n_nulos / total_filas
            if proporcion < self.umbral_minimo_nulos:
                continue  # un par de nulos sueltos no amerita reportarse
            porcentaje = round(proporcion * 100, 1)
            anomalias.append({
                "tipo": "valores_nulos",
                "columnas": [col],
                "descripcion": (
                    f"'{col}' tiene {n_nulos:,} valor(es) faltante(s) ({porcentaje}% de las filas)."
                ),
                "filas_afectadas": n_nulos,
                "porcentaje": porcentaje,
                "indices_atipicos": list(self.df.index[mascara_nulos]),
                "gravedad": "alta" if porcentaje > 15 else "media",
            })
        return anomalias

    def _check_duplicados(self):
        anomalias = []
        total_filas = len(self.df)
        if total_filas == 0:
            return anomalias
        mascara_dup = self.df.duplicated(keep=False)
        n_dup = int(mascara_dup.sum())
        if n_dup == 0:
            return anomalias
        porcentaje = round(n_dup / total_filas * 100, 1)
        indices_dup = list(self.df.index[mascara_dup])
        ejemplos = ", ".join(str(i + 1) for i in indices_dup[:5])  # +1: ver nota en _check_pattern_breaks
        anomalias.append({
            "tipo": "duplicados",
            "columnas": list(self.df.columns),
            "descripcion": (
                f"Hay {n_dup:,} fila(s) duplicada(s) exactamente en todas sus columnas "
                f"({porcentaje}% del total). Ejemplos de filas: {ejemplos}."
            ),
            "filas_afectadas": n_dup,
            "porcentaje": porcentaje,
            "indices_atipicos": indices_dup,
            "gravedad": "alta" if porcentaje > 2 else "media",
        })
        return anomalias

    def _check_business_rules(self):
        anomalias = []
        for regla in self.reglas_negocio:
            try:
                incumplidas = regla["condicion"](self.df)
                n = int(incumplidas.sum())
            except Exception:
                continue
            if n > 0:
                anomalias.append({
                    "tipo": "regla_negocio",
                    "columnas": [],
                    "descripcion": f"{regla.get('descripcion', regla.get('nombre', 'Regla'))}: "
                                   f"{n:,} fila(s) no la cumplen.",
                    "filas_afectadas": n,
                    "porcentaje": round(n / len(self.df) * 100, 1) if len(self.df) else 0,
                    "indices_atipicos": list(self.df.index[incumplidas]),
                    "gravedad": "alta" if n / max(len(self.df), 1) > 0.02 else "media",
                })
        return anomalias

    def _check_temporal_logic(self):
        anomalias = []
        columnas_fecha = detectar_columnas_fecha(self.df)
        if len(columnas_fecha) < 2:
            return anomalias

        pares_a_revisar = []  # lista de (col_anterior, col_posterior, etiqueta)
        vistos = set()

        for tok_anterior, tok_posterior, etiqueta in TOKENS_TEMPORALES_PARES:
            col_anterior = _buscar_columna_por_token_narrativa(columnas_fecha, tok_anterior)
            col_posterior = _buscar_columna_por_token_narrativa(columnas_fecha, tok_posterior)
            if not col_anterior or not col_posterior or col_anterior == col_posterior:
                continue
            pares_a_revisar.append((col_anterior, col_posterior, etiqueta))
            vistos.add((col_anterior, col_posterior))

        # Relaciones que el usuario definió a mano (nombres de columna reales,
        # elegidas directamente desde el diálogo "Configurar relaciones de
        # fechas"): cubren tanto nombres que no calzan con ningún token fijo
        # como relaciones de negocio que no están en TOKENS_TEMPORALES_PARES.
        for par in self.pares_temporales_personalizados:
            col_anterior = par.get("col_anterior")
            col_posterior = par.get("col_posterior")
            if not col_anterior or not col_posterior or col_anterior == col_posterior:
                continue
            if col_anterior not in self.df.columns or col_posterior not in self.df.columns:
                continue  # el dataset cambió y esa columna ya no existe
            if (col_anterior, col_posterior) in vistos:
                continue  # ya cubierto por un par automático, evita duplicar
            etiqueta = par.get("etiqueta") or f"{col_anterior} → {col_posterior}"
            pares_a_revisar.append((col_anterior, col_posterior, etiqueta))
            vistos.add((col_anterior, col_posterior))

        for col_anterior, col_posterior, etiqueta in pares_a_revisar:
            fechas_a = pd.to_datetime(self.df[col_anterior], errors="coerce", format="mixed")
            fechas_p = pd.to_datetime(self.df[col_posterior], errors="coerce", format="mixed")
            invalido = (fechas_p < fechas_a) & fechas_a.notna() & fechas_p.notna()
            n_invalido = int(invalido.sum())

            if n_invalido > 0:
                porcentaje = round(n_invalido / len(self.df) * 100, 1) if len(self.df) else 0
                anomalias.append({
                    "tipo": "temporal",
                    "columnas": [col_anterior, col_posterior],
                    "descripcion": (
                        f"En {n_invalido:,} fila(s) ({porcentaje}%), '{col_posterior}' ocurre "
                        f"antes que '{col_anterior}' ({etiqueta}), lo cual no es consistente."
                    ),
                    "filas_afectadas": n_invalido,
                    "porcentaje": porcentaje,
                    "indices_atipicos": list(self.df.index[invalido]),
                    "gravedad": "alta" if porcentaje > 2 else "media",
                })
        return anomalias

    def _check_pattern_breaks(self):
        """Busca valores atípicos en CADA columna numérica, recorriendo
        todas las filas -- no solo las últimas.

        Usa mediana + MAD (desviación absoluta mediana) en vez de promedio +
        desviación estándar: el promedio y la desviación estándar se dejan
        arrastrar por los mismos valores extremos que se está tratando de
        detectar (un solo valor disparado ya estira el promedio de toda la
        columna), así que terminan siendo una vara de medir poco confiable
        para medirse a sí misma. La mediana y el MAD no se mueven casi nada
        aunque haya un puñado de valores atípicos, así que dan una idea más
        honesta de "qué es lo normal acá". El factor 0.6745 es el que hace
        que este 'z robusto' sea comparable al umbral_z de siempre bajo una
        distribución normal -- no hay que tocar el umbral por este cambio.
        """
        anomalias = []
        num_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            serie = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(serie) < self.minimo_muestras:
                continue  # muy poca historia para calcular una desviación confiable

            mediana = serie.median()
            mad = (serie - mediana).abs().median()
            if not mad or pd.isna(mad):
                continue  # sin variación real en la columna: no hay quiebre que buscar

            z_robusto = 0.6745 * (serie - mediana) / mad
            atipicos = z_robusto[z_robusto.abs() > self.umbral_z]
            if atipicos.empty:
                continue

            gravedad = "alta" if atipicos.abs().max() > self.umbral_z * 1.5 else "media"

            if len(atipicos) <= self.max_anomalias_individuales:
                # Pocos atípicos: se listan uno por uno, con su fila exacta.
                # OJO: `idx` es el índice de pandas (empieza en 0); en el
                # TEXTO se muestra idx+1 para que coincida con el número de
                # fila que el usuario ve en la pestaña Datos (que numera
                # "posición + 1", ver table_model.py). El valor guardado en
                # 'fila_indice'/'indices_atipicos' NO se toca -- sigue siendo
                # el índice real de pandas, porque de eso depende que
                # "Marcar como resuelta" y el motor de contagio encuentren
                # la fila correcta.
                for idx, z in atipicos.items():
                    valor = serie.loc[idx]
                    diferencia_absoluta = valor - mediana
                    diferencia_porcentual = (
                        round(diferencia_absoluta / mediana * 100, 1) if mediana else None
                    )
                    anomalias.append({
                        "tipo": "quiebre_patron",
                        "columnas": [col],
                        "descripcion": (
                            f"En la fila {idx + 1}, '{col}' vale {valor:,.2f}, muy por "
                            f"{'encima' if z > 0 else 'debajo'} de lo típico "
                            f"({mediana:,.2f})."
                        ),
                        "filas_afectadas": 1,
                        "fila_indice": idx,
                        "indices_atipicos": [idx],
                        "gravedad": "alta" if abs(z) > self.umbral_z * 1.5 else "media",
                        "valor_tipico": mediana,
                        "diferencia_absoluta": diferencia_absoluta,
                        "diferencia_porcentual": diferencia_porcentual,
                        "severidad_impacto": _clasificar_severidad_impacto(abs(z), self.umbral_z),
                    })
            else:
                # Muchos atípicos: se agrupan en una sola anomalía para que el
                # informe siga siendo legible (no una fila por valor atípico).
                n_atipicos = len(atipicos)
                porcentaje = round(n_atipicos / len(self.df) * 100, 1) if len(self.df) else 0
                ejemplos = ", ".join(str(i + 1) for i in atipicos.index[:5])
                anomalias.append({
                    "tipo": "quiebre_patron",
                    "columnas": [col],
                    "descripcion": (
                        f"'{col}' tiene {n_atipicos:,} valor(es) atípico(s) "
                        f"({porcentaje}% de las filas), muy alejados de lo típico "
                        f"({mediana:,.2f}). Ejemplos de filas: {ejemplos}."
                    ),
                    "filas_afectadas": n_atipicos,
                    "porcentaje": porcentaje,
                    "indices_atipicos": list(atipicos.index),
                    "gravedad": gravedad,
                    "valor_tipico": mediana,
                    "severidad_impacto": _clasificar_severidad_impacto(
                        atipicos.abs().max(), self.umbral_z
                    ),
                })
        return anomalias


def _es_categorica_util(muestra):
    """Una columna 'sirve' para el chequeo de valor raro solo si de verdad
    repite valores -- una columna de fechas, IDs o texto libre donde casi
    todo es único haría que CUALQUIER fila pareciera "rara" (1 de N
    siempre es un porcentaje chico), lo cual no dice nada. Exige que haya
    como máximo 20 valores distintos, o un 20% del total (lo que sea más
    chico), y que efectivamente haya repetición (nunique > 1)."""
    total = len(muestra)
    return 1 < muestra.nunique() <= min(20, max(2, int(total * 0.2)))


def buscar_coincidencias(df, anomalia, max_coincidencias=1, z_minimo=2.0):
    """Busca si OTRAS columnas también tenían un valor fuera de lo común en
    las mismas filas donde se detectó esta anomalía -- ej. si un día con
    ventas muy bajas también fue un día con stock en cero. Nunca decide si
    "causó" nada: solo devuelve el hecho bruto (qué columna, qué tan
    atípico, qué valor). El lenguaje con el que se cuenta esto (siempre
    "coincidió con", nunca "por culpa de") lo decide narrativa.py.

    Usa un umbral más flojo (z_minimo=2.0) que el detector principal
    (umbral_z=3.0 por defecto) a propósito: acá no se trata de decidir si
    ESA columna tiene una anomalía propia, sino de si vale la pena
    mencionarla como posible pista -- una vara más floja es aceptable
    porque el texto final ya aclara que no es necesariamente la causa."""
    columnas_propias = set(anomalia.get("columnas") or [])
    indices = anomalia.get("indices_atipicos")
    if indices is None:
        idx_unico = anomalia.get("fila_indice")
        indices = [idx_unico] if idx_unico is not None else []
    indices = [i for i in indices if i in df.index][:20]  # tope por rendimiento
    if not indices:
        return []

    candidatas = []
    for col in df.columns:
        if col in columnas_propias:
            continue
        serie = df[col]

        if pd.api.types.is_datetime64_any_dtype(serie):
            continue  # una fecha "coincide consigo misma" siempre -- no es información

        if pd.api.types.is_numeric_dtype(serie):
            serie_num = pd.to_numeric(serie, errors="coerce").dropna()
            if len(serie_num) < 10:
                continue
            mediana = serie_num.median()
            mad = (serie_num - mediana).abs().median()
            if mad and not pd.isna(mad):
                for idx in indices:
                    if idx not in serie_num.index:
                        continue
                    z = 0.6745 * (serie_num.loc[idx] - mediana) / mad
                    if abs(z) >= z_minimo:
                        candidatas.append({
                            "columna": col, "valor": serie_num.loc[idx], "fuerza": abs(z),
                        })
                        break  # con una fila alcanza para mencionar esta columna
            elif _es_categorica_util(serie_num):
                # Columna casi constante (MAD = 0, ej. un stock que casi
                # siempre es el mismo número): un z-score no sirve acá, pero
                # sí sirve preguntar "¿el valor de esta fila es raro dentro
                # de esta columna?", igual que con una columna categórica.
                proporciones = serie_num.value_counts(normalize=True)
                for idx in indices:
                    if idx not in serie_num.index:
                        continue
                    valor = serie_num.loc[idx]
                    proporcion = proporciones.get(valor, 1.0)
                    if proporcion <= 0.05:
                        candidatas.append({
                            "columna": col, "valor": valor, "fuerza": 1 - proporcion,
                        })
                        break
        else:
            muestra = serie.dropna()
            if not _es_categorica_util(muestra):
                continue
            proporciones = muestra.value_counts(normalize=True)
            for idx in indices:
                if idx not in serie.index or pd.isna(serie.loc[idx]):
                    continue
                valor = serie.loc[idx]
                proporcion = proporciones.get(valor, 1.0)
                if proporcion <= 0.05:
                    candidatas.append({
                        "columna": col, "valor": valor, "fuerza": 1 - proporcion,
                    })
                    break

    candidatas.sort(key=lambda c: c["fuerza"], reverse=True)
    return candidatas[:max_coincidencias]


def anomalias_a_notas_celda(anomalias, df, limite_por_anomalia=300):
    """Traduce la lista de anomalías detectadas a notas de celda concretas
    (fila, columna, texto) para pintarlas en la pestaña Datos. Se apoya en
    'indices_atipicos' (o 'fila_indice' para el caso de una sola fila) y
    'columnas', que cada chequeo de SemanticAnomalyDetector ya deja listos
    en su diccionario -- así que agregar un chequeo nuevo alcanza con que
    también rellene esos dos campos para que aparezca marcado en Datos."""
    notas = []
    for a in anomalias:
        indices = a.get("indices_atipicos")
        if indices is None:
            idx_unico = a.get("fila_indice")
            indices = [idx_unico] if idx_unico is not None else []
        if not indices:
            continue
        columnas = [c for c in a.get("columnas", []) if c in df.columns]
        if not columnas:
            continue
        texto = a["descripcion"]
        for idx in indices[:limite_por_anomalia]:
            if idx not in df.index:
                continue
            for col in columnas:
                notas.append((idx, col, texto))
    return notas