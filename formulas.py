"""
Evaluador seguro de fórmulas para el módulo "Excel: Calcular y Arrastrar"
(reemplaza a eval() con una lista blanca de operaciones vía AST).
"""
import ast
import operator

import numpy as np

# ----------------------------------------------------------------------------
# Evaluador seguro de fórmulas (reemplaza a eval()).
#
# eval() con __builtins__ vacío NO es una caja de arena real: cualquier
# objeto Python permite llegar de vuelta a __builtins__/os/sys encadenando
# atributos (ej. ().__class__.__bases__[0].__subclasses__()...), sin
# necesitar nunca la palabra "import". Es el vector de escape descrito en
# CWE-95 / el riesgo de inyección de código de OWASP, y aplica aunque las
# fórmulas solo las pueda escribir el propio usuario de la app: un archivo
# de datos armado a propósito, un futuro uso multiusuario, o un simple
# error de copiar/pegar podrían terminar ejecutando código arbitrario.
#
# En vez de intentar "bloquear lo peligroso" (lista negra, siempre
# incompleta), este evaluador recorre el AST de la fórmula y solo permite
# explícitamente un subconjunto reducido de operaciones -- lista blanca.
# Nunca se llama a eval()/exec()/compile() sobre la fórmula del usuario.

class FormulaSeguridadError(Exception):
    """La fórmula intentó usar algo que no está en la lista blanca permitida."""


_EVAL_OPERADORES_BINARIOS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_EVAL_OPERADORES_UNARIOS = {
    ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_,
}
_EVAL_OPERADORES_COMPARACION = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}

# Métodos de pandas Series/DataFrame permitidos dentro de una fórmula: todos
# de solo lectura (no modifican nada fuera del resultado calculado) y sin
# ninguna forma de alcanzar el sistema de archivos, red, u otros módulos.
_EVAL_METODOS_PERMITIDOS = {
    "sum", "mean", "median", "min", "max", "std", "var", "count", "nunique",
    "abs", "round", "fillna", "dropna", "isna", "notna", "isnull", "notnull",
    "cumsum", "cumprod", "diff", "pct_change", "shift", "astype", "clip",
    "rank", "unique", "value_counts", "quantile", "mode", "tolist",
}
_EVAL_ASTYPE_TIPOS_PERMITIDOS = {"int", "float", "str", "bool"}


class _EvaluadorFormulaSegura(ast.NodeVisitor):
    """Evalúa una expresión aritmética/lógica sobre un DataFrame sin usar
    eval()/exec(). `variables` son los únicos nombres que la fórmula puede
    referenciar (típicamente solo {"df": df}); `funciones` son las únicas
    funciones que se pueden llamar por nombre (SI, SUMA, etc.)."""

    def __init__(self, variables, funciones):
        self.variables = variables
        self.funciones = funciones

    def evaluar(self, expresion):
        try:
            arbol = ast.parse(expresion, mode="eval")
        except SyntaxError as exc:
            raise FormulaSeguridadError(f"Fórmula con error de sintaxis: {exc}")
        return self.visit(arbol.body)

    def visit(self, nodo):
        metodo = getattr(self, f"_visit_{type(nodo).__name__}", None)
        if metodo is None:
            raise FormulaSeguridadError(
                f"La fórmula usa algo que no está permitido ({type(nodo).__name__})."
            )
        return metodo(nodo)

    def _visit_Constant(self, nodo):
        if isinstance(nodo.value, (int, float, bool, str)) or nodo.value is None:
            return nodo.value
        raise FormulaSeguridadError("Tipo de valor no permitido en la fórmula.")

    def _visit_Name(self, nodo):
        if nodo.id in self.variables:
            return self.variables[nodo.id]
        raise FormulaSeguridadError(f"'{nodo.id}' no está permitido en una fórmula.")

    def _visit_BinOp(self, nodo):
        op = _EVAL_OPERADORES_BINARIOS.get(type(nodo.op))
        if op is None:
            raise FormulaSeguridadError("Operador no permitido en la fórmula.")
        return op(self.visit(nodo.left), self.visit(nodo.right))

    def _visit_UnaryOp(self, nodo):
        op = _EVAL_OPERADORES_UNARIOS.get(type(nodo.op))
        if op is None:
            raise FormulaSeguridadError("Operador no permitido en la fórmula.")
        return op(self.visit(nodo.operand))

    def _visit_BoolOp(self, nodo):
        valores = [self.visit(v) for v in nodo.values]
        resultado = valores[0]
        for v in valores[1:]:
            resultado = (resultado & v) if isinstance(nodo.op, ast.And) else (resultado | v)
        return resultado

    def _visit_Compare(self, nodo):
        izquierda = self.visit(nodo.left)
        resultado = None
        for op_nodo, comparador in zip(nodo.ops, nodo.comparators):
            op = _EVAL_OPERADORES_COMPARACION.get(type(op_nodo))
            if op is None:
                raise FormulaSeguridadError("Comparación no permitida en la fórmula.")
            derecha = self.visit(comparador)
            parcial = op(izquierda, derecha)
            resultado = parcial if resultado is None else (resultado & parcial)
            izquierda = derecha
        return resultado

    def _visit_Subscript(self, nodo):
        base = self.visit(nodo.value)
        indice_nodo = nodo.slice
        if isinstance(indice_nodo, ast.Index):  # compatibilidad Python < 3.9
            indice_nodo = indice_nodo.value
        indice = self.visit(indice_nodo)
        try:
            return base[indice]
        except FormulaSeguridadError:
            raise
        except Exception as exc:
            raise FormulaSeguridadError(f"No se pudo indexar en la fórmula: {exc}")

    def _visit_Attribute(self, nodo):
        if nodo.attr.startswith("_") or nodo.attr not in _EVAL_METODOS_PERMITIDOS:
            raise FormulaSeguridadError(f"El método '.{nodo.attr}()' no está permitido en fórmulas.")
        base = self.visit(nodo.value)
        try:
            return getattr(base, nodo.attr)
        except Exception as exc:
            raise FormulaSeguridadError(f"No se pudo usar '.{nodo.attr}': {exc}")

    def _visit_Call(self, nodo):
        if isinstance(nodo.func, ast.Name):
            if nodo.func.id not in self.funciones:
                raise FormulaSeguridadError(f"La función '{nodo.func.id}(...)' no está permitida.")
            funcion = self.funciones[nodo.func.id]
            nombre_metodo = None
        elif isinstance(nodo.func, ast.Attribute):
            nombre_metodo = nodo.func.attr
            funcion = self._visit_Attribute(nodo.func)
        else:
            raise FormulaSeguridadError("Ese tipo de llamada no está permitido en fórmulas.")

        argumentos = [self.visit(a) for a in nodo.args]
        palabras_clave = {kw.arg: self.visit(kw.value) for kw in nodo.keywords if kw.arg}

        if nombre_metodo == "astype":
            tipo_pedido = argumentos[0] if argumentos else None
            if tipo_pedido not in _EVAL_ASTYPE_TIPOS_PERMITIDOS:
                raise FormulaSeguridadError(
                    f"astype('{tipo_pedido}') no está permitido; solo: "
                    + ", ".join(sorted(_EVAL_ASTYPE_TIPOS_PERMITIDOS))
                )

        try:
            return funcion(*argumentos, **palabras_clave)
        except FormulaSeguridadError:
            raise
        except Exception as exc:
            raise FormulaSeguridadError(f"No se pudo calcular la fórmula: {exc}")

    def _visit_List(self, nodo):
        return [self.visit(e) for e in nodo.elts]

    def _visit_Tuple(self, nodo):
        return tuple(self.visit(e) for e in nodo.elts)


def _funciones_formula_seguras():
    """Funciones tipo Excel disponibles por nombre dentro de una fórmula
    (SI, SUMA, etc.). Son la única forma de invocar lógica de np/pd; los
    módulos np/pd en sí NUNCA se exponen como nombre dentro de la fórmula."""

    def _si(condicion, si_verdadero, si_falso):
        return np.where(condicion, si_verdadero, si_falso)

    def _suma(*valores):
        total = valores[0]
        for v in valores[1:]:
            total = total + v
        return total

    def _promedio(*valores):
        return _suma(*valores) / len(valores)

    def _minimo(*valores):
        resultado = valores[0]
        for v in valores[1:]:
            resultado = np.minimum(resultado, v)
        return resultado

    def _maximo(*valores):
        resultado = valores[0]
        for v in valores[1:]:
            resultado = np.maximum(resultado, v)
        return resultado

    return {
        "SI": _si,
        "SUMA": _suma,
        "PROMEDIO": _promedio,
        "MINIMO": _minimo,
        "MAXIMO": _maximo,
        "REDONDEAR": lambda valor, decimales=0: np.round(valor, decimales),
        "ABS": lambda valor: np.abs(valor),
    }


def evaluar_formula_segura(expresion, df):
    """Punto de entrada único para evaluar una fórmula de usuario sobre un
    DataFrame, usado tanto por Indicadores como por la Transformación Excel.
    Lanza FormulaSeguridadError si la fórmula intenta algo fuera de la lista
    blanca (nunca usa eval()/exec())."""
    evaluador = _EvaluadorFormulaSegura(
        variables={"df": df},
        funciones=_funciones_formula_seguras(),
    )
    return evaluador.evaluar(expresion)


def extraer_columnas_formula(expresion):
    """Recorre el AST de una fórmula SIN evaluarla y devuelve el conjunto de
    columnas de 'df' que referencia -- ej. para "SUMA(df['ventas'],
    df['costos'])" devuelve {'ventas', 'costos'}.

    Es la base de un linaje simple: "de qué columnas depende este
    indicador/fórmula", sin necesitar el DataFrame real ni ejecutar nada (así
    que nunca falla por datos faltantes, columnas que no existen todavía, o
    una división por cero -- es pura sintaxis). Sirve para responder, antes
    de calcular nada, preguntas como "si esta columna tiene anomalías, ¿qué
    indicadores la usan?" cruzando el resultado con lo que ya detecta
    anomalias.py.

    Reutiliza el mismo criterio de compatibilidad ast.Index/ast.Constant que
    usa _EvaluadorFormulaSegura._visit_Subscript, para no depender de la
    versión exacta de Python.
    """
    try:
        arbol = ast.parse(expresion, mode="eval")
    except SyntaxError as exc:
        raise FormulaSeguridadError(f"Fórmula con error de sintaxis: {exc}")

    columnas = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Subscript):
            continue
        if not (isinstance(nodo.value, ast.Name) and nodo.value.id == "df"):
            continue
        indice_nodo = nodo.slice
        if isinstance(indice_nodo, ast.Index):  # compatibilidad Python < 3.9
            indice_nodo = indice_nodo.value
        if isinstance(indice_nodo, ast.Constant) and isinstance(indice_nodo.value, str):
            columnas.add(indice_nodo.value)
    return columnas


# ----------------------------------------------------------------------------
# Pestaña Indicadores: el usuario arma sus propios indicadores con
# operaciones básicas o una fórmula simple, en vez de un KPI fijo específico
# de un solo tipo de negocio.
# ----------------------------------------------------------------------------