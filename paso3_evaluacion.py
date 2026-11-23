"""
Paso 3 · Evaluación final de las políticas.

Compara Base, PFA y DLA en el MISMO simulador y sobre los MISMOS escenarios (números
aleatorios comunes), con una semilla distinta a la de afinación y >= 500 escenarios.
Reporta media, desviación, p5/p95 e IC 95 % del ingreso, y la diferencia pareada contra Base.

    python paso3_evaluacion.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from config import cargar_parametros
from simulador import generar_escenarios, simular, resumen
from politicas import politica_base, politica_pfa, politica_dla

SEMILLA_EVAL = 2          # distinta a la de afinación (1)
N_EVAL = 1000             # >= 500

par = cargar_parametros()
with open("resultados/parametros_afinados.json", encoding="utf-8") as f:
    af = json.load(f)

assert SEMILLA_EVAL != af["semilla_afinacion"], "La semilla de evaluación debe ser distinta a la de afinación"
assert N_EVAL >= 500, "Se piden al menos 500 escenarios"

esc = generar_escenarios(N_EVAL, par, semilla=SEMILLA_EVAL)

politicas = {
    "Base": politica_base(af["base"]["p_fijo"]),
    "PFA": politica_pfa(af["pfa"]["p_base"], af["pfa"]["a"]),
    "DLA": politica_dla(af["dla"]["beta"]),
}

# Mismos escenarios para todas las políticas
res = {nombre: simular(pol, esc, par) for nombre, pol in politicas.items()}
tabla = {nombre: resumen(r) for nombre, r in res.items()}

print(f"Evaluación: {N_EVAL} escenarios, semilla {SEMILLA_EVAL} (afinación: semilla {af['semilla_afinacion']})\n")
print(f"{'Política':<6} {'Media':>10} {'Desv.':>9} {'p5':>10} {'p95':>10} {'IC 95 %':>23} {'P(agotar)':>10} {'Vendidos':>9}")
for nombre, r in tabla.items():
    ic = f"[{r['ic_inf']:,.0f}, {r['ic_sup']:,.0f}]"
    print(f"{nombre:<6} {r['ingreso_medio']:>10,.0f} {r['desviacion']:>9,.0f} {r['p5']:>10,.0f} "
          f"{r['p95']:>10,.0f} {ic:>23} {r['prob_agotar']:>10.2f} {r['vendidos_medio']:>9,.0f}")

# Diferencias pareadas (mismos escenarios): error estándar e IC 95 %
z = norm.ppf(0.975)
pares = [("PFA", "Base"), ("DLA", "Base"), ("DLA", "PFA")]
print("\nDiferencia pareada de ingreso (A - B)")
print(f"{'A - B':<10} {'Dif. media':>11} {'Err. est.':>10} {'IC 95 %':>23} {'% mejora':>9} {'A gana en':>10}")
difs = {}
for a, b in pares:
    d = res[a]["ingreso"] - res[b]["ingreso"]
    m, ee = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    pct = 100 * m / res[b]["ingreso"].mean()
    gana = (d > 0).mean()
    print(f"{a + ' - ' + b:<10} {m:>11,.0f} {ee:>10,.0f} {f'[{m - z * ee:,.0f}, {m + z * ee:,.0f}]':>23} "
          f"{pct:>8.1f}% {gana:>9.0%}")
    difs[f"{a}-{b}"] = {"dif_media": m, "error_estandar": ee, "ic_inf": m - z * ee, "ic_sup": m + z * ee,
                        "mejora_%": pct, "prop_escenarios_A_gana": gana}

Path("resultados").mkdir(exist_ok=True)
salida = {"semilla_evaluacion": SEMILLA_EVAL, "n_evaluacion": N_EVAL,
          "parametros_politicas": {k: v for k, v in af.items() if k in ("base", "pfa", "dla")},
          "resumen": tabla, "diferencias_pareadas": difs}
with open("resultados/evaluacion.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, indent=2, ensure_ascii=False, default=float)
print("\nGuardado en resultados/evaluacion.json")