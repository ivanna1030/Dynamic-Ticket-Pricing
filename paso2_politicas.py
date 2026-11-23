"""
Paso 2 · Políticas: afinación de parámetros.

Afina las tres políticas en escenarios de AFINACIÓN (semilla 1) y guarda los parámetros
en resultados/parametros_afinados.json para que la evaluación del paso 3 los use con
escenarios NUEVOS (otra semilla, al menos 500 escenarios).

    python paso2_politicas.py
"""

import json
import time
from pathlib import Path

import numpy as np

from config import cargar_parametros
from simulador import generar_escenarios, simular, resumen
from politicas import (afinar_base, afinar_pfa, afinar_dla, comparar_dla_con_milp,
                       politica_base, politica_pfa, politica_dla, resolver_dla)

SEMILLA_AFINAR = 1        # la evaluación del paso 3 debe usar OTRA semilla (p. ej. 2)
N_AFINAR = 300

par = cargar_parametros()
print("Parámetros:", par.como_dict(), "\n")

# 1) Escenarios de afinación: los mismos para todas las políticas (números aleatorios comunes)
esc = generar_escenarios(N_AFINAR, par, SEMILLA_AFINAR)

# 2) Afinar cada política
inicio = time.time()
p_fijo, tabla_base = afinar_base(esc, par)
theta_pfa, tabla_pfa = afinar_pfa(esc, par)
beta, tabla_dla = afinar_dla(esc, par)
print(f"Afinación en {time.time() - inicio:.0f} s con {N_AFINAR} escenarios (semilla {SEMILLA_AFINAR})\n")


def mostrar(nombre, tabla, k=5):
    print(nombre)
    for fila in tabla[:k]:
        print(f"   theta = {fila['theta']}:  ingreso medio = {fila['ingreso_medio']:,.0f} "
              f"± {fila['error_estandar']:,.0f}   P(agotar) = {fila['prob_agotar']:.2f}")
    print()


mostrar("Base (precio fijo), mejores 5:", tabla_base)
mostrar("PFA (p_base, a), mejores 5:", tabla_pfa)
mostrar("DLA (beta):", tabla_dla, k=len(tabla_dla))

# 3) Validación del método rápido del DLA contra el MILP exacto
val = comparar_dla_con_milp(100, par, semilla=0, beta=beta)
print("DLA rápido vs. MILP (100 estados al azar):")
print(f"   brecha de ingreso planeado: media {val['brecha_media_%']:.2f} %, máxima {val['brecha_max_%']:.2f} %")
print(f"   tiempo por solución: MILP {val['seg_milp']:.3f} s, lagrangiano {val['seg_lagrangiano']:.4f} s\n")

# 4) Guardar para el paso 3
Path("resultados").mkdir(exist_ok=True)
afinados = {
    "semilla_afinacion": SEMILLA_AFINAR,
    "n_afinacion": N_AFINAR,
    "base": {"p_fijo": p_fijo},
    "pfa": {"p_base": theta_pfa[0], "a": theta_pfa[1]},
    "dla": {"beta": beta},
    "validacion_dla": val,
}
with open("resultados/parametros_afinados.json", "w", encoding="utf-8") as f:
    json.dump(afinados, f, indent=2, ensure_ascii=False)
print("Guardado en resultados/parametros_afinados.json")

# 5) Ejemplo: precio recomendado HOY (día 0) por cada política
estado0 = {"I": np.array([float(par.C)]), "d": par.T, "r": np.array([par.r0]),
           "H": np.zeros((1, 7)), "M": np.zeros(1)}
_, plan, _ = resolver_dla(estado0["I"], estado0["r"], par.T, par, beta)
print("\nPrecio del día 0:")
print("   Base:", politica_base(p_fijo)(estado0, par)[0])
print("   PFA: ", politica_pfa(*theta_pfa)(estado0, par)[0])
print("   DLA: ", politica_dla(beta)(estado0, par)[0], " | plan completo del DLA hoy:", plan[0].astype(int).tolist())
