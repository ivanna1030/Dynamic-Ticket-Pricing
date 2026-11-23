# Dynamic-Ticket-Pricing

Proyecto final de **Comportamientos en las Finanzas y Toma de Decisiones** (ITESO).
Un promotor vende C = 2,000 boletos durante T = 60 días; cada día elige un precio de la
malla $40–$200 (pasos de $10) para maximizar el ingreso esperado. El problema se modela con
el marco de Powell y se comparan tres políticas en el mismo simulador.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Todos los parámetros (`Parametros`). Si existe `datos/calibracion.json`, toma de ahí la forma de la curva de llegadas (potencia o exponencial) y su parámetro b. |
| `simulador.py` | Modelo de Powell: escenarios con números aleatorios comunes, transición (`un_dia`), `simular` y `resumen` (media, desviación, p5/p95, IC 95 %, probabilidad de agotar). |
| `politicas.py` | Las tres políticas y sus funciones de afinación. |
| `calibracion.py` | Calibra λ₀(d) con el dataset de Kaggle *Flight Prices* (`itineraries.csv`): ajusta a·(d+1)^(−b) y a·exp(−b·d) y se queda con la de mayor R². |
| `paso2_politicas.py` | Afina las políticas (semilla 1) y guarda `resultados/parametros_afinados.json`. |

## Políticas (paso 2)

| Política | Clase | Parámetros afinables | Regla |
|---|---|---|---|
| Base | (línea base) | `p_fijo` | El mismo precio todos los días. |
| PFA | Policy Function Approximation | `p_base`, `a` | p = malla(p_base − a·(I/C − φ(d))), con φ(d) = fracción de compradores que faltan por llegar. |
| DLA | Direct Lookahead | `beta` | Cada noche resuelve un problema determinista (mochila de elección múltiple / MILP) con demanda esperada en los días que faltan y aplica solo el precio de hoy. |

El problema determinista del DLA se resuelve con relajación lagrangiana (milésimas de segundo) y se
valida contra el MILP exacto de HiGHS (`scipy.optimize.milp`, 0.2–0.4 s) con
`comparar_dla_con_milp`: brecha media de ingreso planeado ≈0.2 %.

## Cómo correrlo

```bash
pip install -r requirements.txt

# 1) Calibración (una sola vez; itineraries.csv en esta carpeta, NO se sube a Git)
python calibracion.py            # extrae las rutas y ajusta; tarda por el tamaño del archivo
python calibracion.py ajustar    # solo vuelve a ajustar con la muestra ya extraída (segundos)

# 2) Afinar políticas
python paso2_politicas.py
```

## Para la evaluación (paso 3)

```python
import json
from config import cargar_parametros
from simulador import generar_escenarios, simular, resumen
from politicas import politica_base, politica_pfa, politica_dla

par = cargar_parametros()
af = json.load(open("resultados/parametros_afinados.json"))

esc = generar_escenarios(1000, par, semilla=2)   # semilla DISTINTA a la de afinación (1), >= 500 escenarios
politicas = {
    "Base": politica_base(af["base"]["p_fijo"]),
    "PFA": politica_pfa(af["pfa"]["p_base"], af["pfa"]["a"]),
    "DLA": politica_dla(af["dla"]["beta"]),
}
res = {nombre: simular(pol, esc, par) for nombre, pol in politicas.items()}   # mismos escenarios
for nombre, r in res.items():
    print(nombre, resumen(r))

# Diferencia pareada (mismos escenarios) con su error estándar
dif = res["DLA"]["ingreso"] - res["Base"]["ingreso"]
print(dif.mean(), dif.std(ddof=1) / len(dif) ** 0.5)
```

`simular` regresa por escenario: `ingreso`, `vendidos`, `agotado`, `dia_agotado` y las
trayectorias `precio`, `ventas`, `inventario`, `referencia` (matrices escenario × día).

Los parámetros por defecto de demanda (WTP, θ, η, γ, κ, r₀, compradores totales) son
supuestos nuestros; solo la forma de λ₀(d) sale de la calibración.
