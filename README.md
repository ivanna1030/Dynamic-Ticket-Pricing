# Dynamic-Ticket-Pricing

Proyecto final de **Comportamientos en las Finanzas y Toma de Decisiones** (ITESO).
Un promotor vende $C = 2,000$ boletos durante $T = 60$ días; cada día elige un precio de la
malla $40–$200 (pasos de $10) para maximizar el ingreso esperado. El problema se modela con
el marco de Powell y se comparan tres políticas en el mismo simulador.

## Estructura del proyecto

```text
Dynamic-Ticket-Pricing/
├── calibracion.py                 # Calibra la curva de llegadas con datos de Kaggle
├── config.py                      # Parámetros del modelo y malla de precios
├── simulador.py                   # Simulador estocástico del marco de Powell
├── politicas.py                   # Políticas Base, PFA y DLA + funciones de afinación
├── paso2_politicas.py             # Ejecuta la afinación (semilla 1)
├── paso3_evaluacion.py            # Evaluación comparativa y pruebas pareadas (semilla 2)
├── dashboard.py                   # Panel interactivo en Streamlit para el promotor
├── requirements.txt               # Dependencias del proyecto
├── datos/
│   ├── calibracion.json           # Parámetros ajustados de λ₀(d) (forma y exponente b)
│   ├── calibracion.png            # Gráfica de ajuste potencia vs. exponencial
│   └── curva_llegadas.csv         # Muestra procesada de ventas por días restantes
└── resultados/
    ├── parametros_afinados.json   # Parámetros óptimos encontrados en el paso 2
    └── evaluacion.json            # Métricas finales e intervalos de confianza del paso 3
```

## Políticas (paso 2)

| Política | Clase | Parámetros afinables | Regla |
|---|---|---|---|
| Base | (línea base) | `p_fijo` | El mismo precio todos los días (`$120`). |
| PFA | Policy Function Approximation | `p_base`, `a` | $p = \text{malla}(p_{\text{base}} - a\cdot(I/C - \phi(d)))$, con $\phi(d)$ = fracción de compradores que faltan por llegar (`$120`, `225`). |
| DLA | Direct Lookahead | `beta` | Cada noche resuelve un problema determinista (mochila de elección múltiple / MILP) con demanda esperada en los días que faltan y aplica solo el precio de hoy (`1.0`). |

El problema determinista del DLA se resuelve con relajación lagrangiana (milésimas de segundo) y se
valida contra el MILP exacto de HiGHS (`scipy.optimize.milp`, 0.2–0.4 s) con
`comparar_dla_con_milp`: brecha media de ingreso planeado ≈ 0.14 %.

## Cómo correrlo

```bash
pip install -r requirements.txt

# 1) Calibración (una sola vez; itineraries.csv en esta carpeta, NO se sube a Git)
python calibracion.py            # extrae las rutas y ajusta; tarda por el tamaño del archivo
python calibracion.py ajustar    # solo vuelve a ajustar con datos/curva_llegadas.csv (segundos)

# 2) Afinar políticas (semilla 1 -> resultados/parametros_afinados.json)
python paso2_politicas.py

# 3) Evaluar políticas (1,000 escenarios, semilla 2 -> resultados/evaluacion.json)
python paso3_evaluacion.py

# 4) Abrir el dashboard interactivo
streamlit run dashboard.py
```