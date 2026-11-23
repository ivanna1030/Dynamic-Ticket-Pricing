"""
Políticas de precio (paso 2 del proyecto).

Las tres políticas reciben SOLO el estado S_t = (I_t, d_t, r_t, H_t) y los parámetros
del modelo; ninguna ve los compradores del día. Cada una se construye con una función
que recibe sus parámetros afinables y regresa la regla politica(estado, par) -> precio.

  1. Base  (precio fijo)        parámetro: el precio p_fijo
  2. PFA   (regla de precios)   parámetros: (p_base, a)
  3. DLA   (lookahead MILP)     parámetro: beta (colchón de capacidad)

Al final están las funciones de afinación, que buscan el mejor parámetro de cada
política en un conjunto de escenarios (el de afinación, NO el de evaluación).
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.stats import norm

from simulador import precio_efectivo, simular, resumen


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def proyectar_a_malla(p, par):
    """Redondea un precio al valor más cercano de la malla P y lo recorta a [precio_min, precio_max]."""
    p = np.round((np.asarray(p, dtype=float) - par.precio_min) / par.paso_precio) * par.paso_precio + par.precio_min
    return np.clip(p, par.precio_min, par.precio_max)


def prob_compra(pef, par):
    """P(V >= Pef) con V ~ Lognormal(mu_V, sigma_V^2). Si Pef <= 0, compra seguro."""
    pef = np.asarray(pef, dtype=float)
    z = (np.log(np.maximum(pef, 1e-9)) - par.mu_V) / par.sigma_V
    return np.where(pef > 0, norm.sf(z), 1.0)


def fraccion_por_llegar(d, par):
    """phi(d): fracción de los compradores esperados de toda la ventana que todavía
    faltan por llegar cuando quedan d días. phi(T) = 1 y phi(0) = 0."""
    lam = par.tasa_base(np.arange(1, par.T + 1))          # lambda0(1), ..., lambda0(T)
    acumulada = np.concatenate([[0.0], np.cumsum(lam)])   # acumulada[d] = suma de lambda0(1..d)
    return acumulada[int(d)] / acumulada[-1]


# ----------------------------------------------------------------------
# 1) Política base: el mejor precio fijo
# ----------------------------------------------------------------------
def politica_base(p_fijo):
    """Cobra el mismo precio todos los días, sin importar el estado."""
    def regla(estado, par):
        return np.full(len(estado["I"]), float(p_fijo))
    return regla


# ----------------------------------------------------------------------
# 2) PFA: precio en función de días que faltan y asientos que quedan
# ----------------------------------------------------------------------
def politica_pfa(p_base, a):
    """Regla de ritmo de venta.

    brecha_t = I_t / C - phi(d_t)
      I_t / C  = fracción de asientos que quedan
      phi(d_t) = fracción de compradores que todavía faltan por llegar
    Si brecha > 0 quedan más asientos de los que 'tocan' a esta altura: vamos atrasados
    y se baja el precio. Si brecha < 0 vamos adelantados y se sube.

      p_t = redondear_a_malla( p_base - a * brecha_t )

    p_base (USD): precio cuando la venta va justo al ritmo esperado.
    a (USD):      cuánto se mueve el precio por cada unidad de brecha
                  (a = 200 y brecha = 0.10 -> el precio baja 20 USD).
    """
    def regla(estado, par):
        brecha = estado["I"] / par.C - fraccion_por_llegar(estado["d"], par)
        return proyectar_a_malla(p_base - a * brecha, par)
    return regla


# ----------------------------------------------------------------------
# 3) DLA: cada día resuelve un problema determinista con demanda esperada
# ----------------------------------------------------------------------
def demanda_esperada(I, r, d, par):
    """Matriz de demanda esperada D[s, k, j] para el lookahead.

    s = escenario, k = 0, ..., d-1 días hacia adelante, j = precio de la malla.
    D[s,k,j] = factor_escasez * lambda0(d - k) * P(V >= Pef(p_j, r_t))

    Supuestos del modelo determinista (es una aproximación del modelo real):
      - Se usa la demanda ESPERADA (sin aleatoriedad).
      - El precio de referencia se queda en su valor de hoy, r_t, en todos los días del
        lookahead. Como el problema se vuelve a resolver cada noche, r_t se actualiza
        con lo que realmente se cobró. Con esto cada día se trata por separado y el
        problema queda lineal.
        (Probamos también suponer que la referencia se mueve hacia el precio que se
        cobra, r_k = theta^k r_t + (1 - theta^k) p_j, y salió peor: ese supuesto es
        optimista con precios que suben, porque ignora que la referencia viene de los
        precios bajos de días anteriores.)
      - La escasez se aplica solo si hoy ya hay escasez (I_t < 10 % de C).
    """
    P = par.malla()
    k = np.arange(d)
    lam = par.tasa_base(d - k)                                        # (K,)
    r = np.asarray(r, dtype=float)
    pef = precio_efectivo(P[None, None, :], r[:, None, None], par)   # (n, 1, J): igual todos los días
    factor = np.where(np.asarray(I) < par.umbral_escasez * par.C, par.kappa, 1.0)
    return factor[:, None, None] * lam[None, :, None] * prob_compra(pef, par)   # (n, K, J)


def resolver_dla(I, r, d, par, beta=1.0, iteraciones=40, orden="final"):
    """Resuelve el problema determinista del DLA para muchos estados a la vez.

    Problema (para cada escenario):
        max   sum_k sum_j p_j y_kj
        s.a.  sum_j x_kj = 1                 un solo precio por día
              0 <= y_kj <= D_kj x_kj         no se vende más que la demanda esperada
              sum_k sum_j y_kj <= beta I_t   capacidad que queda (con colchón beta)
              x_kj en {0, 1}

    Es un problema de mochila de elección múltiple. Se resuelve con relajación
    lagrangiana: a la capacidad se le pone un 'precio sombra' mu >= 0; dado mu, cada día
    se elige el precio que maximiza (p_j - mu) D_kj, y mu se ajusta por bisección hasta
    que la venta planeada cabe en la capacidad. Después se llena la capacidad que sobra
    cambiando algunos días al siguiente precio más bajo (como haría el MILP).

    Cuando hay empate (el óptimo mezcla dos precios y da igual en qué días va cada uno),
    orden="final" pone el precio alto en los primeros días y deja el más bajo para el
    final: se cobra caro mientras hay tiempo y se guarda el precio bajo para cuando ya se
    sabe cómo va la venta. En el simulador esto dio más ingreso que orden="inicio".

    Es mucho más rápido que llamar al MILP 30,000 veces (cada MILP tarda entre 0.2 y 0.4 s, según la computadora);
    `comparar_dla_con_milp` lo valida contra el MILP exacto (HiGHS).

    Regresa (precio de hoy, plan de precios para los d días, mu) por escenario.
    """
    P = par.malla()
    I = np.asarray(I, dtype=float)
    D = demanda_esperada(I, r, d, par)                              # (n, K, J)
    cap = beta * I

    def plan(mu):
        valor = (P[None, None, :] - mu[:, None, None]) * D
        j = valor.argmax(axis=2)                                    # (n, K) precio elegido cada día
        ventas = np.take_along_axis(D, j[..., None], axis=2)[..., 0].sum(axis=1)
        return j, ventas

    n = len(I)
    lo, hi = np.zeros(n), np.full(n, float(P.max()))
    j0, v0 = plan(lo)
    cabe_sin_mu = v0 <= cap                                         # la capacidad no limita: mu = 0
    for _ in range(iteraciones):
        mid = 0.5 * (lo + hi)
        _, v = plan(mid)
        demasiado = v > cap
        lo = np.where(demasiado, mid, lo)
        hi = np.where(demasiado, hi, mid)
    mu = np.where(cabe_sin_mu, 0.0, hi)                             # hi siempre cumple la capacidad
    j_alto, v_alto = plan(mu)                                       # plan factible (vende de menos)
    j_bajo, _ = plan(lo)                                            # plan que se pasa un poco

    # Recuperar una solución entera que use la capacidad como el MILP: se cambia al
    # precio más bajo día por día (en el orden indicado) mientras siga cabiendo.
    D_alto = np.take_along_axis(D, j_alto[..., None], axis=2)[..., 0]
    D_bajo = np.take_along_axis(D, j_bajo[..., None], axis=2)[..., 0]
    extra = D_bajo - D_alto                                         # ventas extra por bajar ese día
    dias = np.arange(D.shape[1]) if orden == "inicio" else np.arange(D.shape[1])[::-1]
    sobra = np.where(cabe_sin_mu, 0.0, cap - v_alto)
    j = j_alto.copy()
    sigue = np.ones(len(I), dtype=bool)                             # se detiene en el primer día que ya no cabe
    for k in dias:
        cabe = extra[:, k] <= sobra
        cambia = sigue & cabe & (extra[:, k] > 0)
        j[cambia, k] = j_bajo[cambia, k]
        sobra = sobra - np.where(cambia, extra[:, k], 0.0)
        sigue = sigue & cabe
    plan_precios = P[j]
    return plan_precios[:, 0], plan_precios, mu


def resolver_dla_milp(I, r, d, par, beta=1.0):
    """El mismo problema determinista, pero resuelto como MILP con HiGHS
    (scipy.optimize.milp). Es para UN estado; se usa para validar `resolver_dla`
    y para mostrar el plan en el dashboard.
    Regresa (precio de hoy, plan de precios, ingreso planeado)."""
    P = par.malla()
    D = demanda_esperada(np.array([I]), np.array([r]), d, par)[0]  # (K, J)
    K, J = D.shape
    nv = K * J                                                      # variables: x (K*J) y y (K*J)
    # milp minimiza, así que se usa -ingreso. El término 1e-4 * p en x solo desempata:
    # si un día no se planea vender nada extra, prefiere el precio más alto (si no, el MILP
    # podría poner 40 USD en un día "sin ventas", lo cual no tiene sentido en la práctica).
    c = np.concatenate([-1e-4 * np.tile(P, K), -np.tile(P, K)])

    filas_un_precio = np.zeros((K, 2 * nv))
    for k in range(K):
        filas_un_precio[k, k * J:(k + 1) * J] = 1
    filas_demanda = np.hstack([-np.diag(D.ravel()), np.eye(nv)])    # y - D x <= 0
    fila_cap = np.concatenate([np.zeros(nv), np.ones(nv)])[None, :]

    restricciones = [
        LinearConstraint(filas_un_precio, 1, 1),
        LinearConstraint(filas_demanda, -np.inf, 0),
        LinearConstraint(fila_cap, -np.inf, beta * I),
    ]
    integralidad = np.concatenate([np.ones(nv), np.zeros(nv)])
    sol = milp(c, constraints=restricciones, integrality=integralidad, bounds=Bounds(0, np.r_[np.ones(nv), D.ravel()]))
    x = sol.x[:nv].reshape(K, J)
    plan_precios = P[x.argmax(axis=1)]
    ingreso_planeado = float(np.tile(P, K) @ sol.x[nv:])
    return plan_precios[0], plan_precios, ingreso_planeado


def ingreso_de_plan(plan_precios, I, r, d, par):
    """Ingreso determinista de un plan: cada día se vende la demanda esperada
    hasta que se acaba la capacidad (para comparar los dos métodos del DLA)."""
    P = par.malla()
    D = demanda_esperada(np.array([I]), np.array([r]), d, par)[0]
    j = np.searchsorted(P, plan_precios)
    ventas = D[np.arange(d), j]
    acumuladas = np.minimum(np.cumsum(ventas), I)
    ventas_reales = np.diff(np.concatenate([[0.0], acumuladas]))
    return float((plan_precios * ventas_reales).sum())


def comparar_dla_con_milp(n_estados=60, par=None, semilla=0, beta=1.0):
    """Valida la relajación lagrangiana contra el MILP exacto en estados al azar.
    Regresa la fracción de estados con el mismo precio de hoy, la brecha de ingreso
    planeado (en %) y el tiempo promedio de cada método."""
    import time
    rng = np.random.default_rng(semilla)
    iguales, brechas, t_milp, t_lag = 0, [], 0.0, 0.0
    for _ in range(n_estados):
        d = int(rng.integers(1, par.T + 1))
        I = float(rng.integers(50, par.C + 1))
        r = float(rng.uniform(par.precio_min, par.precio_max))
        t0 = time.time(); p_m, plan_m, _ = resolver_dla_milp(I, r, d, par, beta); t_milp += time.time() - t0
        t0 = time.time(); p_l, plan_l, _ = resolver_dla(np.array([I]), np.array([r]), d, par, beta); t_lag += time.time() - t0
        ing_m = ingreso_de_plan(plan_m, I, r, d, par)
        ing_l = ingreso_de_plan(plan_l[0], I, r, d, par)
        iguales += int(p_m == p_l[0])
        brechas.append(100 * (ing_m - ing_l) / ing_m)
    return {"mismo_precio_hoy": iguales / n_estados, "brecha_media_%": float(np.mean(brechas)),
            "brecha_max_%": float(np.max(brechas)), "seg_milp": t_milp / n_estados, "seg_lagrangiano": t_lag / n_estados}


def politica_dla(beta=1.0):
    """DLA: cada noche resuelve el problema determinista de los días que faltan
    con la demanda esperada y aplica SOLO el precio de hoy. Mañana vuelve a resolver
    con el estado nuevo.

    beta: colchón de capacidad. beta < 1 guarda boletos por si la demanda sale alta;
    beta > 1 planea vender 'de más' por si sale baja. Se afina en el simulador.
    """
    def regla(estado, par):
        I = estado["I"]
        precio = np.full(len(I), float(par.precio_max))
        quedan = I > 0                                    # si ya se agotó, el precio no importa
        if quedan.any():
            precio[quedan], _, _ = resolver_dla(I[quedan], estado["r"][quedan], estado["d"], par, beta)
        return precio
    return regla


# ----------------------------------------------------------------------
# Afinación de parámetros (sobre escenarios de AFINACIÓN)
# ----------------------------------------------------------------------
def _evaluar_malla(constructor, candidatos, esc, par):
    """Simula cada candidato con los MISMOS escenarios y regresa una tabla ordenada
    de mayor a menor ingreso medio."""
    tabla = []
    for theta in candidatos:
        args = theta if isinstance(theta, tuple) else (theta,)
        res = simular(constructor(*args), esc, par, guardar_trayectorias=False)
        s = resumen(res)
        tabla.append({"theta": theta, "ingreso_medio": s["ingreso_medio"],
                      "error_estandar": s["error_estandar"], "prob_agotar": s["prob_agotar"]})
    tabla.sort(key=lambda fila: -fila["ingreso_medio"])
    return tabla[0]["theta"], tabla


def afinar_base(esc, par):
    """Mejor precio fijo de la malla."""
    return _evaluar_malla(politica_base, [float(p) for p in par.malla()], esc, par)


def afinar_pfa(esc, par, p_bases=None, as_=None):
    """Búsqueda en malla de (p_base, a)."""
    p_bases = par.malla() if p_bases is None else p_bases
    as_ = np.arange(0, 401, 25) if as_ is None else as_
    candidatos = [(float(pb), float(a)) for pb in p_bases for a in as_]
    return _evaluar_malla(politica_pfa, candidatos, esc, par)


def afinar_dla(esc, par, betas=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)):
    """Búsqueda del colchón de capacidad beta."""
    return _evaluar_malla(politica_dla, [float(b) for b in betas], esc, par)
