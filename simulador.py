"""
Simulador del modelo de Powell (paso 1 del proyecto).

Estado:      S_t = (I_t, d_t, r_t, H_t)   asientos, días que faltan, precio de referencia,
                                          ventas de los últimos 7 días (M_t = suma de H_t)
Decisión:    p_t en la malla P
Exógena:     W_{t+1} = (N_{t+1}, V_{t+1,1}, ..., V_{t+1,N})   llegadas y su disposición a pagar
Transición:  I_{t+1} = I_t - Q_t,  d_{t+1} = d_t - 1,  r_{t+1} = theta r_t + (1-theta) p_t,
             H_{t+1} = (Q_t, Q_{t-1}, ..., Q_{t-6})
Ingreso:     p_t * Q_t

Números aleatorios comunes: `generar_escenarios` sortea TODA la aleatoriedad antes de
simular (llegadas y WTP de cada comprador). Así todas las políticas enfrentan exactamente
los mismos compradores y la diferencia de ingreso se debe a la política, no a la suerte.
"""

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------
# Escenarios
# ----------------------------------------------------------------------
def generar_escenarios(n, par, semilla):
    """Sortea n escenarios (n 'temporadas de venta' completas).

    Las llegadas se separan en dos partes para que la escasez no rompa los números
    aleatorios comunes:
      N_base  ~ Poisson(lambda0(d))              llegan siempre
      N_extra ~ Poisson((kappa - 1) lambda0(d))  solo cuentan si hay escasez
    Con escasez llegan N_base + N_extra ~ Poisson(kappa lambda0(d)), igual que en el modelo.

    V[s, t, k] es la WTP del comprador k del día t en el escenario s: los primeros N_base
    son los compradores normales y los siguientes N_extra los que trae la escasez.
    """
    rng = np.random.default_rng(semilla)
    lam = par.tasa_por_dia()                                        # T tasas, una por día
    N_base = rng.poisson(lam, size=(n, par.T))
    N_extra = rng.poisson(max(par.kappa - 1, 0) * lam, size=(n, par.T))
    K = int((N_base + N_extra).max())                               # máximo de compradores en un día
    V = rng.lognormal(par.mu_V, par.sigma_V, size=(n, par.T, K)).astype(np.float32)
    return {"N_base": N_base, "N_extra": N_extra, "V": V, "semilla": semilla}


def n_escenarios(esc):
    return esc["N_base"].shape[0]


# ----------------------------------------------------------------------
# Piezas del modelo
# ----------------------------------------------------------------------
def precio_efectivo(p, r, par):
    """Precio que percibe el comprador: Pef = p + eta max(p - r, 0) - gamma max(r - p, 0)."""
    return p + par.eta * np.maximum(p - r, 0) - par.gamma * np.maximum(r - p, 0)


def estado_inicial(n, par):
    """S_0 = (I_0 = C, d_0 = T, r_0, H_0 = ceros)."""
    return {
        "I": np.full(n, par.C, dtype=float),
        "d": par.T,
        "r": np.full(n, par.r0, dtype=float),
        "H": np.zeros((n, 7)),
        "M": np.zeros(n),
    }


def un_dia(estado, p, esc, t, par):
    """Simula UN día: transición y ingreso.
    estado = S_t, p = precio del día (uno por escenario), t = día de venta (0, ..., T-1).
    Regresa S_{t+1}, el ingreso del día y los boletos vendidos Q_t."""
    I, r = estado["I"], estado["r"]

    # 1) Llegadas: con escasez (I_t < 10 % de C) se suman los compradores extra
    escasez = I < par.umbral_escasez * par.C
    N = esc["N_base"][:, t] + escasez * esc["N_extra"][:, t]

    # 2) Regla de compra: compra quien tiene WTP >= precio efectivo
    pef = precio_efectivo(p, r, par)
    V = esc["V"][:, t, :]                                   # WTP de los compradores del día
    llego = np.arange(V.shape[1]) < N[:, None]              # solo cuentan los que sí llegaron
    A = ((V >= pef[:, None]) & llego).sum(axis=1)           # compradores interesados

    # 3) Ventas: no se puede vender más de lo que queda
    Q = np.minimum(I, A)
    ingreso = p * Q

    # 4) Transición
    H_nuevo = np.column_stack([Q, estado["H"][:, :6]])      # entra Q_t, sale Q_{t-7}
    estado_nuevo = {
        "I": I - Q,
        "d": estado["d"] - 1,
        "r": par.theta * r + (1 - par.theta) * p,
        "H": H_nuevo,
        "M": H_nuevo.sum(axis=1),
    }
    return estado_nuevo, ingreso, Q


# ----------------------------------------------------------------------
# Simulación completa de una política
# ----------------------------------------------------------------------
def simular(politica, esc, par, guardar_trayectorias=True):
    """Corre una política en todos los escenarios.

    politica(estado, par) -> precio de cada escenario. La política solo ve el estado S_t;
    nunca ve los compradores del día (eso vive en `esc` y solo lo usa `un_dia`).

    Regresa un diccionario con, por escenario:
      ingreso       ingreso total de la temporada (USD)
      vendidos      boletos vendidos
      agotado       True si se vendieron los C boletos
      dia_agotado   día de venta en que se agotó (T si no se agotó)
    y, si guardar_trayectorias=True, las matrices (escenario x día) de precio, ventas,
    inventario al inicio del día y precio de referencia.
    """
    n = n_escenarios(esc)
    estado = estado_inicial(n, par)
    ingreso = np.zeros(n)
    tray = {k: np.zeros((n, par.T)) for k in ("precio", "ventas", "inventario", "referencia")}

    for t in range(par.T):
        p = np.asarray(politica(estado, par), dtype=float) * np.ones(n)   # decisión con S_t
        if guardar_trayectorias:
            tray["precio"][:, t] = p
            tray["inventario"][:, t] = estado["I"]
            tray["referencia"][:, t] = estado["r"]
        estado, ingreso_dia, Q = un_dia(estado, p, esc, t, par)          # llega W_{t+1}
        ingreso += ingreso_dia
        tray["ventas"][:, t] = Q

    vendidos = par.C - estado["I"]
    agotado = estado["I"] <= 0
    acumuladas = np.cumsum(tray["ventas"], axis=1)
    dia_agotado = np.where(agotado, np.argmax(acumuladas >= par.C, axis=1), par.T)

    res = {"ingreso": ingreso, "vendidos": vendidos, "agotado": agotado, "dia_agotado": dia_agotado}
    if guardar_trayectorias:
        res.update(tray)
    return res


def resumen(res, nivel=0.95):
    """Media, desviación, percentiles 5/95 e intervalo de confianza del ingreso,
    más la probabilidad de agotar boletos."""
    x = res["ingreso"]
    n = len(x)
    media, desv = x.mean(), x.std(ddof=1)
    z = norm.ppf(0.5 + nivel / 2)                 # 1.96 para 95 %
    ee = desv / np.sqrt(n)
    return {
        "ingreso_medio": media,
        "desviacion": desv,
        "error_estandar": ee,
        "ic_inf": media - z * ee,
        "ic_sup": media + z * ee,
        "p5": np.percentile(x, 5),
        "p95": np.percentile(x, 95),
        "prob_agotar": res["agotado"].mean(),
        "vendidos_medio": res["vendidos"].mean(),
    }
