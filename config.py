"""
Parámetros del problema de precios dinámicos de boletos.

Todo lo que el promotor (o el dashboard) puede cambiar vive en la clase `Parametros`.
Los valores por defecto son provisionales: la forma de la curva de llegadas
(forma_llegadas y b_llegadas) se reemplaza con la calibración de Kaggle cuando existe
el archivo datos/calibracion.json (lo genera calibracion.py).
"""

import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

RUTA_CALIBRACION = Path(__file__).parent / "datos" / "calibracion.json"


@dataclass
class Parametros:
    # --- Recinto y ventana de venta ---
    C: int = 2000                 # capacidad del recinto (boletos)
    T: int = 60                   # días de venta

    # --- Malla de precios (USD) ---
    precio_min: float = 40
    precio_max: float = 200
    paso_precio: float = 10

    # --- Llegadas, d = días que faltan ---
    #   forma "potencia":    lambda0(d) = a * (d + 1)^(-b)
    #   forma "exponencial": lambda0(d) = a * exp(-b * d)
    compradores_totales: float = 5000   # compradores esperados en toda la ventana (sin escasez)
    forma_llegadas: str = "potencia"    # la que mejor ajustó en la calibración de Kaggle
    b_llegadas: float = 0.60            # forma de la curva (se calibra con Kaggle)

    # --- Disposición a pagar (WTP) ~ Lognormal(mu_V, sigma_V^2), en ln(USD) ---
    mu_V: float = float(np.log(95))     # mediana de la WTP = 95 USD
    sigma_V: float = 0.7

    # --- Precio de referencia y aversión a la pérdida ---
    r0: float = 100                     # precio de referencia inicial (USD)
    theta: float = 0.8                  # memoria del precio de referencia (0 a 1)
    eta: float = 0.5                    # peso de la pérdida (precio arriba de la referencia)
    gamma: float = 0.3                  # peso de la ganancia (precio abajo de la referencia)

    # --- Escasez ---
    kappa: float = 1.5                  # factor de llegadas cuando queda poco
    umbral_escasez: float = 0.10        # fracción de C que activa la escasez

    # ------------------------------------------------------------------
    def malla(self):
        """Conjunto de precios permitidos P (USD)."""
        return np.arange(self.precio_min, self.precio_max + 0.5 * self.paso_precio, self.paso_precio)

    def forma(self, d):
        """Forma de la curva sin escala: (d + 1)^(-b) o exp(-b d)."""
        d = np.asarray(d, dtype=float)
        if self.forma_llegadas == "potencia":
            return (d + 1) ** (-self.b_llegadas)
        if self.forma_llegadas == "exponencial":
            return np.exp(-self.b_llegadas * d)
        raise ValueError(f"forma_llegadas desconocida: {self.forma_llegadas}")

    def a_llegadas(self):
        """Escala a de lambda0(d): se elige para que la suma de lambda0 en los T días
        sea igual a compradores_totales."""
        return self.compradores_totales / self.forma(np.arange(1, self.T + 1)).sum()

    def tasa_base(self, d):
        """lambda0(d): compradores esperados por día cuando faltan d días (sin escasez)."""
        return self.a_llegadas() * self.forma(d)

    def tasa_por_dia(self):
        """lambda0 para cada día de venta t = 0, ..., T-1 (en el día t faltan d = T - t días)."""
        return self.tasa_base(self.T - np.arange(self.T))

    def con(self, **cambios):
        """Copia de los parámetros con algunos valores cambiados (útil en el dashboard)."""
        return replace(self, **cambios)

    def como_dict(self):
        return asdict(self)


def cargar_parametros(usar_calibracion=True, **cambios):
    """Parámetros por defecto; si existe datos/calibracion.json, toma de ahí la forma
    de la curva de llegadas (forma_llegadas y b_llegadas)."""
    par = Parametros()
    if usar_calibracion and RUTA_CALIBRACION.exists():
        with open(RUTA_CALIBRACION, encoding="utf-8") as f:
            cal = json.load(f)
        # Archivos viejos (sin "forma_llegadas") traían solo el ajuste exponencial
        par = par.con(forma_llegadas=cal.get("forma_llegadas", "exponencial"),
                      b_llegadas=float(cal["b_llegadas"]))
    return par.con(**cambios) if cambios else par
