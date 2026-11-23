"""
Calibración de la curva de llegadas lambda0(d) con el dataset de Kaggle "Flight Prices"
(Expedia, 2022), usado como mercado sustituto: capacidad fija y fecha límite.

Idea: para cada vuelo se sigue `seatsRemaining` a lo largo de las búsquedas. Cada vez que
baja entre una búsqueda y la siguiente, esos asientos se vendieron. Contando cuántos se
venden por día según los días que faltan para el vuelo, se obtiene la FORMA de la curva
de llegadas y se ajustan dos formas:
        potencia:     lambda0(d) = a * (d + 1)^(-b)
        exponencial:  lambda0(d) = a * exp(-b * d)
Se queda la de mayor R^2; su forma y su b son los que usa el simulador
(config.py los lee de datos/calibracion.json).
La escala a NO se toma de los vuelos (un avión no es un recinto de 2,000 lugares): se
fija con `compradores_totales` en config.py.

Uso (desde la carpeta del repo, con itineraries.csv en la misma carpeta):

    python calibracion.py extraer      # lee el archivo de 31 GB por pedazos (tarda)
    python calibracion.py ajustar      # usa la muestra chica, ajusta y guarda resultados

o las dos cosas seguidas:

    python calibracion.py

Opciones:
    --archivo itineraries.csv
    --rutas LAX-JFK,JFK-LAX,ATL-LGA    (origen-destino, separadas por comas)

Resultados en la carpeta datos/:
    muestra_rutas.csv      filas de las rutas elegidas (no se sube a Git, es grande)
    curva_llegadas.csv     ventas promedio y tarifa promedio por días al vuelo
    calibracion.json       forma elegida, b ajustado y la calidad de los dos ajustes
    calibracion.png        gráfica de la curva y el ajuste

Supuestos (para escribir en la sección de Calibración):
  - Solo vuelos directos (isNonStop) para que cada legId sea un solo avión.
  - Un vuelo se sigue por (legId, fareBasisCode): el conteo de asientos es por clase de tarifa.
  - seatsRemaining = 0 se toma como "sin dato" (Expedia solo muestra el conteo cuando
    quedan pocos lugares), así que no se usa.
  - Si entre dos búsquedas pasan varios días, la baja se reparte igual entre esos días.
  - Solo se usan d = 1, ..., 60 días al vuelo (la ventana de venta del concierto).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CARPETA = Path(__file__).parent
DATOS = CARPETA / "datos"
COLUMNAS = ["legId", "searchDate", "flightDate", "startingAirport", "destinationAirport",
            "fareBasisCode", "isNonStop", "totalFare", "seatsRemaining"]
T = 60


# ----------------------------------------------------------------------
# Paso A: sacar una muestra chica del archivo grande
# ----------------------------------------------------------------------
def extraer(archivo, rutas, tam_pedazo=2_000_000):
    """Lee itineraries.csv por pedazos y guarda solo las filas de las rutas elegidas."""
    archivo = Path(archivo)
    if not archivo.exists():
        raise FileNotFoundError(f"No encuentro {archivo}. Ponlo en la carpeta del repo o usa --archivo.")
    encabezado = pd.read_csv(archivo, nrows=0).columns
    faltan = [c for c in COLUMNAS if c not in encabezado]
    if faltan:
        raise ValueError(f"Al archivo le faltan columnas: {faltan}")

    pares = {tuple(r.split("-")) for r in rutas}
    DATOS.mkdir(exist_ok=True)
    salida = DATOS / "muestra_rutas.csv"
    if salida.exists():
        salida.unlink()

    total, guardadas = 0, 0
    for i, pedazo in enumerate(pd.read_csv(archivo, usecols=COLUMNAS, chunksize=tam_pedazo)):
        total += len(pedazo)
        ruta = list(zip(pedazo["startingAirport"], pedazo["destinationAirport"]))
        filtro = pd.Series([r in pares for r in ruta], index=pedazo.index) & (pedazo["isNonStop"] == True)  # noqa: E712
        sel = pedazo[filtro]
        sel.to_csv(salida, mode="a", header=(guardadas == 0), index=False)
        guardadas += len(sel)
        print(f"  pedazo {i + 1}: {total:,} filas leídas, {guardadas:,} guardadas", flush=True)
    print(f"Listo: {guardadas:,} filas en {salida}")
    return salida


# ----------------------------------------------------------------------
# Paso B: ventas por día según días al vuelo y ajuste de la curva
# ----------------------------------------------------------------------
def curva_empirica(df):
    """Ventas promedio por vuelo-tarifa y por día, según los días que faltan para el vuelo."""
    df = df.copy()
    df["searchDate"] = pd.to_datetime(df["searchDate"])
    df["flightDate"] = pd.to_datetime(df["flightDate"])
    df["dias"] = (df["flightDate"] - df["searchDate"]).dt.days
    df = df[(df["dias"] >= 1) & (df["dias"] <= T) & (df["seatsRemaining"] > 0)]

    # Una observación por vuelo-tarifa y búsqueda (si se repite, la más barata)
    df = (df.sort_values("totalFare")
            .drop_duplicates(["legId", "fareBasisCode", "searchDate"])
            .sort_values(["legId", "fareBasisCode", "searchDate"]))

    # Bajas de asientos entre búsquedas consecutivas del mismo vuelo-tarifa
    g = df.groupby(["legId", "fareBasisCode"])
    df["asientos_antes"] = g["seatsRemaining"].shift(1)
    df["dias_antes"] = g["dias"].shift(1)
    pares = df.dropna(subset=["asientos_antes"])
    pares = pares[pares["dias_antes"] > pares["dias"]]
    baja = np.maximum(pares["asientos_antes"] - pares["seatsRemaining"], 0)
    hueco = (pares["dias_antes"] - pares["dias"]).astype(int)

    # Repartir cada baja entre los días del hueco y contar cuántos vuelos se observaron cada día
    ventas = np.zeros(T + 1)
    observados = np.zeros(T + 1)
    for d_fin, h, b in zip(pares["dias"].to_numpy(int), hueco.to_numpy(int), baja.to_numpy(float)):
        dias_hueco = np.arange(d_fin, d_fin + h)            # d = d_fin, ..., d_antes - 1
        dias_hueco = dias_hueco[dias_hueco <= T]
        ventas[dias_hueco] += b / h
        observados[dias_hueco] += 1

    tarifa = df.groupby("dias")["totalFare"].mean().reindex(range(1, T + 1))
    d = np.arange(1, T + 1)
    curva = pd.DataFrame({
        "dias_al_vuelo": d,
        "ventas_por_vuelo_dia": np.where(observados[1:] > 0, ventas[1:] / np.maximum(observados[1:], 1), np.nan),
        "observaciones": observados[1:],
        "tarifa_promedio": tarifa.to_numpy(),
    })
    return curva


def exponencial(d, a, b):
    return a * np.exp(-b * d)


def potencia(d, a, b):
    return a * (d + 1) ** (-b)


FORMAS = {"potencia": (potencia, 0.5, "a·(d+1)^(-b)"), "exponencial": (exponencial, 0.05, "a·exp(-b·d)")}


def ajustar(curva):
    """Ajusta las dos formas de lambda0(d) por mínimos cuadrados ponderados (pesa más
    los días con más observaciones) y se queda con la de mayor R^2."""
    c = curva.dropna(subset=["ventas_por_vuelo_dia"])
    c = c[c["observaciones"] > 0]
    d = c["dias_al_vuelo"].to_numpy(float)
    y = c["ventas_por_vuelo_dia"].to_numpy(float)
    sigma = 1 / np.sqrt(c["observaciones"].to_numpy(float))

    def r2(f, p):
        res = y - f(d, *p)
        return 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()

    resultado, ajustes = {}, {}
    for nombre, (f, b0, _) in FORMAS.items():
        p, cov = curve_fit(f, d, y, p0=[y.max(), b0], sigma=sigma, maxfev=20000)
        ajustes[nombre] = p
        resultado[f"b_{nombre}"] = float(p[1])
        resultado[f"b_error_estandar_{nombre}"] = float(np.sqrt(cov[1, 1]))
        resultado[f"r2_{nombre}"] = float(r2(f, p))

    elegida = max(FORMAS, key=lambda nombre: resultado[f"r2_{nombre}"])
    resultado = {"forma_llegadas": elegida, "b_llegadas": resultado[f"b_{elegida}"], **resultado,
                 "dias_con_datos": int(len(d)), "observaciones": int(c["observaciones"].sum())}
    return resultado, ajustes


def graficar(curva, ajustes, elegida, ruta_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
    d = curva["dias_al_vuelo"]
    ejes[0].scatter(d, curva["ventas_por_vuelo_dia"], s=14, color="0.4", label="datos (bajas de seatsRemaining)")
    for color, (nombre, p) in zip(["C1", "C0"], ajustes.items()):
        f, _, texto = FORMAS[nombre]
        marca = " (elegida)" if nombre == elegida else ""
        ejes[0].plot(d, f(d, *p), color=color, lw=2 if marca else 1.2, ls="-" if marca else "--",
                     label=f"{texto}, b = {p[1]:.3f}{marca}")
    ejes[0].invert_xaxis()
    ejes[0].set_xlabel("días que faltan para el vuelo")
    ejes[0].set_ylabel("asientos vendidos por vuelo y día")
    ejes[0].legend()
    ejes[1].plot(d, curva["tarifa_promedio"], color="C2")
    ejes[1].invert_xaxis()
    ejes[1].set_xlabel("días que faltan para el vuelo")
    ejes[1].set_ylabel("totalFare promedio (USD)")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


def correr_ajuste(rutas):
    muestra = DATOS / "muestra_rutas.csv"
    if not muestra.exists():
        raise FileNotFoundError("Primero corre: python calibracion.py extraer")
    df = pd.read_csv(muestra)
    curva = curva_empirica(df)
    resultado, ajustes = ajustar(curva)
    resultado["rutas"] = rutas
    if resultado["b_llegadas"] <= 0:
        print("OJO: b salió <= 0 (las ventas NO crecen al acercarse el vuelo). Revisa las rutas antes de usarlo.")
    curva.to_csv(DATOS / "curva_llegadas.csv", index=False)
    with open(DATOS / "calibracion.json", "w", encoding="utf-8") as fh:
        json.dump(resultado, fh, indent=2, ensure_ascii=False)
    graficar(curva, ajustes, resultado["forma_llegadas"], DATOS / "calibracion.png")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"Guardado en {DATOS}")
    return resultado


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibración de lambda0(d) con Kaggle Flight Prices")
    ap.add_argument("paso", nargs="?", default="todo", choices=["extraer", "ajustar", "todo"])
    ap.add_argument("--archivo", default=str(CARPETA / "itineraries.csv"))
    ap.add_argument("--rutas", default="LAX-JFK,JFK-LAX,ATL-LGA")
    args = ap.parse_args()
    rutas = [r.strip().upper() for r in args.rutas.split(",")]

    if args.paso in ("extraer", "todo"):
        extraer(args.archivo, rutas)
    if args.paso in ("ajustar", "todo"):
        correr_ajuste(rutas)
