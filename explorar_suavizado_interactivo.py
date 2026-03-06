from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd


# Configuracion principal
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "input" / "20251001 mhci mhcii 25d1" / "LAMP1 MHCI MHCII 25D1 OVA_n 240.exported.FCS3.csv"
PARAMETRO = "OVA"
ASCENDENTE = False

# Parametros iniciales de suavizado expresados como % del total de fagosomas
VENTANA_PCT_INICIAL = 10.0
VENTANA_PCT_MIN = 0.1
VENTANA_PCT_MAX = 100.0

# Cuantos fagosomas iniciales descartar (post-ordenamiento)
DESCARTAR_INICIALES = 100
DESCARTAR_MIN = 0

# Cuantos fagosomas finales descartar (post-ordenamiento)
DESCARTAR_FINALES = 50
DESCARTAR_FINALES_MIN = 0

# Suavizado de derivada primera (% del total)
D1_VENTANA_PCT_INICIAL = 5.0
D1_VENTANA_PCT_MIN = 0.1
D1_VENTANA_PCT_MAX = 100.0

# Suavizado de derivada segunda (% del total)
D2_VENTANA_PCT_INICIAL = 7.0
D2_VENTANA_PCT_MIN = 0.1
D2_VENTANA_PCT_MAX = 100.0

# Opciones: "moving_median", "moving_average", "ewm"
METODO_SUAVIZADO = "moving_average"
EWM_ALPHA = 0.25

# Marcado de puntos donde la derivada siguiente cruza cero
SUAVIZAR_D3_PARA_CRUCES = True


def suavizar_serie(serie: pd.Series, metodo: str, ventana: int, ewm_alpha: float) -> pd.Series:
    if ventana < 1:
        ventana = 1

    if metodo == "moving_median":
        return serie.rolling(window=ventana, center=True, min_periods=1).median()

    if metodo == "moving_average":
        return serie.rolling(window=ventana, center=True, min_periods=1).mean()

    if metodo == "ewm":
        return serie.ewm(alpha=ewm_alpha, adjust=False).mean()

    raise ValueError("Metodo invalido. Use: moving_median, moving_average o ewm")


def _forzar_ventana_impar(ventana: int) -> int:
    # Para filtros centrados conviene ventana impar para simetria.
    return ventana if ventana % 2 == 1 else ventana + 1


def _pct_a_ventana(pct: float, n_puntos: int) -> int:
    if n_puntos < 1:
        return 1

    pct = max(0.1, float(pct))
    ventana = int(round((pct / 100.0) * n_puntos))
    ventana = max(1, min(ventana, n_puntos))
    return _forzar_ventana_impar(ventana)


def _candidatos_por_cruce_cero(derivada_siguiente: np.ndarray) -> list[int]:
    """Devuelve indices candidatos donde la derivada siguiente cruza por cero."""
    if len(derivada_siguiente) < 3:
        return []

    d = np.asarray(derivada_siguiente, dtype=float)
    candidatos: list[int] = []

    for i in range(1, len(d)):
        anterior = d[i - 1]
        actual = d[i]

        if anterior == 0.0 and actual == 0.0:
            continue

        if anterior == 0.0 or actual == 0.0 or (anterior > 0 and actual < 0) or (anterior < 0 and actual > 0):
            candidatos.append(i)

    # Quitar duplicados conservando orden.
    vistos = set()
    unicos = []
    for c in candidatos:
        if c not in vistos:
            vistos.add(c)
            unicos.append(c)
    return unicos


def cargar_y_preparar_datos(csv_path: Path, parametro: str, ascendente: bool) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if parametro not in df.columns:
        raise KeyError(f"La columna '{parametro}' no existe en {csv_path.name}")

    df_ordenado = df.sort_values(by=parametro, ascending=ascendente).reset_index(drop=True)
    serie = pd.to_numeric(df_ordenado[parametro], errors="coerce").interpolate(limit_direction="both")
    y_original = serie.to_numpy(dtype=float)

    if len(y_original) < 3:
        raise ValueError("Hay muy pocos datos para calcular derivadas (minimo 3 puntos).")

    x = np.arange(1, len(y_original) + 1)
    return x, y_original


def main() -> None:
    x, y_original = cargar_y_preparar_datos(CSV_PATH, PARAMETRO, ASCENDENTE)

    n_total = len(y_original)
    ventana_inicial = _pct_a_ventana(VENTANA_PCT_INICIAL, n_total)
    ventana_d1_inicial = _pct_a_ventana(D1_VENTANA_PCT_INICIAL, n_total)
    ventana_d2_inicial = _pct_a_ventana(D2_VENTANA_PCT_INICIAL, n_total)
    serie_suavizada = suavizar_serie(
        pd.Series(y_original),
        metodo=METODO_SUAVIZADO,
        ventana=ventana_inicial,
        ewm_alpha=EWM_ALPHA,
    )
    y_suavizada = serie_suavizada.to_numpy(dtype=float)
    d1_raw = np.gradient(y_suavizada)
    d1 = suavizar_serie(
        pd.Series(d1_raw),
        metodo=METODO_SUAVIZADO,
        ventana=ventana_d1_inicial,
        ewm_alpha=EWM_ALPHA,
    ).to_numpy(dtype=float)
    d2_raw = np.gradient(d1)
    d2 = suavizar_serie(
        pd.Series(d2_raw),
        metodo=METODO_SUAVIZADO,
        ventana=ventana_d2_inicial,
        ewm_alpha=EWM_ALPHA,
    ).to_numpy(dtype=float)

    fig, (ax_param, ax_d1, ax_d2) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(bottom=0.34, hspace=0.30)

    titulo = (
        f"{CSV_PATH.name}\n"
        f"Parametro: {PARAMETRO} | Metodo: {METODO_SUAVIZADO}"
    )
    fig.suptitle(titulo, fontsize=11)

    linea_original, = ax_param.plot(x, y_original, color="0.70", linewidth=1.0, label="Original")
    linea_suavizada, = ax_param.plot(x, y_suavizada, color="tab:blue", linewidth=1.3, label="Suavizada")
    ax_param.set_ylabel(PARAMETRO)
    ax_param.grid(True, linestyle="--", alpha=0.3)
    ax_param.legend(loc="best", fontsize=8)

    linea_d1_raw, = ax_d1.plot(x, d1_raw, color="0.65", linewidth=1.0, label="D1 sin suavizar")
    linea_d1, = ax_d1.plot(x, d1, color="tab:orange", linewidth=1.2, label="D1 suavizada")
    linea_d1_cero, = ax_d1.plot([], [], "o", color="tab:red", markersize=4, label="D2=0")
    ax_d1.set_ylabel("Derivada 1")
    ax_d1.grid(True, linestyle="--", alpha=0.3)
    ax_d1.legend(loc="best", fontsize=8)

    linea_d2_raw, = ax_d2.plot(x, d2_raw, color="0.65", linewidth=1.0, label="D2 sin suavizar")
    linea_d2, = ax_d2.plot(x, d2, color="tab:green", linewidth=1.2, label="D2 suavizada")
    linea_d2_cero, = ax_d2.plot([], [], "o", color="tab:red", markersize=4, label="D3=0")
    ax_d2.set_ylabel("Derivada 2")
    ax_d2.set_xlabel("Cuenta acumulativa de fagosomas")
    ax_d2.grid(True, linestyle="--", alpha=0.3)
    ax_d2.legend(loc="best", fontsize=8)

    slider_ax_descartar_ini = fig.add_axes([0.15, 0.18, 0.70, 0.022])
    slider_descartar_ini = Slider(
        ax=slider_ax_descartar_ini,
        label="Descartar iniciales",
        valmin=DESCARTAR_MIN,
        valmax=max(DESCARTAR_MIN, len(y_original) - 3),
        valinit=min(DESCARTAR_INICIALES, max(0, len(y_original) - 3)),
        valstep=1,
    )

    slider_ax_descartar_fin = fig.add_axes([0.15, 0.145, 0.70, 0.022])
    slider_descartar_fin = Slider(
        ax=slider_ax_descartar_fin,
        label="Descartar finales",
        valmin=DESCARTAR_FINALES_MIN,
        valmax=max(DESCARTAR_FINALES_MIN, len(y_original) - 3),
        valinit=min(DESCARTAR_FINALES, max(0, len(y_original) - 3)),
        valstep=1,
    )

    slider_ax_param = fig.add_axes([0.15, 0.11, 0.70, 0.022])
    slider_ventana = Slider(
        ax=slider_ax_param,
        label="Ventana suavizado (%)",
        valmin=VENTANA_PCT_MIN,
        valmax=VENTANA_PCT_MAX,
        valinit=VENTANA_PCT_INICIAL,
        valstep=0.1,
    )

    slider_ax_d1 = fig.add_axes([0.15, 0.075, 0.70, 0.022])
    slider_ventana_d1 = Slider(
        ax=slider_ax_d1,
        label="Ventana suavizado D1 (%)",
        valmin=D1_VENTANA_PCT_MIN,
        valmax=D1_VENTANA_PCT_MAX,
        valinit=D1_VENTANA_PCT_INICIAL,
        valstep=0.1,
    )

    slider_ax_d2 = fig.add_axes([0.15, 0.04, 0.70, 0.022])
    slider_ventana_d2 = Slider(
        ax=slider_ax_d2,
        label="Ventana suavizado D2 (%)",
        valmin=D2_VENTANA_PCT_MIN,
        valmax=D2_VENTANA_PCT_MAX,
        valinit=D2_VENTANA_PCT_INICIAL,
        valstep=0.1,
    )

    def actualizar(_valor: float) -> None:
        descartar_ini = int(slider_descartar_ini.val)
        descartar_fin = int(slider_descartar_fin.val)
        ventana_pct = float(slider_ventana.val)
        ventana_d1_pct = float(slider_ventana_d1.val)
        ventana_d2_pct = float(slider_ventana_d2.val)

        # Recorta por ambos extremos sin romper cuando los sliders se solapan.
        if descartar_fin == 0:
            y_base = y_original[descartar_ini:]
        else:
            y_base = y_original[descartar_ini:-descartar_fin]

        if len(y_base) < 3:
            linea_original.set_data([], [])
            linea_suavizada.set_data([], [])
            linea_d1_raw.set_data([], [])
            linea_d1.set_data([], [])
            linea_d1_cero.set_data([], [])
            linea_d2_raw.set_data([], [])
            linea_d2.set_data([], [])
            linea_d2_cero.set_data([], [])
            ax_param.set_title(
                "Ajusta recorte inicial/final para dejar al menos 3 puntos",
                fontsize=9,
            )
            fig.canvas.draw_idle()
            return

        x_actual = np.arange(1, len(y_base) + 1)
        n_actual = len(y_base)
        ventana = _pct_a_ventana(ventana_pct, n_actual)
        ventana_d1 = _pct_a_ventana(ventana_d1_pct, n_actual)
        ventana_d2 = _pct_a_ventana(ventana_d2_pct, n_actual)

        y_s = suavizar_serie(
            pd.Series(y_base),
            metodo=METODO_SUAVIZADO,
            ventana=ventana,
            ewm_alpha=EWM_ALPHA,
        ).to_numpy(dtype=float)
        d1_raw_actual = np.gradient(y_s)
        d1_actual = suavizar_serie(
            pd.Series(d1_raw_actual),
            metodo=METODO_SUAVIZADO,
            ventana=ventana_d1,
            ewm_alpha=EWM_ALPHA,
        ).to_numpy(dtype=float)
        d2_raw_actual = np.gradient(d1_actual)
        d2_actual = suavizar_serie(
            pd.Series(d2_raw_actual),
            metodo=METODO_SUAVIZADO,
            ventana=ventana_d2,
            ewm_alpha=EWM_ALPHA,
        ).to_numpy(dtype=float)

        # Puntos en D1 donde D2 suavizada cruza cero.
        candidatos_d1 = _candidatos_por_cruce_cero(d2_actual)

        # Puntos en D2 donde D3 cruza cero. D3 tambien se suaviza antes de buscar cruces.
        d3_actual = np.gradient(d2_actual)
        if SUAVIZAR_D3_PARA_CRUCES:
            d3_actual = suavizar_serie(
                pd.Series(d3_actual),
                metodo=METODO_SUAVIZADO,
                ventana=ventana_d2,
                ewm_alpha=EWM_ALPHA,
            ).to_numpy(dtype=float)

        candidatos_d2 = _candidatos_por_cruce_cero(d3_actual)

        linea_original.set_data(x_actual, y_base)
        linea_suavizada.set_data(x_actual, y_s)
        linea_d1_raw.set_data(x_actual, d1_raw_actual)
        linea_d1.set_data(x_actual, d1_actual)
        idx_d1 = np.array(candidatos_d1, dtype=int)
        linea_d1_cero.set_data(x_actual[idx_d1], d1_actual[idx_d1])
        linea_d2_raw.set_data(x_actual, d2_raw_actual)
        linea_d2.set_data(x_actual, d2_actual)
        idx_d2 = np.array(candidatos_d2, dtype=int)
        linea_d2_cero.set_data(x_actual[idx_d2], d2_actual[idx_d2])

        ax_param.relim()
        ax_param.autoscale_view()
        ax_d1.relim()
        ax_d1.autoscale_view()
        ax_d2.relim()
        ax_d2.autoscale_view()

        ax_param.set_title(
            (
                f"Descartar ini: {descartar_ini} | Descartar fin: {descartar_fin} | "
                f"Vp: {ventana_pct:.1f}% ({ventana}) | "
                f"Vd1: {ventana_d1_pct:.1f}% ({ventana_d1}) | "
                f"Vd2: {ventana_d2_pct:.1f}% ({ventana_d2})"
            ),
            fontsize=9,
        )
        fig.canvas.draw_idle()

    slider_descartar_ini.on_changed(actualizar)
    slider_descartar_fin.on_changed(actualizar)
    slider_ventana.on_changed(actualizar)
    slider_ventana_d1.on_changed(actualizar)
    slider_ventana_d2.on_changed(actualizar)
    actualizar(ventana_inicial)
    plt.show()


if __name__ == "__main__":
    main()
