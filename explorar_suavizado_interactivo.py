from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd


# Configuracion principal
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "input" / "20251010 mhci,mhcii,25d1,lamp" / "MHCI MHCII 25D1 LAMP1 OVA_ctrl 15.exported.FCS3.csv"
PARAMETRO = 'Comp-Alexa405-A'
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

# Parametros del CSV que no se quieren graficar en los paneles adicionales.
# Nota: PARAMETRO siempre se excluye automaticamente porque ya se grafica arriba.
PARAMETROS_NO_GRAFICAR = [
    "Time","Event#"
]

# En cada figura adicional se mostraran como maximo esta cantidad de graficos.
# Solo aplica a los parametros no ordenados.
MAX_GRAFICOS_POR_FIGURA_NUEVA = 3

# Suavizado para los parametros no ordenados en las figuras nuevas.
EXTRAS_VENTANA_PCT_INICIAL = 5.0
EXTRAS_VENTANA_PCT_MIN = 0.1
EXTRAS_VENTANA_PCT_MAX = 100.0


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


def encontrar_escalon(
    d1: np.ndarray,
    x_d1candidates: list[int],
    x_d2candidates: list[int],
) -> list[int]:
    if len(d1) == 0 or not x_d1candidates:
        return []

    idx_d1min = min(x_d1candidates, key=lambda i: d1[i])

    extremo_izq = None
    extremo_der = None
    for candidato in x_d2candidates:
        if candidato < idx_d1min:
            if extremo_izq is None or candidato > extremo_izq:
                extremo_izq = candidato
        elif candidato > idx_d1min:
            if extremo_der is None or candidato < extremo_der:
                extremo_der = candidato

    # Convencion de salida: [inicio, punto_mayor_pendiente, fin]
    if extremo_izq is None or extremo_der is None:
        return []

    return [extremo_izq, idx_d1min, extremo_der]


def _media_segmento(valores: np.ndarray) -> float:
    if len(valores) == 0:
        return np.nan
    return float(np.nanmean(valores))


def _fmt_media(valor: float) -> str:
    if np.isnan(valor):
        return "nan"
    return f"{valor:.4g}"


def cargar_y_preparar_datos(csv_path: Path, parametro: str, ascendente: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.replace('"', '').str.replace(' ', '')
    while parametro not in df.columns:
        print(f"[WARN] {csv_path.name}: falta columna {parametro}")
        print(df.columns)
        parametro=input("Ingrese nombre del parametro a ordenar: ")

    df_ordenado = df.sort_values(by=parametro, ascending=ascendente).reset_index(drop=True)
    serie = pd.to_numeric(df_ordenado[parametro], errors="coerce").interpolate(limit_direction="both")
    y_original = serie.to_numpy(dtype=float)

    if len(y_original) < 3:
        raise ValueError("Hay muy pocos datos para calcular derivadas (minimo 3 puntos).")

    x = np.arange(1, len(y_original) + 1)
    return df_ordenado, x, y_original


def obtener_series_adicionales(
    df_ordenado: pd.DataFrame,
    parametro_principal: str,
    parametros_no_graficar: list[str],
) -> list[tuple[str, np.ndarray]]:
    excluidos = {p.replace(' ', '') for p in parametros_no_graficar}
    excluidos.add(parametro_principal.replace(' ', ''))

    series_adicionales: list[tuple[str, np.ndarray]] = []
    for columna in df_ordenado.columns:
        if columna in excluidos:
            continue

        serie_numerica = pd.to_numeric(df_ordenado[columna], errors="coerce").interpolate(limit_direction="both")
        if serie_numerica.notna().sum() < 3:
            continue

        series_adicionales.append((columna, serie_numerica.to_numpy(dtype=float)))

    return series_adicionales


def main() -> None:
    df_ordenado, x, y_original = cargar_y_preparar_datos(CSV_PATH, PARAMETRO, ASCENDENTE)
    series_adicionales = obtener_series_adicionales(df_ordenado, PARAMETRO, PARAMETROS_NO_GRAFICAR)

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

    fig, (ax_param, ax_d1, ax_d2, ax_d3) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(bottom=0.34, hspace=0.30)

    titulo = (
        f"{CSV_PATH.name}\n"
        f"Parametro: {PARAMETRO} | Metodo: {METODO_SUAVIZADO}"
    )
    fig.suptitle(titulo, fontsize=11)

    linea_original, = ax_param.plot(x, y_original, color="0.70", linewidth=1.0, label="Original")
    linea_suavizada, = ax_param.plot(x, y_suavizada, color="tab:blue", linewidth=1.3, label="Suavizada")
    esc, = ax_param.plot([], [], "o", color="tab:red", markersize=4, label="Escalon")
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
    ax_d2.grid(True, linestyle="--", alpha=0.3)
    ax_d2.legend(loc="best", fontsize=8)

    d3_raw = np.gradient(d2)
    if SUAVIZAR_D3_PARA_CRUCES:
        d3_filtrada = suavizar_serie(
            pd.Series(d3_raw),
            metodo=METODO_SUAVIZADO,
            ventana=ventana_d2_inicial,
            ewm_alpha=EWM_ALPHA,
        ).to_numpy(dtype=float)
    else:
        d3_filtrada = d3_raw.copy()

    linea_d3_raw, = ax_d3.plot(x, d3_raw, color="0.65", linewidth=1.0, label="D3 sin filtrar")
    linea_d3, = ax_d3.plot(x, d3_filtrada, color="tab:brown", linewidth=1.2, label="D3 filtrada")
    ax_d3.set_ylabel("Derivada 3")
    ax_d3.set_xlabel("Cuenta acumulativa de fagosomas")
    ax_d3.grid(True, linestyle="--", alpha=0.3)
    ax_d3.legend(loc="best", fontsize=8)

    lineas_adicionales: list[tuple[object, object, object, object, object, str, np.ndarray, object]] = []
    figuras_extra: list[object] = []
    sliders_extras: list[object] = []
    if len(series_adicionales) > 0:
        max_por_figura = max(2, int(MAX_GRAFICOS_POR_FIGURA_NUEVA))
        extras_por_figura = max_por_figura
        n_figuras_extra = int(np.ceil(len(series_adicionales) / extras_por_figura))
        colores = plt.cm.tab20(np.linspace(0, 1, max(1, len(series_adicionales))))

        for pagina in range(n_figuras_extra):
            i_inicio = pagina * extras_por_figura
            i_fin = min((pagina + 1) * extras_por_figura, len(series_adicionales))
            bloque = series_adicionales[i_inicio:i_fin]

            n_filas_extra = len(bloque)
            alto_figura_extra = max(8, 2 * n_filas_extra + 2)
            fig_extra, ejes_extra = plt.subplots(n_filas_extra, 1, figsize=(12, alto_figura_extra), sharex=True)
            fig_extra.subplots_adjust(bottom=0.12, hspace=0.30)
            figuras_extra.append(fig_extra)

            if n_filas_extra == 1:
                ejes_extra = [ejes_extra]

            fig_extra.suptitle(
                f"{CSV_PATH.name} | Figura extra {pagina + 1}/{n_figuras_extra}\n"
                f"Resto de parametros (excluyendo: {PARAMETRO})",
                fontsize=11,
            )

            slider_ax_extra = fig_extra.add_axes([0.15, 0.04, 0.70, 0.025])
            slider_extra = Slider(
                ax=slider_ax_extra,
                label="Ventana suavizado extras (%)",
                valmin=EXTRAS_VENTANA_PCT_MIN,
                valmax=EXTRAS_VENTANA_PCT_MAX,
                valinit=EXTRAS_VENTANA_PCT_INICIAL,
                valstep=0.1,
            )
            sliders_extras.append(slider_extra)

            for i_local, (ax_extra, (nombre_columna, y_columna)) in enumerate(zip(ejes_extra, bloque)):
                color = colores[(i_inicio + i_local) % len(colores)]
                linea_extra, = ax_extra.plot(x, y_columna, color=color, linewidth=1.0, label=nombre_columna)
                linea_ini = ax_extra.axvline(1, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8, label="Inicio escalon")
                linea_fin = ax_extra.axvline(1, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8, label="Fin escalon")
                linea_ini.set_visible(False)
                linea_fin.set_visible(False)
                txt_medias = ax_extra.text(
                    0.01,
                    0.99,
                    "",
                    transform=ax_extra.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
                )
                ax_extra.set_ylabel(nombre_columna)
                ax_extra.grid(True, linestyle="--", alpha=0.3)
                ax_extra.legend(loc="best", fontsize=7)
                lineas_adicionales.append((ax_extra, linea_extra, linea_ini, linea_fin, txt_medias, nombre_columna, y_columna, slider_extra))

            ejes_extra[-1].set_xlabel("Cuenta acumulativa de fagosomas")

    txt_inicio = ax_param.text(0, 0, "", fontsize=8, color="tab:red", ha="left", va="bottom")
    txt_fin = ax_param.text(0, 0, "", fontsize=8, color="tab:red", ha="left", va="bottom")
    txt_pendiente = ax_param.text(0, 0, "", fontsize=8, color="tab:purple", ha="left", va="bottom")
    txt_ancho = ax_param.text(0, 0, "", fontsize=8, color="tab:green", ha="left", va="bottom")

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
            esc.set_data([], [])
            linea_d1_raw.set_data([], [])
            linea_d1.set_data([], [])
            linea_d1_cero.set_data([], [])
            linea_d2_raw.set_data([], [])
            linea_d2.set_data([], [])
            linea_d2_cero.set_data([], [])
            linea_d3_raw.set_data([], [])
            linea_d3.set_data([], [])
            for _ax_extra, linea_extra, linea_ini, linea_fin, txt_medias, _nombre, _y, _slider_extra in lineas_adicionales:
                linea_extra.set_data([], [])
                linea_ini.set_visible(False)
                linea_fin.set_visible(False)
                txt_medias.set_text("")
            txt_inicio.set_text("")
            txt_fin.set_text("")
            txt_pendiente.set_text("")
            txt_ancho.set_text("")
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
        d3_raw_actual = np.gradient(d2_actual)
        d3_actual = d3_raw_actual.copy()
        if SUAVIZAR_D3_PARA_CRUCES:
            d3_actual = suavizar_serie(
                pd.Series(d3_actual),
                metodo=METODO_SUAVIZADO,
                ventana=ventana_d2,
                ewm_alpha=EWM_ALPHA,
            ).to_numpy(dtype=float)

        candidatos_d2 = _candidatos_por_cruce_cero(d3_actual)

        escalon= encontrar_escalon(d1_actual ,candidatos_d1 , candidatos_d2)

        linea_original.set_data(x_actual, y_base)
        linea_suavizada.set_data(x_actual, y_s)
        idx_escalon = np.array(escalon, dtype=int)
        try:
            esc.set_data(x_actual[idx_escalon], y_s[idx_escalon])
        except:
            esc.set_data([], [])
        linea_d1_raw.set_data(x_actual, d1_raw_actual)
        linea_d1.set_data(x_actual, d1_actual)
        idx_d1 = np.array(candidatos_d1, dtype=int)
        linea_d1_cero.set_data(x_actual[idx_d1], d1_actual[idx_d1])
        linea_d2_raw.set_data(x_actual, d2_raw_actual)
        linea_d2.set_data(x_actual, d2_actual)
        idx_d2 = np.array(candidatos_d2, dtype=int)
        linea_d2_cero.set_data(x_actual[idx_d2], d2_actual[idx_d2])
        linea_d3_raw.set_data(x_actual, d3_raw_actual)
        linea_d3.set_data(x_actual, d3_actual)

        if descartar_fin == 0:
            recorte = slice(descartar_ini, None)
        else:
            recorte = slice(descartar_ini, -descartar_fin)

        for ax_extra, linea_extra, linea_ini, linea_fin, txt_medias, _nombre, y_columna, slider_extra in lineas_adicionales:
            y_extra = y_columna[recorte]
            ventana_extra = _pct_a_ventana(float(slider_extra.val), n_actual)
            y_extra_suavizado = suavizar_serie(
                pd.Series(y_extra),
                metodo=METODO_SUAVIZADO,
                ventana=ventana_extra,
                ewm_alpha=EWM_ALPHA,
            ).to_numpy(dtype=float)
            linea_extra.set_data(x_actual, y_extra_suavizado)
            if len(idx_escalon) == 3:
                idx_inicio_tmp = int(idx_escalon[0])
                idx_fin_tmp = int(idx_escalon[2])
                linea_ini.set_xdata([x_actual[idx_inicio_tmp], x_actual[idx_inicio_tmp]])
                linea_fin.set_xdata([x_actual[idx_fin_tmp], x_actual[idx_fin_tmp]])
                linea_ini.set_visible(True)
                linea_fin.set_visible(True)

                y_izq = y_extra[:idx_inicio_tmp]
                y_esc = y_extra[idx_inicio_tmp:idx_fin_tmp + 1]
                y_der = y_extra[idx_fin_tmp + 1:]

                y_izq_f = y_extra_suavizado[:idx_inicio_tmp]
                y_esc_f = y_extra_suavizado[idx_inicio_tmp:idx_fin_tmp + 1]
                y_der_f = y_extra_suavizado[idx_fin_tmp + 1:]

                txt_medias.set_text(
                    "Orig  L/E/D: "
                    f"{_fmt_media(_media_segmento(y_izq))} / "
                    f"{_fmt_media(_media_segmento(y_esc))} / "
                    f"{_fmt_media(_media_segmento(y_der))}\n"
                    "Filtr L/E/D: "
                    f"{_fmt_media(_media_segmento(y_izq_f))} / "
                    f"{_fmt_media(_media_segmento(y_esc_f))} / "
                    f"{_fmt_media(_media_segmento(y_der_f))}"
                )
            else:
                linea_ini.set_visible(False)
                linea_fin.set_visible(False)
                txt_medias.set_text("Sin escalon detectado")

        if len(idx_escalon) == 3:
            idx_inicio = int(idx_escalon[0])
            idx_medio = int(idx_escalon[1])
            idx_fin = int(idx_escalon[2])
            n_total_considerado = len(x_actual)

            txt_inicio.set_position((x_actual[idx_inicio], y_s[idx_inicio]))
            txt_inicio.set_text(f"ini: {idx_inicio + 1}/{n_total_considerado}")

            txt_fin.set_position((x_actual[idx_fin], y_s[idx_fin]))
            txt_fin.set_text(f"fin: {idx_fin + 1}/{n_total_considerado}")

            txt_pendiente.set_position((x_actual[idx_medio], y_s[idx_medio]))
            txt_pendiente.set_text(f"pendiente: {d1_actual[idx_medio]:.4g}")

            n_escalon = idx_fin - idx_inicio + 1
            x_ancho = x_actual[idx_inicio]
            y_ancho = y_s[idx_fin]
            txt_ancho.set_position((x_ancho, y_ancho))
            txt_ancho.set_text(f"N escalon: {n_escalon}")
        else:
            txt_inicio.set_text("")
            txt_fin.set_text("")
            txt_pendiente.set_text("")
            txt_ancho.set_text("")

        ax_param.relim()
        ax_param.autoscale_view()
        ax_d1.relim()
        ax_d1.autoscale_view()
        ax_d2.relim()
        ax_d2.autoscale_view()
        ax_d3.relim()
        ax_d3.autoscale_view()
        for ax_extra, _linea_extra, _linea_ini, _linea_fin, _txt_medias, _nombre, _y, _slider_extra in lineas_adicionales:
            ax_extra.relim()
            ax_extra.autoscale_view()

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
        for fig_extra in figuras_extra:
            fig_extra.canvas.draw_idle()

    slider_descartar_ini.on_changed(actualizar)
    slider_descartar_fin.on_changed(actualizar)
    slider_ventana.on_changed(actualizar)
    slider_ventana_d1.on_changed(actualizar)
    slider_ventana_d2.on_changed(actualizar)
    for slider_extra in sliders_extras:
        slider_extra.on_changed(actualizar)
    actualizar(ventana_inicial)
    plt.show()


if __name__ == "__main__":
    main()
