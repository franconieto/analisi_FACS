from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Configuracion principal
BASE_DIR = Path(__file__).resolve().parent
NOMBRE_EXPERIMENTO = "20231027 baf,nh4 ova-647,25d1-488"
INPUT_DIR = BASE_DIR / "input" / NOMBRE_EXPERIMENTO
OUTPUT_DIR = BASE_DIR / "output" / NOMBRE_EXPERIMENTO

PARAMETRO = "OVA"
ASCENDENTE = False

# Suavizado del parametro (% del total de fagosomas considerados)
VENTANA_PCT_INICIAL = 10.0
VENTANA_PCT_MIN = 0.1

# Recorte por extremos (post-ordenamiento)
DESCARTAR_INICIALES = 100
DESCARTAR_FINALES = 50

# Suavizado de derivada primera (% del total)
D1_VENTANA_PCT_INICIAL = 5.0
D1_VENTANA_PCT_MIN = 0.1

# Suavizado de derivada segunda (% del total)
D2_VENTANA_PCT_INICIAL = 7.0
D2_VENTANA_PCT_MIN = 0.1

# Opciones: "moving_median", "moving_average", "ewm"
METODO_SUAVIZADO = "moving_average"
EWM_ALPHA = 0.25

# Marcado de puntos donde la derivada siguiente cruza cero
SUAVIZAR_D3_PARA_CRUCES = True

# Filtro para quitar extremos (muy grandes y muy negativos) en parametros secundarios.
FILTRO_EXTREMOS_IQR_FACTOR = 3.0
TIPO_FILTRO_VALORES_GRANDES = "IQR_bilateral"


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
    return ventana if ventana % 2 == 1 else ventana + 1


def _pct_a_ventana(pct: float, n_puntos: int) -> int:
    if n_puntos < 1:
        return 1

    pct = max(VENTANA_PCT_MIN, float(pct))
    ventana = int(round((pct / 100.0) * n_puntos))
    ventana = max(1, min(ventana, n_puntos))
    return _forzar_ventana_impar(ventana)


def _candidatos_por_cruce_cero(derivada_siguiente: np.ndarray) -> list[int]:
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

    if extremo_izq is None or extremo_der is None:
        return []

    return [extremo_izq, idx_d1min, extremo_der]


def _limpiar_nombre_archivo(nombre: str) -> str:
    base = nombre.replace(".csv", "")
    permitidos = "-_() "
    return "".join(c for c in base if c.isalnum() or c in permitidos).strip().replace(" ", "_")


def _filtrar_valores_extremos(x: np.ndarray, y: np.ndarray, iqr_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filtra extremos con limites por IQR: [Q1-k*IQR, Q3+k*IQR]."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x)

    mask_finito = np.isfinite(y)
    if mask_finito.sum() < 3:
        return x[mask_finito], y[mask_finito], mask_finito

    y_validos = y[mask_finito]
    q1 = np.percentile(y_validos, 25)
    q3 = np.percentile(y_validos, 75)
    iqr = q3 - q1
    limite_inferior = q1 - iqr_factor * iqr
    limite_superior = q3 + iqr_factor * iqr

    mask = mask_finito & (y >= limite_inferior) & (y <= limite_superior)
    return x[mask], y[mask], mask


def _columnas_numericas_secundarias(df: pd.DataFrame, parametro: str) -> list[str]:
    columnas: list[str] = []
    for col in df.columns:
        if col == parametro:
            continue
        serie = pd.to_numeric(df[col], errors="coerce")
        if serie.notna().sum() >= 3:
            columnas.append(col)
    return columnas


def _serializar_remocion_por_columna(remocion: dict[str, int]) -> str:
    if not remocion:
        return ""
    return "; ".join(f"{k}:{v}" for k, v in sorted(remocion.items()))


def _graficar_figura_derivadas(
    csv_path: Path,
    x_actual: np.ndarray,
    y_base: np.ndarray,
    y_s: np.ndarray,
    d1_raw: np.ndarray,
    d1: np.ndarray,
    d2_raw: np.ndarray,
    d2: np.ndarray,
    d3_raw: np.ndarray,
    d3: np.ndarray,
    candidatos_d1: list[int],
    candidatos_d2: list[int],
    escalon: list[int],
    ventana: int,
    ventana_d1: int,
    ventana_d2: int,
) -> tuple[Path, dict[str, float | int]]:
    fig, (ax_param, ax_d1, ax_d2, ax_d3) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    titulo = (
        f"{csv_path.name}\n"
        f"Figura 1: Parametro + derivadas | Metodo: {METODO_SUAVIZADO} | "
        f"Vp: {VENTANA_PCT_INICIAL:.1f}% ({ventana}) | "
        f"Vd1: {D1_VENTANA_PCT_INICIAL:.1f}% ({ventana_d1}) | "
        f"Vd2: {D2_VENTANA_PCT_INICIAL:.1f}% ({ventana_d2})"
    )
    fig.suptitle(titulo, fontsize=10)

    ax_param.plot(x_actual, y_base, color="0.70", linewidth=1.0, label="Original")
    ax_param.plot(x_actual, y_s, color="tab:blue", linewidth=1.3, label="Suavizada")

    idx_ini = None
    idx_med = None
    idx_fin = None
    pendiente_escalon = np.nan
    pendiente_recta = np.nan
    n_escalon = np.nan

    if len(escalon) == 3:
        idx = np.array(escalon, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(x_actual))]
        if len(idx) == 3:
            ax_param.plot(x_actual[idx], y_s[idx], "o", color="tab:red", markersize=4, label="Escalon")
            idx_ini, idx_med, idx_fin = int(idx[0]), int(idx[1]), int(idx[2])
            n_total = len(x_actual)
            n_escalon = idx_fin - idx_ini + 1
            pendiente_escalon = float(d1[idx_med])

            x_ini = x_actual[idx_ini]
            x_fin = x_actual[idx_fin]
            y_ini = y_s[idx_ini]
            y_fin = y_s[idx_fin]
            ax_param.plot(
                [x_ini, x_fin],
                [y_ini, y_fin],
                color="tab:purple",
                linewidth=1.4,
                linestyle="-",
                alpha=0.35,
            )

            if x_fin != x_ini:
                pendiente_recta = (y_fin - y_ini) / (x_fin - x_ini)
                ax_param.text(
                    x_actual[idx_ini],
                    y_s[idx_fin],
                    f"m recta: {pendiente_recta:.4g}",
                    fontsize=8,
                    color="tab:purple",
                    ha="left",
                    va="bottom",
                )

            ax_param.text(50 + x_actual[idx_ini], y_s[idx_ini], f"ini: {idx_ini + 1}/{n_total}", fontsize=8, color="tab:red")
            ax_param.text(50 + x_actual[idx_fin], y_s[idx_fin], f"fin: {idx_fin + 1}/{n_total}", fontsize=8, color="tab:red")
            ax_param.text(50 + x_actual[idx_med], y_s[idx_med], f"pendiente: {d1[idx_med]:.4g}", fontsize=8, color="tab:purple")
            ax_param.text(50 + x_actual[idx_med], y_s[idx_med] - 50, f"N escalon: {n_escalon}", fontsize=8, color="tab:green", va="top")

    ax_param.set_ylabel(PARAMETRO)
    ax_param.grid(True, linestyle="--", alpha=0.3)
    ax_param.legend(loc="best", fontsize=8)

    ax_d1.plot(x_actual, d1_raw, color="0.65", linewidth=1.0, label="D1 sin suavizar")
    ax_d1.plot(x_actual, d1, color="tab:orange", linewidth=1.2, label="D1 suavizada")
    idx_d1 = np.array(candidatos_d1, dtype=int)
    idx_d1 = idx_d1[(idx_d1 >= 0) & (idx_d1 < len(x_actual))]
    idx_d1 = idx_d1[np.isfinite(d1[idx_d1])]
    ax_d1.plot(x_actual[idx_d1], d1[idx_d1], "o", color="tab:red", markersize=4, label="D2=0")
    ax_d1.set_ylabel("Derivada 1")
    ax_d1.grid(True, linestyle="--", alpha=0.3)
    ax_d1.legend(loc="best", fontsize=8)

    ax_d2.plot(x_actual, d2_raw, color="0.65", linewidth=1.0, label="D2 sin suavizar")
    ax_d2.plot(x_actual, d2, color="tab:green", linewidth=1.2, label="D2 suavizada")
    idx_d2 = np.array(candidatos_d2, dtype=int)
    idx_d2 = idx_d2[(idx_d2 >= 0) & (idx_d2 < len(x_actual))]
    idx_d2 = idx_d2[np.isfinite(d2[idx_d2])]
    ax_d2.plot(x_actual[idx_d2], d2[idx_d2], "o", color="tab:red", markersize=4, label="D3=0")
    ax_d2.set_ylabel("Derivada 2")
    ax_d2.grid(True, linestyle="--", alpha=0.3)
    ax_d2.legend(loc="best", fontsize=8)

    ax_d3.plot(x_actual, d3_raw, color="0.65", linewidth=1.0, label="D3 sin filtrar")
    ax_d3.plot(x_actual, d3, color="tab:brown", linewidth=1.2, label="D3 filtrada")
    ax_d3.set_ylabel("Derivada 3")
    ax_d3.set_xlabel("Cuenta acumulativa de fagosomas")
    ax_d3.grid(True, linestyle="--", alpha=0.3)
    ax_d3.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    salida = OUTPUT_DIR / f"analisis_derivadas_{_limpiar_nombre_archivo(csv_path.name)}.png"
    fig.savefig(salida, dpi=220)
    plt.close(fig)

    metricas = {
        "idx_inicio_escalon": (int(idx_ini + 1) if idx_ini is not None else np.nan),
        "idx_fin_escalon": (int(idx_fin + 1) if idx_fin is not None else np.nan),
        "idx_pendiente_max_escalon": (int(idx_med + 1) if idx_med is not None else np.nan),
        "pendiente_max_escalon": float(pendiente_escalon) if np.isfinite(pendiente_escalon) else np.nan,
        "pendiente_recta_escalon": float(pendiente_recta) if np.isfinite(pendiente_recta) else np.nan,
        "n_escalon": float(n_escalon) if np.isfinite(n_escalon) else np.nan,
    }
    return salida, metricas


def _graficar_figura_parametro_y_secundarios(
    csv_path: Path,
    x_actual: np.ndarray,
    y_base: np.ndarray,
    y_s: np.ndarray,
    d1: np.ndarray,
    df_base: pd.DataFrame,
    columnas_secundarias: list[str],
    escalon: list[int],
) -> tuple[Path, dict[str, int]]:
    nrows = 1 + len(columnas_secundarias)
    fig_altura = max(8.0, 2.0 * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_altura), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax_param = axes[0]
    axes_secundarios = axes[1:]

    titulo = (
        f"{csv_path.name}\n"
        f"Figura 2: Parametro ordenado + parametros secundarios filtrados por {TIPO_FILTRO_VALORES_GRANDES}"
    )
    fig.suptitle(titulo, fontsize=10)

    ax_param.plot(x_actual, y_base, color="0.70", linewidth=1.0, label="Original")
    ax_param.plot(x_actual, y_s, color="tab:blue", linewidth=1.3, label="Suavizada")

    if len(escalon) == 3:
        idx = np.array(escalon, dtype=int)
        idx = idx[(idx >= 0) & (idx < len(x_actual))]
        if len(idx) == 3:
            ax_param.plot(x_actual[idx], y_s[idx], "o", color="tab:red", markersize=4, label="Escalon")
            idx_ini, idx_med, idx_fin = int(idx[0]), int(idx[1]), int(idx[2])
            n_total = len(x_actual)
            n_escalon = idx_fin - idx_ini + 1

            x_ini = x_actual[idx_ini]
            x_fin = x_actual[idx_fin]
            y_ini = y_s[idx_ini]
            y_fin = y_s[idx_fin]
            ax_param.plot(
                [x_ini, x_fin],
                [y_ini, y_fin],
                color="tab:purple",
                linewidth=1.4,
                linestyle="-",
                alpha=0.35,
            )

            if x_fin != x_ini:
                pendiente_recta = (y_fin - y_ini) / (x_fin - x_ini)
                ax_param.text(
                    x_actual[idx_ini],
                    y_s[idx_fin],
                    f"m recta: {pendiente_recta:.4g}",
                    fontsize=8,
                    color="tab:purple",
                    ha="left",
                    va="bottom",
                )

            ax_param.text(50 + x_actual[idx_ini], y_s[idx_ini], f"ini: {idx_ini + 1}/{n_total}", fontsize=8, color="tab:red")
            ax_param.text(50 + x_actual[idx_fin], y_s[idx_fin], f"fin: {idx_fin + 1}/{n_total}", fontsize=8, color="tab:red")
            ax_param.text(50 + x_actual[idx_med], y_s[idx_med], f"pendiente: {d1[idx_med]:.4g}", fontsize=8, color="tab:purple")
            ax_param.text(50 + x_actual[idx_med], y_s[idx_med] - 50, f"N escalon: {n_escalon}", fontsize=8, color="tab:green", va="top")

    ax_param.set_ylabel(PARAMETRO)
    ax_param.grid(True, linestyle="--", alpha=0.3)
    ax_param.legend(loc="best", fontsize=8)

    remocion_por_columna: dict[str, int] = {}
    for ax_sec, col in zip(axes_secundarios, columnas_secundarias):
        y_col = pd.to_numeric(df_base[col], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float)
        x_col = x_actual.copy()
        mask_finito = np.isfinite(y_col)
        x_col = x_col[mask_finito]
        y_col = y_col[mask_finito]

        x_fil, y_fil, _ = _filtrar_valores_extremos(x_col, y_col, FILTRO_EXTREMOS_IQR_FACTOR)
        remocion_por_columna[col] = int(len(y_col) - len(y_fil))

        ax_sec.scatter(x_fil, y_fil, s=8, color="tab:gray", alpha=0.75, label=f"{col} filtrado")
        ax_sec.set_ylabel(col)
        ax_sec.grid(True, linestyle="--", alpha=0.3)
        ax_sec.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel("Cuenta acumulativa de fagosomas")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    salida = OUTPUT_DIR / f"analisis_parametro_y_resto_{_limpiar_nombre_archivo(csv_path.name)}.png"
    fig.savefig(salida, dpi=220)
    plt.close(fig)
    return salida, remocion_por_columna


def procesar_csv(csv_path: Path) -> dict[str, object] | None:
    global PARAMETRO
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.replace('"', '')
    df.columns = df.columns.str.replace(' ', '')
    while PARAMETRO not in df.columns:
        print(f"[WARN] {csv_path.name}: falta columna {PARAMETRO}")
        print(df.columns)
        PARAMETRO=input("Ingrese nombre del parametro a ordenar: ")

    df_ordenado = df.sort_values(by=PARAMETRO, ascending=ASCENDENTE).reset_index(drop=True)
    serie = pd.to_numeric(df_ordenado[PARAMETRO], errors="coerce").interpolate(limit_direction="both")
    y_original = serie.to_numpy(dtype=float)

    if DESCARTAR_FINALES == 0:
        y_base = y_original[DESCARTAR_INICIALES:]
        df_base = df_ordenado.iloc[DESCARTAR_INICIALES:].reset_index(drop=True)
    else:
        y_base = y_original[DESCARTAR_INICIALES:-DESCARTAR_FINALES]
        df_base = df_ordenado.iloc[DESCARTAR_INICIALES:-DESCARTAR_FINALES].reset_index(drop=True)

    if len(y_base) < 3:
        print(f"[WARN] {csv_path.name}: menos de 3 puntos tras recorte")
        return None

    x_actual = np.arange(1, len(y_base) + 1)
    n_actual = len(y_base)

    ventana = _pct_a_ventana(VENTANA_PCT_INICIAL, n_actual)
    ventana_d1 = _pct_a_ventana(D1_VENTANA_PCT_INICIAL, n_actual)
    ventana_d2 = _pct_a_ventana(D2_VENTANA_PCT_INICIAL, n_actual)

    y_s = suavizar_serie(
        pd.Series(y_base),
        metodo=METODO_SUAVIZADO,
        ventana=ventana,
        ewm_alpha=EWM_ALPHA,
    ).to_numpy(dtype=float)

    d1_raw = np.gradient(y_s)
    d1 = suavizar_serie(
        pd.Series(d1_raw),
        metodo=METODO_SUAVIZADO,
        ventana=ventana_d1,
        ewm_alpha=EWM_ALPHA,
    ).to_numpy(dtype=float)

    d2_raw = np.gradient(d1)
    d2 = suavizar_serie(
        pd.Series(d2_raw),
        metodo=METODO_SUAVIZADO,
        ventana=ventana_d2,
        ewm_alpha=EWM_ALPHA,
    ).to_numpy(dtype=float)

    d3_raw = np.gradient(d2)
    d3 = d3_raw.copy()
    if SUAVIZAR_D3_PARA_CRUCES:
        d3 = suavizar_serie(
            pd.Series(d3),
            metodo=METODO_SUAVIZADO,
            ventana=ventana_d2,
            ewm_alpha=EWM_ALPHA,
        ).to_numpy(dtype=float)

    candidatos_d1 = _candidatos_por_cruce_cero(d2)
    candidatos_d2 = _candidatos_por_cruce_cero(d3)
    escalon = encontrar_escalon(d1, candidatos_d1, candidatos_d2)

    columnas_secundarias = _columnas_numericas_secundarias(df_base, PARAMETRO)
    salida_fig1, metricas = _graficar_figura_derivadas(
        csv_path=csv_path,
        x_actual=x_actual,
        y_base=y_base,
        y_s=y_s,
        d1_raw=d1_raw,
        d1=d1,
        d2_raw=d2_raw,
        d2=d2,
        d3_raw=d3_raw,
        d3=d3,
        candidatos_d1=candidatos_d1,
        candidatos_d2=candidatos_d2,
        escalon=escalon,
        ventana=ventana,
        ventana_d1=ventana_d1,
        ventana_d2=ventana_d2,
    )

    salida_fig2, remocion_por_columna = _graficar_figura_parametro_y_secundarios(
        csv_path=csv_path,
        x_actual=x_actual,
        y_base=y_base,
        y_s=y_s,
        d1=d1,
        df_base=df_base,
        columnas_secundarias=columnas_secundarias,
        escalon=escalon,
    )

    return {
        "csv": csv_path.name,
        "salida_figura_1": str(salida_fig1),
        "salida_figura_2": str(salida_fig2),
        "n_total_ordenado": int(len(y_original)),
        "n_post_recorte": int(len(y_base)),
        "idx_inicio_escalon": metricas["idx_inicio_escalon"],
        "idx_fin_escalon": metricas["idx_fin_escalon"],
        "idx_pendiente_max_escalon": metricas["idx_pendiente_max_escalon"],
        "pendiente_max_escalon": metricas["pendiente_max_escalon"],
        "pendiente_recta_escalon": metricas["pendiente_recta_escalon"],
        "n_escalon": metricas["n_escalon"],
        "columnas_secundarias_filtradas": len(columnas_secundarias),
        "puntos_removidos_por_columna": _serializar_remocion_por_columna(remocion_por_columna),
    }


def main() -> None:
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del experimento: {INPUT_DIR}")

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No hay archivos CSV en: {INPUT_DIR}")

    print(f"Experimento: {NOMBRE_EXPERIMENTO}")
    print(f"CSV detectados: {len(csv_files)}")

    generados = 0
    resultados: list[dict[str, object]] = []
    for csv_path in csv_files:
        resultado = procesar_csv(csv_path)
        if resultado is not None:
            generados += 1
            resultados.append(resultado)
            print(
                f"[OK] {csv_path.name} -> {resultado['salida_figura_1']} | "
                f"{resultado['salida_figura_2']}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excel_salida = OUTPUT_DIR / "resumen_filtros_y_escalon.xlsx"

    df_config = pd.DataFrame(
        [
            {"parametro": "NOMBRE_EXPERIMENTO", "valor": NOMBRE_EXPERIMENTO},
            {"parametro": "PARAMETRO_ORDENADO", "valor": PARAMETRO},
            {"parametro": "ASCENDENTE", "valor": ASCENDENTE},
            {"parametro": "DESCARTAR_INICIALES", "valor": DESCARTAR_INICIALES},
            {"parametro": "DESCARTAR_FINALES", "valor": DESCARTAR_FINALES},
            {"parametro": "METODO_SUAVIZADO", "valor": METODO_SUAVIZADO},
            {"parametro": "VENTANA_PCT_INICIAL", "valor": VENTANA_PCT_INICIAL},
            {"parametro": "D1_VENTANA_PCT_INICIAL", "valor": D1_VENTANA_PCT_INICIAL},
            {"parametro": "D2_VENTANA_PCT_INICIAL", "valor": D2_VENTANA_PCT_INICIAL},
            {"parametro": "EWM_ALPHA", "valor": EWM_ALPHA},
            {"parametro": "SUAVIZAR_D3_PARA_CRUCES", "valor": SUAVIZAR_D3_PARA_CRUCES},
            {"parametro": "TIPO_FILTRO_VALORES_GRANDES", "valor": TIPO_FILTRO_VALORES_GRANDES},
            {
                "parametro": "FILTRO_VALORES_GRANDES_IQR_FACTOR",
                "valor": FILTRO_EXTREMOS_IQR_FACTOR,
            },
            {
                "parametro": "APLICACION_FILTRO_VALORES_GRANDES",
                "valor": "Solo parametros secundarios del CSV (no parametro ordenado ni derivadas)",
            },
        ]
    )

    df_resultados = pd.DataFrame(resultados)
    with pd.ExcelWriter(excel_salida, engine="openpyxl") as writer:
        df_config.to_excel(writer, sheet_name="configuracion", index=False)
        df_resultados.to_excel(writer, sheet_name="resumen_por_csv", index=False)

    print(f"Figuras generadas: {generados}")
    print(f"Salida: {OUTPUT_DIR}")
    print(f"Excel resumen: {excel_salida}")


if __name__ == "__main__":
    main()
