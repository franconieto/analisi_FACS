from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Filtro de fagosomas aberrantes (outliers)
APLICAR_FILTRO_ABERRANTES = True
# Criterio recomendado para este caso: IQR con umbral alto (3.0) para sacar extremos.
IQR_FACTOR = 3.0
# Cantidad minima de parametros en los que una fila debe ser aberrante para eliminarse.
MIN_PARAMETROS_ABERRANTES = 1
# Parametros a excluir del filtro (identificadores, etiquetas, etc.).
COLUMNAS_EXCLUIDAS_FILTRO = {"Event #"}

# Cantidad de fagosomas con mayor valor (post-ordenamiento) que no se grafican.
DESCARTAR_PRIMEROS_N = 150

# Suavizado de curvas (se aplica a todos los parametros excepto al usado para ordenar)
APLICAR_SUAVIZADO = True
# Opciones: "moving_median", "moving_average", "ewm"
METODO_SUAVIZADO = "moving_average"
VENTANA_SUAVIZADO = 200
EWM_ALPHA = 0.25

# Deteccion de escalon en la curva del parametro usado para ordenar
DETECTAR_ESCALON = True
VENTANA_SUAVIZADO_ESCALON = 21
FACTOR_UMBRAL_PENDIENTE = 3.0
LONGITUD_MIN_ESCALON = 200
LONGITUD_MAX_ESCALON = 2400
PASO_BUSQUEDA_ESCALON = 100
TOP_CANDIDATOS_ESCALON = 20


def cargar_experimento_como_dataframes(nombre_experimento: str) -> dict[str, pd.DataFrame]:
	"""
	Carga todos los CSV dentro de input/<nombre_experimento> y devuelve:
	{
		"archivo1.csv": DataFrame,
		"archivo2.csv": DataFrame,
		...
	}
	"""
	carpeta_experimento = INPUT_DIR / nombre_experimento

	if not carpeta_experimento.exists() or not carpeta_experimento.is_dir():
		raise FileNotFoundError(f"No existe la carpeta del experimento: {carpeta_experimento}")

	archivos_csv = sorted(carpeta_experimento.glob("*.csv"))
	if not archivos_csv:
		raise FileNotFoundError(f"No se encontraron CSV en: {carpeta_experimento}")

	datos_por_csv: dict[str, pd.DataFrame] = {}
	for ruta_csv in archivos_csv:
		df = pd.read_csv(ruta_csv, encoding="utf-8-sig")
		datos_por_csv[ruta_csv.name] = df

	return datos_por_csv


def ordenar_dataframe_por_columna(
	datos_por_csv: dict[str, pd.DataFrame],
	nombre_csv: str,
	columna: str,
	ascendente: bool = False,
) -> pd.DataFrame:
	"""
	Ordena un DataFrame por una columna sin desarmar filas.
	Devuelve un DataFrame nuevo (no modifica el original).
	"""
	if nombre_csv not in datos_por_csv:
		raise KeyError(f"No existe el CSV '{nombre_csv}' en el diccionario.")

	df = datos_por_csv[nombre_csv]
	if columna not in df.columns:
		raise KeyError(f"La columna '{columna}' no existe en {nombre_csv}.")

	return df.sort_values(by=columna, ascending=ascendente).reset_index(drop=True)


def _columnas_numericas_ordenadas(df, columna_orden):
	columnas_numericas = []
	for col in df.columns:
		serie_numerica = pd.to_numeric(df[col], errors="coerce")
		if serie_numerica.notna().any():
			columnas_numericas.append(col)

	if not columnas_numericas:
		raise ValueError("No hay columnas numericas para graficar.")

	if columna_orden not in columnas_numericas:
		raise ValueError(
			f"La columna para ordenar '{columna_orden}' no es numerica o no existe."
		)

	columnas_numericas.remove(columna_orden)
	return [columna_orden] + columnas_numericas


def filtrar_fagosomas_aberrantes_iqr(
	df,
	iqr_factor=3.0,
	min_parametros_aberrantes=1,
	columnas_excluidas=None,
):
	"""
	Elimina filas con valores aberrantes usando IQR por columna numerica.
	Una fila se descarta si excede el umbral en al menos N parametros.
	"""
	if columnas_excluidas is None:
		columnas_excluidas = set()

	columnas_numericas = []
	for col in df.columns:
		if col in columnas_excluidas:
			continue
		serie = pd.to_numeric(df[col], errors="coerce")
		if serie.notna().any():
			columnas_numericas.append(col)

	if not columnas_numericas:
		return df.copy().reset_index(drop=True), 0

	conteo_aberrancias = pd.Series(0, index=df.index, dtype="int64")

	for col in columnas_numericas:
		serie = pd.to_numeric(df[col], errors="coerce")
		q1 = serie.quantile(0.25)
		q3 = serie.quantile(0.75)
		iqr = q3 - q1

		if pd.isna(iqr) or iqr == 0:
			continue

		limite_inferior = q1 - iqr_factor * iqr
		limite_superior = q3 + iqr_factor * iqr

		es_aberrante = (serie < limite_inferior) | (serie > limite_superior)
		conteo_aberrancias = conteo_aberrancias.add(es_aberrante.fillna(False).astype(int))

	mascara_conservar = conteo_aberrancias < min_parametros_aberrantes
	df_filtrado = df.loc[mascara_conservar].reset_index(drop=True)
	eliminados = int((~mascara_conservar).sum())

	return df_filtrado, eliminados


def suavizar_serie(serie, metodo="moving_median", ventana=9, ewm_alpha=0.25):
	"""
	Suaviza una serie numerica para reducir ruido visual sin alterar filas.
	"""
	if ventana < 1:
		ventana = 1

	if metodo == "moving_median":
		return serie.rolling(window=ventana, center=True, min_periods=1).median()

	if metodo == "moving_average":
		return serie.rolling(window=ventana, center=True, min_periods=1).mean()

	if metodo == "ewm":
		return serie.ewm(alpha=ewm_alpha, adjust=False).mean()

	raise ValueError(
		"Metodo de suavizado invalido. Use: 'moving_median', 'moving_average' o 'ewm'."
	)


def detectar_limites_escalon(
	serie,
	ventana_suavizado=21,
	factor_umbral_pendiente=3.0,
	longitud_min_escalon=5,
	longitud_max_escalon=2000,
	paso_busqueda=25,
	top_candidatos=6,
):
	"""
	Detecta inicio y fin del escalon ancho en una serie ordenada mediante
	barrido iterativo de candidatos en multiples anchos y seleccion por puntaje.
	Devuelve (idx_inicio, idx_fin) o None si no se detecta escalon.
	"""
	serie_num = pd.to_numeric(serie, errors="coerce")
	serie_limpia = serie_num.dropna().reset_index(drop=True)

	if len(serie_limpia) < max(20, longitud_min_escalon + 2):
		return None

	serie_suavizada = suavizar_serie(
		serie_limpia, metodo="moving_median", ventana=ventana_suavizado
	)
	y = serie_suavizada.to_numpy(dtype=float)

	if len(y) < max(20, longitud_min_escalon + 2):
		return None

	# En orden descendente, el escalon aparece como region con caida sostenida.
	d = y[:-1] - y[1:]
	d = np.nan_to_num(d, nan=0.0)
	if len(d) < 3:
		return None

	med_d = float(np.median(d))
	mad_d = float(np.median(np.abs(d - med_d)))
	umbral_d = med_d + factor_umbral_pendiente * mad_d if mad_d > 0 else med_d

	min_w = max(2, int(longitud_min_escalon))
	max_w = min(int(longitud_max_escalon), len(y) - 1)
	if max_w < min_w:
		max_w = min_w

	paso = max(1, int(paso_busqueda))
	top_n = max(1, int(top_candidatos))

	mejor = None

	for w in range(min_w, max_w + 1, paso):
		# Caida acumulada para ancho w: y[i] - y[i + w]
		drop_w = y[:-w] - y[w:]
		if len(drop_w) == 0:
			continue

		k = min(top_n, len(drop_w))
		idx_top = np.argpartition(drop_w, -k)[-k:]

		for idx in idx_top:
			idx = int(idx)
			drop_total = float(drop_w[idx])
			if drop_total <= 0:
				continue

			drop_promedio = drop_total / w
			if drop_promedio < umbral_d:
				continue

			inside = d[idx : idx + w]
			if len(inside) == 0:
				continue

			# Penaliza candidatos con pendiente muy irregular dentro del escalon.
			in_std = float(np.std(inside))

			# Evalua estabilidad de mesetas antes y despues del escalon.
			lado = max(10, min_w // 4)
			pre = d[max(0, idx - lado) : idx]
			post = d[idx + w : min(len(d), idx + w + lado)]
			pre_std = float(np.std(pre)) if len(pre) > 1 else in_std
			post_std = float(np.std(post)) if len(post) > 1 else in_std

			rango_y = float(np.percentile(y, 95) - np.percentile(y, 5))
			rango_y = rango_y if rango_y > 0 else 1.0

			# Score: privilegia contraste total y ancho, penaliza ruido interno.
			score = (
				(drop_total / rango_y) * (1.0 + 0.35 * (w / min_w))
				- 0.25 * (in_std / (abs(drop_promedio) + 1e-12))
				- 0.10 * ((pre_std + post_std) / (abs(drop_promedio) + 1e-12))
			)

			if (mejor is None) or (score > mejor["score"]):
				mejor = {
					"inicio": idx,
					"fin": idx + w,
					"score": score,
					"drop_promedio": drop_promedio,
				}

	if mejor is None:
		return None

	idx_inicio = int(mejor["inicio"])
	idx_fin = int(mejor["fin"])

	# Refinamiento local de bordes con umbral relativo a pendiente media detectada.
	umbral_ref = max(umbral_d, 0.60 * mejor["drop_promedio"])
	while idx_inicio > 0 and d[idx_inicio - 1] >= umbral_ref:
		idx_inicio -= 1
	while idx_fin < len(d) and d[idx_fin] >= umbral_ref:
		idx_fin += 1

	if (idx_fin - idx_inicio) < min_w:
		return None

	return int(idx_inicio), int(idx_fin)


def graficar_parametros_por_csv(
	datos_por_csv,
	columna_ordenar,
	ascendente=False,
	descartar_primeros_n=0,
	aplicar_filtro_aberrantes=False,
	iqr_factor=3.0,
	min_parametros_aberrantes=1,
	columnas_excluidas_filtro=None,
	aplicar_suavizado=False,
	metodo_suavizado="moving_median",
	ventana_suavizado=9,
	ewm_alpha=0.25,
	detectar_escalon=False,
	ventana_suavizado_escalon=21,
	factor_umbral_pendiente=3.0,
	longitud_min_escalon=5,
	longitud_max_escalon=2000,
	paso_busqueda_escalon=25,
	top_candidatos_escalon=6,
):
	"""
	Genera figuras con bloques de 4 CSV (2x2).
	Dentro de cada bloque (cada CSV), se grafica cada parametro numerico en filas verticales.
	El primer parametro graficado es la columna usada para ordenar.
	"""
	if not datos_por_csv:
		raise ValueError("El diccionario de CSV esta vacio.")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	nombres_csv = list(datos_por_csv.keys())
	grupos = [nombres_csv[i : i + 4] for i in range(0, len(nombres_csv), 4)]
	rutas_guardadas = []
	resumen_filtro = {}
	resumen_escalon = {}

	for numero_figura, grupo_csv in enumerate(grupos, start=1):
		figura = plt.figure(figsize=(18, 12), constrained_layout=True)
		subfiguras = figura.subfigures(2, 2)

		if aplicar_suavizado:
			detalle_suavizado = f"suavizado={metodo_suavizado}"
			if metodo_suavizado in {"moving_median", "moving_average"}:
				detalle_suavizado += f"(w={ventana_suavizado})"
			if metodo_suavizado == "ewm":
				detalle_suavizado += f"(alpha={ewm_alpha})"
		else:
			detalle_suavizado = "suavizado=off"

		detalle_iqr = f"IQR_factor={iqr_factor}" if aplicar_filtro_aberrantes else "IQR=off"

		figura.suptitle(
			(
				f"Experimento ordenado por '{columna_ordenar}' - Figura {numero_figura}\n"
				f"{detalle_iqr} | {detalle_suavizado}"
			),
			fontsize=14,
		)

		for indice in range(4):
			fila = indice // 2
			columna = indice % 2
			subfig = subfiguras[fila, columna]

			if indice >= len(grupo_csv):
				ax_vacio = subfig.subplots(1, 1)
				ax_vacio.axis("off")
				continue

			nombre_csv = grupo_csv[indice]
			df_actual = datos_por_csv[nombre_csv]

			if aplicar_filtro_aberrantes:
				df_actual, eliminados = filtrar_fagosomas_aberrantes_iqr(
					df_actual,
					iqr_factor=iqr_factor,
					min_parametros_aberrantes=min_parametros_aberrantes,
					columnas_excluidas=columnas_excluidas_filtro,
				)
				resumen_filtro[nombre_csv] = eliminados
			else:
				resumen_filtro[nombre_csv] = 0

			datos_procesados = {nombre_csv: df_actual}
			df_ordenado = ordenar_dataframe_por_columna(
				datos_procesados, nombre_csv, columna_ordenar, ascendente=ascendente
			)
			df_ordenado = df_ordenado.iloc[descartar_primeros_n:].reset_index(drop=True)

			if df_ordenado.empty:
				ax_vacio = subfig.subplots(1, 1)
				ax_vacio.axis("off")
				ax_vacio.set_title(
					f"{nombre_csv}\nSin datos tras descartar {descartar_primeros_n}",
					fontsize=9,
				)
				continue

			columnas = _columnas_numericas_ordenadas(df_ordenado, columna_ordenar)
			cantidad_fagosomas = len(df_ordenado)
			eje_x = range(1, cantidad_fagosomas + 1)

			ejes = subfig.subplots(len(columnas), 1, sharex=True)
			if len(columnas) == 1:
				ejes = [ejes]

			subfig.suptitle(nombre_csv, fontsize=10)

			for ax, nombre_columna in zip(ejes, columnas):
				serie = pd.to_numeric(df_ordenado[nombre_columna], errors="coerce")

				if aplicar_suavizado and nombre_columna != columna_ordenar:
					serie = suavizar_serie(
						serie,
						metodo=metodo_suavizado,
						ventana=ventana_suavizado,
						ewm_alpha=ewm_alpha,
					)

				ax.plot(eje_x, serie, linewidth=1.1)
				ax.set_ylabel(nombre_columna, fontsize=8)
				ax.grid(True, linestyle="--", alpha=0.3)

				if detectar_escalon and nombre_columna == columna_ordenar:
					limites = detectar_limites_escalon(
						serie,
						ventana_suavizado=ventana_suavizado_escalon,
						factor_umbral_pendiente=factor_umbral_pendiente,
						longitud_min_escalon=longitud_min_escalon,
						longitud_max_escalon=longitud_max_escalon,
						paso_busqueda=paso_busqueda_escalon,
						top_candidatos=top_candidatos_escalon,
					)
					if limites is not None:
						idx_inicio, idx_fin = limites
						x_inicio = idx_inicio + 1
						x_fin = idx_fin + 1
						ax.axvline(x_inicio, color="tab:red", linestyle="--", linewidth=1.2)
						ax.axvline(x_fin, color="tab:red", linestyle="--", linewidth=1.2)
						ax.text(
							x_inicio,
							ax.get_ylim()[1],
							"inicio escalon",
							fontsize=7,
							color="tab:red",
							verticalalignment="top",
						)
						ax.text(
							x_fin,
							ax.get_ylim()[1],
							"fin escalon",
							fontsize=7,
							color="tab:red",
							verticalalignment="top",
						)
						resumen_escalon[nombre_csv] = (x_inicio, x_fin)
					else:
						resumen_escalon[nombre_csv] = None

			ejes[-1].set_xlabel("Cuenta acumulativa de fagosomas", fontsize=8)

		ruta_salida = OUTPUT_DIR / f"figura_parametros_{numero_figura}.png"
		figura.savefig(ruta_salida, dpi=220)
		rutas_guardadas.append(ruta_salida)
		plt.close(figura)

	return rutas_guardadas, resumen_filtro, resumen_escalon


if __name__ == "__main__":
	nombre_experimento = "20250924 mhci mhcii 25d1"
	datos = cargar_experimento_como_dataframes(nombre_experimento)

	print(f"Experimento: {nombre_experimento}")
	print(f"CSV cargados: {len(datos)}")
	for nombre_csv, df in datos.items():
		print(f"- {nombre_csv}: {len(df)} fagosomas")

	columna_a_ordenar = "OVA"
	rutas, resumen_filtro, resumen_escalon = graficar_parametros_por_csv(
		datos,
		columna_a_ordenar,
		ascendente=False,
		descartar_primeros_n=50,
		aplicar_filtro_aberrantes=APLICAR_FILTRO_ABERRANTES,
		iqr_factor=IQR_FACTOR,
		min_parametros_aberrantes=MIN_PARAMETROS_ABERRANTES,
		columnas_excluidas_filtro=COLUMNAS_EXCLUIDAS_FILTRO,
		aplicar_suavizado=APLICAR_SUAVIZADO,
		metodo_suavizado=METODO_SUAVIZADO,
		ventana_suavizado=VENTANA_SUAVIZADO,
		ewm_alpha=EWM_ALPHA,
		detectar_escalon=DETECTAR_ESCALON,
		ventana_suavizado_escalon=VENTANA_SUAVIZADO_ESCALON,
		factor_umbral_pendiente=FACTOR_UMBRAL_PENDIENTE,
		longitud_min_escalon=LONGITUD_MIN_ESCALON,
		longitud_max_escalon=LONGITUD_MAX_ESCALON,
		paso_busqueda_escalon=PASO_BUSQUEDA_ESCALON,
		top_candidatos_escalon=TOP_CANDIDATOS_ESCALON,
	)

	if APLICAR_FILTRO_ABERRANTES:
		print("\nFagosomas eliminados por criterio aberrante (IQR):")
		for nombre_csv, eliminados in resumen_filtro.items():
			print(f"- {nombre_csv}: {eliminados}")

	if DETECTAR_ESCALON:
		print("\nLimites del escalon (x_inicio, x_fin) en parametro de ordenamiento:")
		for nombre_csv, limites in resumen_escalon.items():
			if limites is None:
				print(f"- {nombre_csv}: no detectado")
			else:
				print(f"- {nombre_csv}: {limites}")

	print("\nFiguras generadas:")
	for ruta in rutas:
		print(f"- {ruta}")
