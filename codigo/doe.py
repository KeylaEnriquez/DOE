
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

MATRICULA = 2006646

T_MIN = 10
T_MAX = 100

TEMPERATURA_FINAL = 0.1
MAX_ITERACIONES = 1500

TAMANOS = [5, 10, 15]
REPLICAS = 10

ALPHA = 0.05
BOOTSTRAP_REPLICAS = 5000

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = CARPETA_PROYECTO / "resultados"
CARPETA_GRAFICAS = CARPETA_RESULTADOS / "graficas"

CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_GRAFICAS.mkdir(exist_ok=True)

# =====================================================
# 2. GENERACION DE LOS NIVELES DE TEMPERATURA
# =====================================================

random.seed(MATRICULA)
r = random.random()

fracciones = {
    "Bajo": 0.3 * r,
    "Medio": 0.3 * r + 0.3,
    "Alto": 0.3 * r + 0.6
}

temperaturas = {
    nivel: T_MIN + fraccion * (T_MAX - T_MIN)
    for nivel, fraccion in fracciones.items()
}

print("\n========== DISEÑO EXPERIMENTAL ==========")
print(f"Matricula: {MATRICULA}")
print(f"Valor de r: {r:.12f}")
print("\nTemperaturas experimentales:")

for nivel, temperatura in temperaturas.items():
    print(f"{nivel}: {temperatura:.6f}")

# =====================================================
# 3. GENERACION DE INSTANCIAS TSP
# =====================================================

def generar_instancia(n):
    
    rng = np.random.default_rng(42 + n)
    coordenadas = rng.random((n, 2))

    distancias = np.sqrt(
        np.sum(
            (coordenadas[:, np.newaxis] - coordenadas) ** 2,
            axis=2
        )
    )

    return coordenadas, distancias

# =====================================================
# 4. FUNCION DE COSTO
# =====================================================

def costo_ruta(ruta, distancias):

    n = len(ruta)

    return sum(
        distancias[ruta[i], ruta[(i + 1) % n]]
        for i in range(n)
    )

# =====================================================
# 5. RECOCIDO SIMULADO CON 2-OPT
# =====================================================

def simulated_annealing(
    temp_inicial,
    distancias,
    semilla,
    max_iter=MAX_ITERACIONES
):

    rng = np.random.default_rng(semilla)
    n = len(distancias)

    ruta_actual = rng.permutation(n).tolist()
    mejor_ruta = ruta_actual.copy()

    costo_actual = costo_ruta(ruta_actual, distancias)
    mejor_costo = costo_actual

    tiempo_inicio = time.time()

    for iteracion in range(max_iter):

        progreso = iteracion / max_iter

        T = temp_inicial * (
            TEMPERATURA_FINAL / temp_inicial
        ) ** progreso

        i, j = sorted(
            rng.choice(n, size=2, replace=False)
        )

        if j - i < 2:
            continue

        ruta_vecina = ruta_actual.copy()
        ruta_vecina[i:j + 1] = ruta_vecina[i:j + 1][::-1]
        costo_vecino = costo_ruta(ruta_vecina, distancias)
        delta_costo = costo_vecino - costo_actual

        if (delta_costo < 0 or rng.random() < math.exp(-delta_costo / T)):

            ruta_actual = ruta_vecina
            costo_actual = costo_vecino

            if costo_actual < mejor_costo:

                mejor_costo = costo_actual
                mejor_ruta = ruta_actual.copy()

    tiempo_ejecucion = time.time() - tiempo_inicio

    return mejor_costo, mejor_ruta, tiempo_ejecucion

# =====================================================
# 6. EJECUCION DE LAS 90 CORRIDAS
# =====================================================

resultados = []
contador = 0

print("\n========== EJECUTANDO EXPERIMENTOS ==========\n")

for n in TAMANOS:

    coordenadas, distancias = generar_instancia(n)

    for nivel, temperatura in temperaturas.items():

        for replica in range(1, REPLICAS + 1):

            contador += 1
            semilla = 202600 + contador

            costo, ruta, tiempo = simulated_annealing(
                temperatura,
                distancias,
                semilla
            )

            resultados.append({
                "tamano": n,
                "temperatura": nivel,
                "valor_temperatura": temperatura,
                "replica": replica,
                "semilla": semilla,
                "costo": costo,
                "tiempo": tiempo
            })

            print(
                f"Corrida {contador}/90 | "
                f"n={n} | "
                f"Temperatura={nivel} | "
                f"Replica={replica} | "
                f"Costo={costo:.6f}"
            )

# =====================================================
# 7. GUARDAR RESULTADOS
# =====================================================

df = pd.DataFrame(resultados)
archivo_datos = CARPETA_RESULTADOS / "resultados_experimento.csv"
df.to_csv(archivo_datos, index=False)

print("\nResultados guardados en:")
print(archivo_datos)

# =====================================================
# 8. ANOVA DE DOS FACTORES
# =====================================================

print("\n========== ANOVA DE DOS FACTORES ==========")

modelo = ols(
    "costo ~ C(temperatura) * C(tamano)",
    data=df
).fit()

tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print(tabla_anova)

archivo_anova = CARPETA_RESULTADOS / "resultados_anova.csv"

tabla_anova.to_csv(archivo_anova)

print("\n========== INTERPRETACION ANOVA ==========")

for efecto, fila in tabla_anova.iterrows():

    p_valor = fila["PR(>F)"]

    if p_valor < ALPHA:
        print(
            f"{efecto}: SIGNIFICATIVO "
            f"(p = {p_valor:.6f})"
        )
    else:
        print(
            f"{efecto}: NO SIGNIFICATIVO "
            f"(p = {p_valor:.6f})"
        )

# =====================================================
# 9. BOOTSTRAP DE LA MEDIANA
# =====================================================

def bootstrap_mediana(datos, semilla, B=BOOTSTRAP_REPLICAS):

    rng = np.random.default_rng(semilla)
    datos = np.asarray(datos)
    medianas = []

    for _ in range(B):

        muestra = rng.choice(
            datos,
            size=len(datos),
            replace=True
        )

        medianas.append(np.median(muestra))

    mediana = np.median(datos)

    limite_inferior = np.percentile(medianas, 2.5)
    limite_superior = np.percentile(medianas, 97.5)

    return mediana, limite_inferior, limite_superior

resultados_bootstrap = []

for n in TAMANOS:

    for indice, (nivel, temperatura) in enumerate(
        temperaturas.items()
    ):

        datos = df[
            (df["tamano"] == n) &
            (df["temperatura"] == nivel)
        ]["costo"].values

        mediana, inferior, superior = bootstrap_mediana(
            datos,
            semilla=300000 + n * 10 + indice
        )

        resultados_bootstrap.append({
            "tamano": n,
            "temperatura": nivel,
            "valor_temperatura": temperatura,
            "mediana_bootstrap": mediana,
            "IC_95_inferior": inferior,
            "IC_95_superior": superior
        })

df_bootstrap = pd.DataFrame(resultados_bootstrap)

df_bootstrap.to_csv(
    CARPETA_RESULTADOS / "resultados_bootstrap.csv",
    index=False
)

print("\n========== TABLA COMPLETA BOOTSTRAP ==========")
print(df_bootstrap.to_string(index=False))

# =====================================================
# 10. IDENTIFICAR LA MEJOR TEMPERATURA POR TAMANO
# =====================================================

print("\n========== MEJORES TEMPERATURAS ==========")

resumen = []

for n in TAMANOS:

    datos_tamano = df_bootstrap[
        df_bootstrap["tamano"] == n
    ]

    mejor = datos_tamano.loc[
        datos_tamano["mediana_bootstrap"].idxmin()
    ]

    resumen.append({
        "tamano": n,
        "mejor_nivel": mejor["temperatura"],
        "temperatura_optima": mejor["valor_temperatura"],
        "mediana_bootstrap": mejor["mediana_bootstrap"],
        "IC_95_inferior": mejor["IC_95_inferior"],
        "IC_95_superior": mejor["IC_95_superior"]
    })

    print(
        f"\nn = {n}\n"
        f"Mejor nivel: {mejor['temperatura']}\n"
        f"Temperatura: {mejor['valor_temperatura']:.6f}\n"
        f"Mediana: {mejor['mediana_bootstrap']:.6f}\n"
        f"IC 95%: "
        f"[{mejor['IC_95_inferior']:.6f}, "
        f"{mejor['IC_95_superior']:.6f}]"
    )

df_resumen = pd.DataFrame(resumen)

df_resumen.to_csv(
    CARPETA_RESULTADOS / "tabla_temperaturas_optimas.csv",
    index=False
)

# =====================================================
# 11. GRAFICA DE EFECTOS PRINCIPALES
# =====================================================

fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

media_temperatura = df.groupby(
    "temperatura",
    observed=False
)["costo"].mean().reindex(
    ["Bajo", "Medio", "Alto"]
)

ejes[0].plot(
    media_temperatura.index,
    media_temperatura.values,
    marker="o",
    linewidth=2
)

ejes[0].set_title("Efecto principal de la temperatura")
ejes[0].set_xlabel("Nivel de temperatura")
ejes[0].set_ylabel("Costo promedio")
ejes[0].grid(True)

media_tamano = df.groupby("tamano")["costo"].mean()

ejes[1].plot(
    media_tamano.index,
    media_tamano.values,
    marker="o",
    linewidth=2
)

ejes[1].set_title("Efecto principal del tamaño")
ejes[1].set_xlabel("Número de ciudades (n)")
ejes[1].set_ylabel("Costo promedio")
ejes[1].grid(True)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "efectos_principales.png",
    dpi=300
)

plt.close()

# =====================================================
# 12. GRAFICA DE INTERACCION
# =====================================================

tabla_interaccion = df.pivot_table(
    index="temperatura",
    columns="tamano",
    values="costo",
    aggfunc="mean"
).reindex(["Bajo", "Medio", "Alto"])

plt.figure(figsize=(9, 6))

for n in TAMANOS:

    plt.plot(
        tabla_interaccion.index,
        tabla_interaccion[n],
        marker="o",
        linewidth=2,
        label=f"n = {n}"
    )

plt.title("Interacción entre temperatura y tamaño")
plt.xlabel("Nivel de temperatura")
plt.ylabel("Costo promedio")
plt.legend(title="Tamaño TSP")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICAS / "interaccion.png",
    dpi=300
)

plt.close()

# =====================================================
# 13. GRAFICA DE RESULTADOS POR TEMPERATURA
# =====================================================

for n in TAMANOS:

    datos_n = df[df["tamano"] == n]
    plt.figure(figsize=(8, 5))

    for nivel in ["Bajo", "Medio", "Alto"]:

        datos_nivel = datos_n[
            datos_n["temperatura"] == nivel
        ]["costo"]

        plt.scatter(
            [temperaturas[nivel]] * len(datos_nivel),
            datos_nivel,
            label=nivel
        )

    plt.xlabel("Temperatura inicial")
    plt.ylabel("Costo de la ruta")
    plt.title(f"Resultados de Recocido Simulado - n = {n}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        CARPETA_GRAFICAS / f"resultados_n_{n}.png",
        dpi=300
    )

    plt.close()

# =====================================================
# 14. RESUMEN FINAL
# =====================================================

print("\n========== RESUMEN FINAL ==========")

print(f"Total de corridas: {len(df)}")

print("\nTemperaturas utilizadas:")

for nivel, temperatura in temperaturas.items():
    print(f"{nivel}: {temperatura:.6f}")

print("\nTabla de mejores temperaturas:")

print(df_resumen.to_string(index=False))

print("\nArchivos generados correctamente.")

print(f"Resultados: {CARPETA_RESULTADOS}")
print(f"Graficas: {CARPETA_GRAFICAS}")