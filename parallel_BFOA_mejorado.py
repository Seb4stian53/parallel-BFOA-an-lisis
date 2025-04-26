from multiprocessing import Manager
import time
from bacteria_mejorada import bacteria
import numpy
from fastaReader import fastaReader
import copy
import pandas as pd
from datetime import datetime
import os

def ejecutar_bfoa(config, secuencias, nombres, ejecucion):
    """Ejecuta una instancia completa del algoritmo"""
    resultados = {
        'ejecucion': ejecucion,
        'mejor_bacteria': None,
        'fitness': None,
        'blosum': None,
        'interaccion': None,
        'nfe': None,
        'tiempo': 0,
        'mejor_alineamiento': None
    }
    
    manager = Manager()
    poblacion = manager.list([tuple(copy.deepcopy(seq) for seq in secuencias) for _ in range(config['bacterias'])])
    
    operador = bacteria(config['bacterias'])
    
    if config['usar_perfil']:
        print(f"Ejecución {ejecucion}: Calculando perfil de conservación...")
        operador.calcula_perfil_conservacion(secuencias)
    
    start_time = time.time()
    
    for iter in range(config['iteraciones']):
        print(f"Ejecución {ejecucion}: Iteración {iter+1}/{config['iteraciones']}")
        
        operador.tumbo(len(secuencias), poblacion, config['tumbo'], config['usar_perfil'])
        operador.cuadra(len(secuencias), poblacion)
        operador.creaGranListaPares(poblacion)
        operador.evaluaBlosum()
        operador.creaTablasAtractRepel(poblacion, config['dAttr'], config['wAttr'], config['dAttr'], config['wRep'])
        operador.creaTablaInteraction()
        operador.creaTablaFitness()
        
        best_data = operador.obtieneBest(operador.getNFE())
        
        operador.replaceWorst(poblacion, best_data['indice'])
        operador.resetListas(config['bacterias'])
    
    resultados['tiempo'] = time.time() - start_time
    resultados['mejor_bacteria'] = best_data['indice']
    resultados['fitness'] = best_data['fitness']
    resultados['blosum'] = best_data['blosum']
    resultados['interaccion'] = best_data['interaccion']
    resultados['nfe'] = best_data['nfe']
    resultados['mejor_alineamiento'] = copy.deepcopy(poblacion[best_data['indice']])
    
    # Validación final
    if numpy.isinf(resultados['fitness']) or numpy.isnan(resultados['fitness']):
        print(f"¡Advertencia en ejecución {ejecucion}: Fitness inválido, usando valor BLOSUM!")
        resultados['fitness'] = best_data['blosum']
    
    return resultados

def main():
    config = {
        'bacterias': 4,
        'iteraciones': 3,
        'tumbo': 2,
        'nado': 3,
        'dAttr': 0.1,
        'wAttr': 0.002,
        'wRep': 0.001,
        'usar_perfil': True
    }
    
    print("\n=== Algoritmo BFOA Mejorado ===")
    print("=== 30 Ejecuciones con Reporte Detallado ===")
    
    # Cargar secuencias
    try:
        lector = fastaReader()
        secuencias = [list(seq) for seq in lector.seqs]
        nombres = lector.names
        print(f"\nSecuencias cargadas: {len(secuencias)}")
        for i, (nombre, seq) in enumerate(zip(nombres, secuencias)):
            print(f"Secuencia {i+1}: {nombre} ({len(seq)} residuos)")
    except Exception as e:
        print(f"\nError cargando secuencias: {e}")
        return
    
    # Crear directorio para resultados si no existe
    if not os.path.exists('resultados'):
        os.makedirs('resultados')
    
    # DataFrame para resultados
    columnas = [
        'Ejecución',
        'Mejor Bacteria',
        'Fitness',
        'BLOSUM Score',
        'Interacción',
        'NFE',
        'Tiempo (s)'
    ]
    df_resultados = pd.DataFrame(columns=columnas)
    
    # Ejecutar 30 veces
    for ejecucion in range(1, 31):
        print(f"\n\n=== EJECUCIÓN {ejecucion}/30 ===")
        resultados = ejecutar_bfoa(config, secuencias, nombres, ejecucion)
        
        # Guardar en DataFrame
        df_resultados.loc[ejecucion-1] = [
            ejecucion,
            resultados['mejor_bacteria'],
            resultados['fitness'],
            resultados['blosum'],
            resultados['interaccion'],
            resultados['nfe'],
            resultados['tiempo']
        ]
        
        # Guardar alineamiento
        with open(f"resultados/alineamiento_ejecucion_{ejecucion}.fasta", 'w') as f:
            for nombre, sec in zip(nombres, resultados['mejor_alineamiento']):
                f.write(f">{nombre}\n{''.join(sec)}\n")
    
    # Generar reporte Excel
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_reporte = f"resultados/reporte_BFOA_30ejecuciones_{fecha}.xlsx"
    
    with pd.ExcelWriter(nombre_reporte) as writer:
        df_resultados.to_excel(writer, sheet_name='Resultados', index=False)
        
        # Resumen estadístico
        resumen = df_resultados.describe()
        resumen.to_excel(writer, sheet_name='Estadísticas')
    
    print(f"\n=== Reporte generado: {nombre_reporte} ===")
    print("\nResumen de ejecuciones:")
    print(df_resultados.head())
    
    # Guardar configuración
    with open(f"resultados/configuracion_{fecha}.txt", 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nSecuencias: {len(secuencias)}")
        f.write(f"\nTotal ejecuciones: 30")

if __name__ == "__main__":
    main()