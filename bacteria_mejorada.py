import math
from multiprocessing import Manager, Pool
from pickle import FALSE, TRUE
from evaluadorBlosum import evaluadorBlosum
import numpy
from fastaReader import fastaReader
import random
import copy
import concurrent.futures

class bacteria():
    
    def __init__(self, numBacterias):
        manager = Manager()
        self.blosumScore = manager.list([0.0]*numBacterias)
        self.tablaAtract = manager.list([0.0]*numBacterias)
        self.tablaRepel = manager.list([0.0]*numBacterias)
        self.tablaInteraction = manager.list([0.0]*numBacterias)
        self.tablaFitness = manager.list([0.0]*numBacterias)
        self.granListaPares = manager.list([[]]*numBacterias)
        self.NFE = manager.list([0]*numBacterias)
        self.perfilConservacion = None

    def resetListas(self, numBacterias):
        manager = Manager()
        self.blosumScore = manager.list([0.0]*numBacterias)
        self.tablaAtract = manager.list([0.0]*numBacterias)
        self.tablaRepel = manager.list([0.0]*numBacterias)
        self.tablaInteraction = manager.list([0.0]*numBacterias)
        self.tablaFitness = manager.list([0.0]*numBacterias)
        self.granListaPares = manager.list([[]]*numBacterias)
        self.NFE = manager.list([0]*numBacterias)

    def calcula_perfil_conservacion(self, secuencias):
        max_len = max(len(seq) for seq in secuencias)
        perfil = []
        
        for i in range(max_len):
            columna = []
            for seq in secuencias:
                if i < len(seq):
                    columna.append(seq[i])
                else:
                    columna.append('-')
            
            frecuencias = {}
            total = 0
            for aa in columna:
                if aa != '-':
                    frecuencias[aa] = frecuencias.get(aa, 0) + 1
                    total += 1
            
            if not frecuencias or total == 0:
                perfil.append(1.0)
                continue
                
            entropia = 0
            for count in frecuencias.values():
                p = count / total
                entropia -= p * math.log2(p) if p > 0 else 0
            
            perfil.append(entropia)
        
        perfil = numpy.array(perfil)
        min_val = numpy.min(perfil)
        max_val = numpy.max(perfil)
        perfil = (perfil - min_val) / (max_val - min_val + 1e-10)
        self.perfilConservacion = perfil
        return perfil

    def cuadra(self, numSec, poblacion):
        for i in range(len(poblacion)):
            bacterTmp = list(poblacion[i])
            maxLen = max(len(seq) for seq in bacterTmp[:numSec])
            
            for t in range(numSec):
                gap_count = maxLen - len(bacterTmp[t])
                if gap_count > 0:
                    bacterTmp[t].extend(["-"] * gap_count)
                    poblacion[i] = tuple(bacterTmp)

    def tumbo(self, numSec, poblacion, numGaps, usar_perfil=False):
        for i in range(len(poblacion)):
            bacterTmp = list(poblacion[i])
            
            for _ in range(numGaps):
                seqnum = random.randint(0, len(bacterTmp)-1)
                
                if usar_perfil and self.perfilConservacion is not None and len(bacterTmp[seqnum]) > 0:
                    prob_gaps = numpy.array(self.perfilConservacion[:len(bacterTmp[seqnum])])
                    prob_gaps = numpy.nan_to_num(prob_gaps / numpy.sum(prob_gaps), nan=0.0)
                    pos = numpy.random.choice(len(prob_gaps), p=prob_gaps)
                else:
                    pos = random.randint(0, len(bacterTmp[seqnum])-1)
                
                bacterTmp[seqnum].insert(pos, "-")
            
            poblacion[i] = tuple(bacterTmp)

    def creaGranListaPares(self, poblacion):   
        for i in range(len(poblacion)):
            pares = []
            bacterTmp = list(poblacion[i])
            
            for j in range(len(bacterTmp[0])):
                column = [seq[j] for seq in bacterTmp]
                pares.extend(self.obtener_pares_unicos(column))
            
            self.granListaPares[i] = pares

    def evaluaFila(self, fila, num):
        evaluador = evaluadorBlosum()
        score = 0.0
        for par in fila:
            try:
                score += float(evaluador.getScore(par[0], par[1]))
            except:
                score += -8.0
        self.blosumScore[num] = score
    
    def evaluaBlosum(self):
        with Pool() as pool:
            args = [(copy.deepcopy(self.granListaPares[i]), i) for i in range(len(self.granListaPares))]
            pool.starmap(self.evaluaFila, args)

    def obtener_pares_unicos(self, columna):
        pares_unicos = set()
        for i in range(len(columna)):
            for j in range(i+1, len(columna)):
                par = tuple(sorted([columna[i], columna[j]]))
                pares_unicos.add(par)
        return list(pares_unicos)

    def compute_diff(self, args):
        indexBacteria, otherBlosumScore, self.blosumScore, d, w = args
        try:
            diff = (float(self.blosumScore[indexBacteria]) - float(otherBlosumScore)) ** 2.0
            exp_val = numpy.clip(w * diff, -100, 100)
            return float(d) * numpy.exp(exp_val)
        except:
            return 0.0

    def compute_cell_interaction(self, indexBacteria, d, w, atracTrue):
        with Pool() as pool:
            args = [(indexBacteria, otherBlosumScore, self.blosumScore, d, w) 
                   for otherBlosumScore in self.blosumScore]
            results = pool.map(self.compute_diff, args)
        
        total = numpy.clip(sum(results), -1e10, 1e10)
        
        if atracTrue:
            self.tablaAtract[indexBacteria] = total
        else:
            self.tablaRepel[indexBacteria] = total
        
    def creaTablaAtract(self, poblacion, d, w):
        for indexBacteria in range(len(poblacion)):
            self.compute_cell_interaction(indexBacteria, d, w, TRUE)

    def creaTablaRepel(self, poblacion, d, w):
        for indexBacteria in range(len(poblacion)):
            self.compute_cell_interaction(indexBacteria, d, w, FALSE)
    
    def creaTablasAtractRepel(self, poblacion, dAttr, wAttr, dRepel, wRepel):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.creaTablaAtract, poblacion, dAttr, wAttr)
            executor.submit(self.creaTablaRepel, poblacion, dRepel, wRepel)

    def creaTablaInteraction(self):
        for i in range(len(self.tablaAtract)):
            try:
                inter = float(self.tablaAtract[i]) + float(self.tablaRepel[i])
                self.tablaInteraction[i] = numpy.clip(inter, -1e10, 1e10)
            except:
                self.tablaInteraction[i] = 0.0

    def creaTablaFitness(self):
        for i in range(len(self.tablaInteraction)):
            try:
                blosum = float(self.blosumScore[i])
                inter = float(self.tablaInteraction[i])
                self.tablaFitness[i] = numpy.clip(blosum + inter, -1e10, 1e10)
            except:
                self.tablaFitness[i] = float(self.blosumScore[i])
    
    def getNFE(self):
        return sum(self.NFE)
        
    def obtieneBest(self, globalNFE):
        bestIdx = 0
        for i in range(1, len(self.tablaFitness)):
            if float(self.tablaFitness[i]) > float(self.tablaFitness[bestIdx]):
                bestIdx = i
        
        best_data = {
            'indice': bestIdx,
            'fitness': float(self.tablaFitness[bestIdx]),
            'blosum': float(self.blosumScore[bestIdx]),
            'interaccion': float(self.tablaInteraction[bestIdx]),
            'nfe': globalNFE
        }
        return best_data

    def replaceWorst(self, poblacion, best_idx):
        worst = 0
        for i in range(1, len(self.tablaFitness)):
            if float(self.tablaFitness[i]) < float(self.tablaFitness[worst]):
                worst = i
        poblacion[worst] = copy.deepcopy(poblacion[best_idx])