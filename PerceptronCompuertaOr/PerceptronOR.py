#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------
# Nombre:            perceptron-or.py
#
# Descripción:       Perceptrón simple de una sola capa
#
# Autor:             José Christian Padilla Navarro
#
# Creado el:         19 de julio del 2021
#-------------------------------------------------------------------

#----------------------------
#    Tabla de entrada (OR):
#----------------------------
#
#   *****************
#   * X_1 * X_2 * d *
#   *****************
#   *  0  *  0  * 0 *
#   *  0  *  1  * 1 * 
#   *  1  *  0  * 1 *
#   *  1  *  1  * 1 *
#   *****************

class Perceptron():

	def __init__(self):
		
		#Valores iniciales de la tabla de entrada
		self.x1 = ([0,0,1,1])
		self.x2 = ([0,1,0,1])
		self.d  = ([0,1,1,1])
		self.numeromuestras = len(self.d)
		
		#Valores iniciales para los pesos
		self.w1 = 0.3
		self.w2 = 0.5

		#Valor del umbral (theta)
		self.umbral = 0.4
		
		#Valor del factor de aprendizaje 
		#(si es muy alto puede ser que no termine de aprender bien, si es muy bajo puede tardar mucho)
		self.aprendizaje = 0.2
		
		#Valor inicial de las epochs (cada epoch es un ajuste en los pesos y el umbral)
		self.epochs = 0
		
		#Imprimo los valorres iniciales
		print "********************************************"
		print "Total de epochs: ", self.epochs	
		print "w1: ", self.w1
		print "w2: ", self.w2
		print "umbral: ", self.umbral
		
		#Arranca el entrenamiento de la red
		self.entrenamiento()
			
			
	def entrenamiento(self):
		
		errores = True
		
		while errores:
			
			for i in range (self.numeromuestras):
				
				#Calcular el valor de z
				z = ((self.x1[i]*self.w1) + (self.x2[i] * self.w2)) - self.umbral
				
				if z>=0:
					z=1
				else:
					z=0
				
				#Calculamos el error 
				#(para este caso específico buscamos que el error sea 0, pero normalmente debe rondar el 0.1, depende lo que se busque)
			        error = (self.d[i]-z)
			        
				
				if error>0:
					
					#Sí encontramos que existe error
					errores = True
					
					#Ajustamos el umbral (theta)
					self.umbral = self.umbral + (-(self.aprendizaje * error))
					
					#Ajustamos los pesos
					self.w1 = self.w1 + (self.x1[i] * error * self.aprendizaje)
					self.w2 = self.w2 + (self.x2[i] * error * self.aprendizaje)
					
					#Incrementamos las epochs 
					self.epochs = self.epochs + 1
										
					#Imprimo los nuevos valores
					print "********************************************"
					print "Número de epoch: ", self.epochs	
					print "w1: ", self.w1
					print "w2: ", self.w2
					print "umbral: ", self.umbral
									
				else:
					errores = False
							

if __name__=="__main__":
	Perceptron()
