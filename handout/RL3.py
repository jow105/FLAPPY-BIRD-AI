from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import random


PIPEGAPSIZE  = 100
BIRDHEIGHT = 24

class QNet(object):

	def __init__(self):
		"""
		Initialize neural net here.

		Args:
			num_inputs: Number of nodes in input layer
			num_hidden1: Number of nodes in the first hidden layer
			num_hidden2: Number of nodes in the second hidden layer
			num_output: Number of nodes in the output layer
			lr: learning rate
		"""
		self.num_inputs = 2
		self.num_hidden1 = 10
		self.num_hidden2 = 10
		self.num_output = 2
		self.lr = 0.001
		self.states_withreward = []
		self.build()

	def build(self):
		"""
		Builds the neural network using keras, and stores the model in self.model.
		Uses shape parameters from init and the learning rate self.lr.
		"""
		model = Sequential()
		model.add(Dense(self.num_hidden1, init='lecun_uniform', input_shape=(self.num_inputs,)))
		model.add(Activation('relu'))

		model.add(Dense(self.num_hidden2, init='lecun_uniform'))
		model.add(Activation('relu'))

		model.add(Dense(self.num_output, init='lecun_uniform'))
		model.add(Activation('linear'))

		rms = RMSprop(lr=self.lr)
		model.compile(loss='mse', optimizer=rms)
		self.model = model


	def flap(self, input_data):
		"""
		Use the neural net as a Q function to act.
		Use self.model.predict to do the prediction.

		Args:
			input_data (Input object): contains information you may use about the 
			current state.

		Returns:
			(choice, prediction, debug_str): 
				choice (int) is 1 if bird flaps, 0 otherwise. Will be passed
					into the update function below.
				prediction (array-like) is the raw output of your neural network,
					returned by self.model.predict. Will be passed into the update function below.
				debug_str (str) will be printed on the bottom of the game
		"""

		# state = your state in numpy array
		# prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0]
		# choice = make choice based on prediction
		# debug_str = ""
		# return (choice, prediction, debug_str)
		
		state = np.array([(input_data.distX, input_data.distY)])
		prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0] # Q

		if(np.max(prediction) == prediction[0]):
			choice = 1
		if(np.max(prediction) == prediction[1]):
			choice = 0

		debug_str = ""
		
		return (choice,prediction,debug_str)


	def update(self, last_input, last_choice, last_prediction, crash, scored, playerY, pipVelX):
		"""
		Use Q-learning to update the neural net here
		Use self.model.fit to back propagate

		Args:
			last_input (Input object): contains information you may use about the
				input used by the most recent flap() 
			last_choice: the choice made by the most recent flap()
			last_prediction: the prediction made by the most recent flap()
			crash: boolean value whether the bird crashed
			scored: boolean value whether the bird scored
			playerY: y position of the bird, used for calculating new state
			pipVelX: velocity of pipe, used for calculating new state

		Returns:
			None
		"""
		# This is how you calculate the new (x,y) distances
		# new_distX = last_input.distX + pipVelX
		# new_distY = last_input.pipeY - playerY

		# state = compute new state in numpy array
		# reward = compute your reward
		# prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)

		# update old prediction from flap() with reward + gamma * np.max(prediction)
		# record updated prediction and old state in your mini-batch
		# if batch size is large enough, back propagate
		# self.model.fit(old states, updated predictions, batch_size=size, epochs=1)
		
		new_distX = last_input.distX + pipVelX
		new_distY = last_input.pipeY - playerY

		state = np.array([(new_distX,new_distY)])

		gamma = 0.1
		reward_f = 0
		reward_nf = 0

		if(crash):
			if(last_choice == 0):
				reward_nf += -10000
			if(last_choice == 1):
				reward_f += -10000
		elif(scored):
			if(last_choice == 0):
				reward_nf += 100
			if(last_choice == 1):
				reward_f += 100
		else:
			if(new_distY >= -40 and new_distY <= 30):
				if(new_distX <= 50 and new_distX >= 25):
					if(last_choice == 0):
						reward_nf += 5
					if(last_choice == 1):
						reward_f += -10
			else:
				if(last_choice == 0):
					reward_nf += 0.0000826211*new_distY**3 + 0.00780627*new_distY**2 - 0.0293447*new_distY -23.3761
				if(last_choice == 1):
					reward_f += -0.000212251*new_distY**3 - 0.0220655*new_distY**2 + 0.055207*new_distY + 38.9316

		prediction = self.model.predict(state.reshape(1,self.num_inputs), batch_size = 1) # not state, change so that its the R-value estimates


		if(last_choice == 0):
			last_prediction[1] = reward_nf + gamma*np.max(prediction)
		if(last_choice == 1):
			last_prediction[0] = reward_f + gamma*np.max(prediction)

		old_state = np.array([(last_input.distX,last_input.distY)])

		# array of old states and predictions
		old_states = []
		updated_predictions = []

		self.states_withreward += [((last_input.distX,last_input.distY),last_prediction)]

		for s in self.states_withreward:
			(dists, R) = s
			old_states += [dists]
			updated_predictions += [R]

		old_states_numpy = np.asarray(old_states)
		updated_predictions_numpy = np.asarray(updated_predictions)

		mini_batch = 400

		if(len(self.states_withreward) > mini_batch):
			indices = random.sample(xrange(len(self.states_withreward)),50)
			x = []
			y = []

			for i in indices:
				x += [old_states[i]]
				y += [updated_predictions[i]]

			self.model.fit(np.asarray(x),np.asarray(y),batch_size=50,epochs=1)

			self.states_withreward = self.states_withreward[1:]
		
		return None

class Input:
	def __init__(self, playerX, playerY, pipeX, pipeY,
				distX, distY):
		"""
		playerX: x position of the bird
		playerY: y position of the bird
		pipeX: x position of the next pipe
		pipeY: y position of the next pipe
		distX: x distance between the bird and the next pipe
		distY: y distance between the bird and the next pipe
		"""
		self.playerX = playerX
		self.playerY = playerY
		self.pipeX = pipeX
		self.pipeY = pipeY
		self.distX = distX
		self.distY = distY

