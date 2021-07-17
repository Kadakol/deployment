from fastapi import UploadFile
from pydantic import BaseModel

from tensorflow.keras.models import load_model

import os
import cv2
import numpy as np

class AlexNetCatDogClassifier():
	"""Handle details related inference using trained AlexNet model"""
	def __init__(self):
		model_path = os.path.join('models', 'alexnet.h5')
		self.loaded_model = load_model(model_path)

	async def classify(self, file):
		# print(file)
		# print(type(file))
		contents = await file.read()
		# print(type(contents))
		nparr = np.fromstring(contents, np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		img = cv2.resize(img, (227, 227))
		img = img[...,::-1].astype(np.float32)
		img = np.expand_dims(img, axis=0)
		# print(f'image_attrs = shape: {img.shape}, len: {len(contents)}')
		output = self.loaded_model.predict(img)
		# print(output)

		response = {'filename':file.filename, 'content_type':file.content_type}
		if output[0][0] > output[0][1]:
			response['predicted_class'] ='cat'
			response['confidence'] = float(output[0][0])
		else:
			response['predicted_class'] ='dog'
			response['confidence'] = float(output[0][1])

		return response

