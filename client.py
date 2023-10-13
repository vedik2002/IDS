import tensorflow as tf
import flwr as fl
from flwr.common.parameter import parameters_to_ndarrays

model = tf.keras.applications.MobileNetV2((32,32,3),classes=10,weights=None)
model.compile("adam", "sparse_categorical_crossentropy",metrics = ["accuracy"])
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

model.fit(x_train,y_train,epochs=30,batch_size=32)


class FlowerClient(fl.client.NumPyClient):
  #Fit - function for client
  def fit(self,parameters,config): #Parameters - > parameters from the server to the client in one round
  #Config - > Dictionary of Strings, passed from server to client
    model.set_weights(parameters)
    model.fit(x_train,y_train,epochs=1,batch_size=32)
    return model.get_weights(), len(x_train), {"empty":0}#Dictionary containing accuracy or any metrics to be sent to server"
  #length of training data also needs to be sent to server so as to aggregate the data differently in the server
  def evaluate(self parameters, config): #used once the server has aggregated all the parameters of clients after training, it will then send back the aggregated weights to all the clients to evaluate this new model on their own data
    model.set_weights(parameters)
    #model.evaluate()
    loss,accuracy = model.evaluate(x_test,y_test)
    return loss,len(x_test),{"accuracy":accuracy}
