from os import path
import json

class ModelConfig:
    def __init__(
        self,
        activation_function,
        input_activation_function,
        output_activation_function,
        loss,
        optimizer
    ):
        self.activation_function = activation_function
        self.input_activation_function = input_activation_function
        self.output_activation_function = output_activation_function
        self.loss = loss
        self.optimizer = optimizer

    def isValid(self):
        return (
                self.activation_function is not None and 
                self.input_activation_function is not None and
                self.output_activation_function is not None and
                self.loss is not None and
                self.optimizer is not None
            )

    def __str__(self):
        return \
            "activation        : {} \n" \
            "input activation  : {} \n"  \
            "output activation : {} \n" \
            "loss              : {} \n" \
            "optimizer         : {} \n" \
        .format(
            self.activation_function,
            self.input_activation_function,
            self.output_activation_function,
            self.loss,
            self.optimizer
        )


class ModelConfigReader:
    def __init__(self): 
        self.configsAsJson = None
        self.configs = []
    
    def load(self, filename):
        if not path.exists(filename):
            raise ValueError("{} does not exists".format(filename))

        with open(filename, 'r') as stream:
            self.configsAsJson = json.load(stream)

        for configJson in self.configsAsJson['models']:
            configObject = ModelConfig(
                configJson['activation_function'],
                configJson['input_activation_function'],
                configJson['output_activation_function'],
                configJson['loss'],
                configJson['optimizer'],
            )

            if configObject.isValid():
                self.configs.append(configObject)
    
    def configsLength(self):
        return len(self.configs)