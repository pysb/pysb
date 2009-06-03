import sys


class BngGenerator(object):

    def __init__(self, model):
        self.model = model
        self.__content = None

    def __get_content(self):
        if self.__content == None:
            self.generate_content()
        return self.__content

    content = property(fget=__get_content)

    def generate_content(self):
        self.__content = ''
        self.generate_parameters()
        #self.generate_species()
        #self.generate_rules()

    def generate_parameters(self):
        self.__content += "begin parameters\n"
        for p in self.model.parameters:
            self.__content += "  %20s %f\n" % (p.name, p.value)
        self.__content += "end parameters\n"
