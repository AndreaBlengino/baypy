class Model:


    def __init__(self):

        self.data = None
        self.y_name = None
        self.initial_values = None
        self.priors = None
        self.variable_names = None


    def set_data(self, data, y_name):

        self.data = data
        self.y_name = y_name


    def set_initial_values(self, values):

        self.initial_values = values


    def set_priors(self, priors):

        self.priors = priors
        self.variable_names = list(self.priors.keys())
        self.variable_names.insert(0, self.variable_names.pop(self.variable_names.index('intercept')))
