class Transaction(object):

    def __init__(self, index, sender, receiver, amount):
        # Store internally
        self.index = index
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

    def to_dict(self):
        # Transform object into a dictionary for future transformation in JSON
        # The names of the fields are the name of the variables
        return self.__dict__
