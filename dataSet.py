
class DataSet:

    def __init__(self, samples, labels, input_length, target_length, batch_size):
        self.batch_size = batch_size
        self.samples = samples[:len(samples) - (len(samples) % self.batch_size)].reshape((-1, self.batch_size, input_length))
        self.labels  = labels[:len(labels) - (len(labels) % self.batch_size)].reshape((-1, self.batch_size, target_length))
        self.data_set = zip(self.samples, self.labels)
    
    def __iter__(self):
        for batch in self.data_set:
            yield batch

