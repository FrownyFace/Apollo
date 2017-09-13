import tensorflow as tf

class Apollo:
    def __init__(self, params):

    def train(self):

    def build_model(self):

if __name__ == '__main__':

    params = {
        'gen_layers': [[9, 50, 'gen_W1'],
                       [50, 50, 'gen_W2'],
                       [50, 1, 'gen_W3']],

        'dis_layers': [[9, 50, 'dis_W1'],
                       [50, 1, 'dis_W2']],

        'lr': 0.001,
        'epochs': 100,
        'train_path': './data/something',
        'test_path': 'test_o.csv',
        'batch_size': 25,
        'export_base_path': './export_models',

    }
    d = GAN(params)
    d.train()
