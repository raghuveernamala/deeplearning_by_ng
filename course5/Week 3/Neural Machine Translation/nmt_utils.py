import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from keras.utils import to_categorical
import keras.backend as K


fake = Faker()
fake.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']

def create_date():
    """
        Creates some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS), locale=random.choice(LOCALES))

        case_change = random.choice([0,1,2])
        if case_change == 1:
            human_readable = human_readable.upper()
        elif case_change == 2:
            human_readable = human_readable.lower()
        # if case_change == 0, do nothing

        machine_readable = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def create_dataset(n_examples):
    """
        Creates a dataset with n_examples and vocabularies
        :n_examples: the number of examples to generate
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []

    for i in tqdm(range(n_examples)):
        h, m, _ = create_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))

    human = dict(zip(list(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(list(machine_vocab) + ['<unk>', '<pad>']))
    machine = {v:k for k,v in inv_machine.items()}
 
    Tx=20
    # Extract inputs and targets from tuples in your dataset
    sources, targets = zip(*dataset)
    # Preprocess your data
    sources = np.array([string_to_int(i, Tx, human) for i in sources])
    targets = [string_to_int(t, Tx, machine) for t in targets]
    targets = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine)), targets)))
    
    return dataset, human, machine, inv_machine, sources, targets



def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    return rep


def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
    
    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
    
    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """
    
    l = [inv_vocab[i] for i in ints]
    return l


EXAMPLES = ['3 May 1979', '5 Apr 09', '20th February 2016', 'Wed 10 Jul 2007']

def run_example(model, input_vocabulary, inv_output_vocabulary, text):
    encoded = string_to_int(text, Tx, input_vocabulary)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return int_to_string(prediction, inv_output_vocabulary)

def run_examples(model, input_vocabulary, inv_output_vocabulary, examples=EXAMPLES):
    predicted = []
    for example in examples:
        predicted.append(''.join(run_example(model, input_vocabulary, inv_output_vocabulary, example)))
        print('input:', example)
        print('output:', predicted[-1])
    return predicted


def get_data_recurrent(m, Tx, n_h, attention_column=None):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param m: the number of samples to retrieve.
    :param Tx: the number of time steps of your series.
    :param n_h: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    if attention_column is None:
        attention_column = np.random.randint(low=0, high=Tx)
    x = np.random.standard_normal(size=(m, Tx, n_h))
    y = np.random.randint(low=0, high=2, size=(m, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, n_h))

    return x, y

def get_activations(model, inputs, layer_name=None):
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations