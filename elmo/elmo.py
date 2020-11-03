import mxnet as mx
import gluonnlp as nlp


class ELMo:

    def __init__(self, model='elmo_2x1024_128_2048cnn_1xhighway',
                 tokenizer=None,
                 ctx=mx.cpu()):
        """
        Parameters
        ----------
        model: str, default 'elmo_2x1024_128_2048cnn_1xhighway'
            One of the following ELMo models:
            1. elmo_2x1024_128_2048cnn_1xhighway
            2. elmo_2x2048_256_2048cnn_1xhighway
            3. elmo_2x4096_512_2048cnn_2xhighway
        tokenizer: default `gluonnlp.data.SacreMosesTokenizer`
            Sentence tokenizer, see `gluonnlp.data.transforms`.
        ctx: mxnet.context, default `mx.cpu()`
        """
        self.tokenizer = tokenizer or nlp.data.SacreMosesTokenizer()
        self.vocab = nlp.vocab.ELMoCharVocab()
        self.elmo_bilm, _ = nlp.model.get_model(model,
                                                dataset_name='gbw',
                                                pretrained=True,
                                                ctx=ctx)

    def embedding(self, sentence_list):
        """
        Parameters
        ----------
        sentence_list: list of str

        Returns
        -------
        output : list of NDArray
            A list of activations at each layer of the network, each of shape
            (batch_size, sequence_length, embedding_size)
        hidden_state : (list of list of NDArray, list of list of NDArray)
            The states. First tuple element is the forward layer states, while the second is
            the states from backward layer. Each is a list of states for each layer.
            The state of each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
        """
        data, valid_lengths = [], []
        for sentence in sentence_list:
            x = ['<bos>'] + self.tokenizer(sentence) + ['<eos>']
            data.append(self.vocab[x])
            valid_lengths.append(len(x))
        data = nlp.data.batchify.Pad(pad_val=0)(data)
        valid_lengths = nlp.data.batchify.Stack()(valid_lengths)

        batch_size = len(sentence_list)
        length = data.shape[1]
        hidden_state = self.elmo_bilm.begin_state(mx.nd.zeros, batch_size=batch_size)
        mask = mx.nd.arange(length).expand_dims(0).broadcast_axes(axis=(0,), size=(batch_size,))
        mask = mask < valid_lengths.expand_dims(1).astype('float32')
        output, hidden_state = self.elmo_bilm(data, hidden_state, mask)
        return output, hidden_state
