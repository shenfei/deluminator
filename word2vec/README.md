Word Embedding
=====

Tested under TensorFlow 0.12.0

### Training

```
python word2vec.py \
    --corpus_list corpus/file/name/list \
    --train_dir path/to/save/train/data \
    --save_path path/to/save/model
```

`--corpus_list`: specify corpus. A file with each line of a corpus file path.
                 Make sure there is no duplicated names.
                 Each file is a space seperated text file.
`--train_dir`: path to store training TFRecord files.
`--save_path`: specify results store path.

Use `--help` to see other options and refer to source code for details.

### Use pretrained results

Use `pickle.load` to load pretrained results. There are four python objects:

- `id2word`: word list
- `word2id`: dict from word to id
- `word_freq`: word frequency list, with the same order of `id2word`
- `embedding`: numpy.ndarray, word embedding vectors
