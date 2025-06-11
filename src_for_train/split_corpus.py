import pickle

dataset_path = '../kb_wikipedia_mixed_rd.pickle' # path to compmix-ir corpus
num_splits = 8

# Load the dataset
with open(dataset_path, "rb") as fIn:
    data = pickle.load(fIn)

corpus_sentences = [item["evidence_text"] for item in data]

# Split the corpus into 8 parts
split_size = len(corpus_sentences) // num_splits
for i in range(num_splits):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i != num_splits - 1 else len(corpus_sentences)
    split_sentences = corpus_sentences[start_idx:end_idx]
    split_path = f'../compmix_corpus_8_splits/kb_wikipedia_mixed_rd_split_{i}.pickle' # path to compmix-ir splited small corpora
    with open(split_path, 'wb') as fOut:
        pickle.dump(split_sentences, fOut)

