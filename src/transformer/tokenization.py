
import os
import glob
import regex as re
import numpy as np
from typing import BinaryIO, IO, Any
from jaxtyping import Bool, Float, Int
from tqdm import tqdm

class Tokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.rev_vocab = {value: key for key, value in vocab.items()}
        self.merges = merges
        self.rev_merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens.sort(key=lambda x: len(x), reverse=True) 
        
    def encode(self, string: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if self.special_tokens:
            re_specials = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            string = re.split(re_specials, string)
        else:
            string = [string]

        indices = []
        for substring in string:
            pretokens = []        
            if substring in self.special_tokens:
                indices.append(self.rev_vocab[substring.encode('utf-8')])
                continue
            
            for pretoken in re.finditer(PAT, substring):
                pretoken = pretoken.group(0)
                pretoken = [bytes([s]) for s in pretoken.encode('utf-8')]
                pretokens.append(pretoken)
            
            for pretoken in pretokens:
                indxs = [self.rev_vocab[b] for b in pretoken]
                indices.extend(self.merge(indxs))

        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8", errors="replace")
        return string
    
    def merge(self, indices: list[int]) -> list[int]:
        l = len(indices)
        pairs = [(self.vocab[indices[i]], self.vocab[indices[i+1]]) for i in range(l-1)]
        ranks = [self.rev_merges.get(pair, float("inf")) for pair in pairs]
        while True:
            if l <= 1:
                return indices
            min_rank = min(ranks)
            if min_rank == float("inf"):
                break
            min_index = ranks.index(min_rank)
            
            if min_index == 0:
                new_index = self.rev_vocab.get((pairs[min_index][0] + pairs[min_index][1]))
                indices = [new_index] + indices[2:]
            elif min_index == l - 2:
                new_index = self.rev_vocab.get((pairs[min_index][0] + pairs[min_index][1]))
                indices = indices[:-2] + [new_index]
            else:
                new_index = self.rev_vocab.get((pairs[min_index][0] + pairs[min_index][1]))
                indices = indices[:min_index] + [new_index] + indices[min_index+2:]
            
            l -= 1
            pairs = [(self.vocab[indices[i]], self.vocab[indices[i+1]]) for i in range(l - 1)]
            ranks = [self.rev_merges.get(pair, float("inf")) for pair in pairs]

        return indices
        
    def encode_iterable(self, iterable: list[str]) -> list[int]:
        result = []
        lines = iter(iterable)
        for line in tqdm(lines):
            result.extend(self.encode(line))
        return result
    
    def to_files(self, vocab_filepath: str, merges_filepath: str) -> None:
        with open(vocab_filepath, "w", encoding="latin-1") as f:
            f.write(str({k: v.decode("latin-1") for k, v in self.vocab.items()}))
        with open(merges_filepath, "w", encoding="latin-1") as f:
            for merge in self.merges:
                f.write(f"{merge[0].decode('latin-1')}|{merge[1].decode('latin-1')}||")

def from_files(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = None) -> "Tokenizer":
    with open(vocab_filepath, "r", encoding="latin-1") as f:
        vocab = {int(k): bytes(v.encode("latin-1")) for k, v in eval(f.read()).items()}
    with open(merges_filepath, "r", encoding="latin-1") as f:
        merges = [tuple(merge.split("|")) for merge in f.read().split("||")[:-1]]
        merges = [(bytes(a.encode("latin-1")), bytes(b.encode("latin-1"))) for a, b in merges]
    return Tokenizer(vocab, merges, special_tokens)
    
def find_chunk_boundaries(
    filepath: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    
    Taken directly from CS 336 Assignment 1 starter code
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    with open(filepath, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(filepath: str, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(filepath, "rb") as f:
        counts = {}
        f.seek(start)
        chunk = f.read(end - start)
        for token in special_tokens:
            chunk = chunk.split(token)
        for subchunk in chunk:
            subchunk = subchunk.decode("utf-8", errors="ignore")
            for pretoken in re.finditer(PAT, subchunk):
                pretoken = pretoken.group(0)
                pretoken = tuple([bytes([s]) for s in pretoken.encode('utf-8')])
                if pretoken not in counts.keys():
                    counts[pretoken] = 0
                counts[pretoken] += 1
        
    return counts

def train_bpe(input_file: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:  
    # Initialize vocabulary
    special_tokens = [bytes(b.encode("utf-8")) for b in special_tokens]
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token

    # Pretokenization
    print("Pretokenizing...")
    counts = []
    num_processes = 4
    parallelize = True
    if parallelize:
        from multiprocessing import Pool
        with Pool(processes=num_processes) as pool:
            boundaries = find_chunk_boundaries(input_file, num_processes, b"<|endoftext|>")
            results = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                results.append(pool.apply_async(pretokenize, (input_file, start, end, special_tokens)))
            for result in results:
                counts.append(result.get())
    else:
        boundaries = find_chunk_boundaries(input_file, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            counts.append(pretokenize(input_file, start, end, special_tokens))
    
    pretokenized_counts = {}
    for count in counts:
        for key in count.keys():
            if key not in pretokenized_counts:
                pretokenized_counts[key] = count[key]
            else:
                pretokenized_counts[key] += count[key]

    # Get pair counts
    pretoken_list = list(pretokenized_counts.keys())
    pair_counts = {}
    pair_indices = {}
    for i, pretoken in enumerate(pretoken_list):
        for j in range(len(pretoken) - 1):
            pair = (pretoken[j], pretoken[j+1])
            if pair not in pair_counts:
                pair_counts[pair] = 0
                pair_indices[pair] = []
            pair_counts[pair] += 1 * pretokenized_counts[pretoken]
            if i not in pair_indices[pair]:
                pair_indices[pair].append(i)
    
    # Compute merges   
    print("Computing merges...")
    merges = []
    l = len(vocab)
    while l < vocab_size:
        # Merge max pair count
        highest = max(pair_counts.values())
        highest_pairs = [key for key, value in pair_counts.items() if value == highest]
        to_merge = max(highest_pairs)
        merge = (to_merge, to_merge[0] + to_merge[1])
        vocab[l] = merge[1]
        merges.append(to_merge)

        # merge pair_counts directly
        lcount, rcount = 0, 0
        for pretoken_idx in list(pair_indices[to_merge]):
            pretoken = pretoken_list[pretoken_idx]
            idxs = [i for i in range(len(pretoken) - 1) if pretoken[i] == to_merge[0] and pretoken[i+1] == to_merge[1]]
            if not idxs:
                continue
            original = pretoken
            count = pretokenized_counts[pretoken]
            for i in range(len(idxs)):
                idx = idxs[i] - i
                if idx < 0 or idx > len(pretoken) - 2 or pretoken[idx] != to_merge[0] or pretoken[idx+1] != to_merge[1]:
                    continue
                new_token = pretoken[:idx] + (vocab[l],) + pretoken[idx+2:]
                if idx != 0:
                    lpair = (pretoken[idx-1], merge[1])
                    old_lpair = (pretoken[idx-1], to_merge[0])
                    if lpair not in pair_counts.keys():
                        pair_counts[lpair] = 0
                        pair_indices[lpair] = []
                    pair_counts[lpair] += count
                    if old_lpair in pair_counts:
                        pair_counts[old_lpair] -= count
                    if pretoken_idx not in pair_indices[lpair]:
                        pair_indices[lpair].append(pretoken_idx)
                if idx != len(pretoken) - 2:
                    rpair = (merge[1], pretoken[idx+2])
                    old_rpair = (to_merge[1], pretoken[idx+2])
                    if rpair not in pair_counts.keys():
                        pair_counts[rpair] = 0
                        pair_indices[rpair] = []
                    pair_counts[rpair] += count
                    if old_rpair in pair_counts:
                        pair_counts[old_rpair] -= count
                    if pretoken_idx not in pair_indices[rpair]:
                        pair_indices[rpair].append(pretoken_idx)
                pretoken_list[pretoken_idx] = new_token
                pretoken = new_token
                pretokenized_counts[new_token] = count
            del pretokenized_counts[original]
        del pair_indices[merge[0]]
        del pair_counts[merge[0]]
        l += 1
        
    return vocab, merges

def main():
    BASE_DIR = "."
    vocab_size = 10000
    input_path = f"{BASE_DIR}/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_filepath = f"{BASE_DIR}/data/tokenizer_vocab_{vocab_size}.json"
    merges_filepath = f"{BASE_DIR}/data/tokenizer_merges_{vocab_size}.txt"
    dataset_filepath = f"{BASE_DIR}/data/dataset_{vocab_size}.npy"

    if os.path.exists(vocab_filepath) and os.path.exists(merges_filepath):
        print("Loading tokenizer...")
        tokenizer = from_files(vocab_filepath, merges_filepath, special_tokens)
    else:
        print("Training tokenizer...")
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens)
        tokenizer.to_files(vocab_filepath, merges_filepath)
    
    print("Loading dataset...")
    if os.path.exists(dataset_filepath):
        dataset = np.load(dataset_filepath)
    else:
        dataset = tokenizer.encode_iterable(open(input_path, "r", encoding="utf-8"))
        np.save(dataset_filepath, np.array(dataset))

if __name__ == "__main__":
   main()