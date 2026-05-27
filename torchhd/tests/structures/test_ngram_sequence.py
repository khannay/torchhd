#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import torch
import string

from torchhd import structures, functional
from torchhd.tensors.map import MAPTensor

seed = 2147483644
letters = list(string.ascii_lowercase)


class TestNGramSequence:
    def test_creation_dim(self):
        S = structures.NGramSequence(10000)
        assert torch.allclose(S.value, torch.zeros(10000))
        assert S.n == 3
        assert len(S) == 0

    def test_creation_tensor(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(5, 10000, generator=generator)
        value = functional.ngrams(hv, n=3)
        S = structures.NGramSequence(value, n=3, size=5)
        assert torch.allclose(S.value, value)
        assert S.n == 3
        assert len(S) == 5

    def test_append_no_ngram_until_window_full(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(5, 10000, generator=generator)
        S = structures.NGramSequence(10000, n=3)

        S.append(hv[0])
        assert torch.allclose(S.value, torch.zeros(10000))
        assert len(S) == 1

        S.append(hv[1])
        assert torch.allclose(S.value, torch.zeros(10000))
        assert len(S) == 2

        S.append(hv[2])
        assert not torch.allclose(S.value, torch.zeros(10000))
        assert len(S) == 3

    def test_append_matches_ngrams(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.NGramSequence(10000, n=3)
        for i in range(5):
            S.append(hv[i])

        expected = functional.ngrams(hv[:5], n=3)
        assert torch.allclose(S.value, expected)

    def test_append_bigram(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.NGramSequence(10000, n=2)
        for i in range(4):
            S.append(hv[i])

        expected = functional.ngrams(hv[:4], n=2)
        assert torch.allclose(S.value, expected)

    def test_encode_ngram(self):
        hv = MAPTensor(
            [
                [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
            ]
        )
        S = structures.NGramSequence(8, n=3)
        e = S.encode_ngram(hv)

        # encode manually: permute(hv[0], 2) * permute(hv[1], 1) * permute(hv[2], 0)
        expected = functional.bind(
            functional.bind(
                functional.permute(hv[0], shifts=2),
                functional.permute(hv[1], shifts=1),
            ),
            functional.permute(hv[2], shifts=0),
        )
        assert torch.allclose(e, expected)

    def test_encode_ngram_consistent_with_append(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)
        S = structures.NGramSequence(10000, n=3)
        S.append(hv[0])
        S.append(hv[1])
        S.append(hv[2])

        e = S.encode_ngram(hv[:3])
        assert S.contains(e) > torch.tensor(0.9)

    def test_contains(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 1000, generator=generator)
        S = structures.NGramSequence(1000, n=3)
        for i in range(5):
            S.append(hv[i])

        present = S.encode_ngram(hv[1:4])
        absent = S.encode_ngram(hv[10:13])

        assert S.contains(present) > torch.tensor(0.5)
        assert S.contains(absent) < torch.tensor(0.5)

    def test_length(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 10000, generator=generator)
        S = structures.NGramSequence(10000, n=3)

        assert len(S) == 0
        S.append(hv[0])
        assert len(S) == 1
        S.append(hv[1])
        assert len(S) == 2
        S.append(hv[2])
        assert len(S) == 3

    def test_clear(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(8, 10000, generator=generator)
        S = structures.NGramSequence(10000, n=3)
        for i in range(5):
            S.append(hv[i])

        S.clear()
        assert torch.allclose(S.value, torch.zeros(10000))
        assert len(S) == 0
        assert S._buffer == []

    def test_from_tensor_matches_append(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)

        S_append = structures.NGramSequence(10000, n=3)
        for i in range(10):
            S_append.append(hv[i])

        S_from = structures.NGramSequence.from_tensor(hv[:10], n=3)
        assert torch.allclose(S_append.value, S_from.value)
        assert len(S_append) == len(S_from)

    def test_from_tensor_buffer_continues(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)

        S_from = structures.NGramSequence.from_tensor(hv[:5], n=3)
        S_from.append(hv[5])

        S_append = structures.NGramSequence(10000, n=3)
        for i in range(6):
            S_append.append(hv[i])

        assert torch.allclose(S_append.value, S_from.value)

    def test_n1_is_multiset(self):
        generator = torch.Generator()
        generator.manual_seed(seed)
        hv = functional.random(len(letters), 10000, generator=generator)

        S = structures.NGramSequence(10000, n=1)
        for i in range(5):
            S.append(hv[i])

        # For n=1 each window is a single element (no permutation), so the
        # result is equivalent to a multiset.
        expected = functional.multiset(hv[:5])
        assert torch.allclose(S.value, expected)
