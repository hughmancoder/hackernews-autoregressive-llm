from asyncio import base_events
from asyncio import base_events
from posixpath import split
import unittest
import torch

from mainrun.data import train_tokenizer, BPETokenizer, get_batch, iter_full_split
from mainrun.model_llama import GPTConfig, GPT

class TestMainFunctionality(unittest.TestCase):

    def test_tokenizer(self):
        titles = ["hello world", "hacker news is great", "testing tokenizer"]
        vocab_size = 50
        tk = train_tokenizer(titles, vocab_size)
        tokenizer = BPETokenizer(tk)
        
        self.assertTrue(tokenizer.vocab_size > 0)
        
        text = "This is a tokenizer to test"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"\nEncoded:  {encoded}")
        print(f"Decoded: {decoded}")
        
        self.assertIsInstance(encoded, list)
        self.assertTrue(len(encoded) > 0)
        self.assertIsInstance(decoded, str)

    def test_get_batch(self):
        split_ids = torch.arange(100)
        ptr = 0
        block_size = 8
        batch_size = 4
        
        x, y, new_ptr = get_batch(split_ids, ptr, block_size, batch_size, torch.device("cpu"))
        
        self.assertEqual(x.shape, (batch_size, block_size))
        self.assertEqual(y.shape, (batch_size, block_size))
        self.assertEqual(new_ptr, ptr + block_size * batch_size)

    def test_iter_full_split(self):
        block_size = 4
        batch_size = 2
        split_ids = torch.arange(20)
        
        
        batches = list(iter_full_split(split_ids, block_size, batch_size, torch.device("cpu")))
        
        
        self.assertEqual(len(batches), 2)
        x0, y0 = batches[0]

        print(f"\n--- Batch 0 ---")
        print(f"x0:\n{x0}")
        print(f"y0:\n{y0}")
        print(f"\nTotal batches: {len(batches)}")

        self.assertEqual(x0.shape, (batch_size, block_size))
        self.assertEqual(y0.shape, (batch_size, block_size))

    def test_gpt_forward(self):
        cfg = GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=2,
            d_model=32,
            dropout=0.0
        )
        model = GPT(cfg)
        
        batch_size = 2
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
        y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size))
        
        logits, loss = model(x, y)
        
        self.assertEqual(logits.shape, (batch_size, cfg.block_size, cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)


if __name__ == '__main__':
    unittest.main()
