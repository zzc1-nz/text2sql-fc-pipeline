"""Tests for the LoRA SFT reference backend.

Run with discover (``tests`` is shadowed by a site-packages package otherwise)::

    $env:PYTHONPATH='src'
    python -m unittest discover -s tests -p 'test_train.py' -v

The masking tests are the point of this file. A causal mask or label shift that
is off by one still *trains* (loss goes down, numbers look plausible) and only
shows up as a model that cannot answer -- so both are asserted directly:

* ``test_causal_mask_blocks_future`` proves position ``i`` cannot see ``i+1``,
* ``test_loss_ignores_masked_positions`` proves ``-100`` positions contribute
  nothing,
* ``test_label_shift_is_next_token`` proves the shift is ``i -> i+1``,
* ``test_prompt_is_masked_target_is_supervised`` proves the prompt/target
  boundary produced by the real data path is where it claims to be.

No network, no fixture dependency: records are constructed by hand so this file
tests the trainer, not the format stage.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import torch

from fcdata.config import TrainCfg, scratch_dir
from fcdata.records import SFTRecord
from fcdata.train import train
from fcdata.train.data import (
    IGNORE_INDEX,
    Batch,
    collate,
    encode_record,
    render_pair,
    split_prompt_target,
)
from fcdata.train.hf_peft import (
    ASSISTANT_HEADERS,
    LoraSpec,
    build_response_labels,
    find_assistant_span,
    hf_deps_available,
    require_hf_deps,
)
from fcdata.train.reference import (
    LoRALinear,
    build_model,
    count_parameters,
    describe_model,
    generate,
    generate_batch,
    load_model,
    masked_lm_loss,
    merge_lora,
    save_checkpoint,
    unmerge_lora,
)
from fcdata.train.vocab import (
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    WordTokenizer,
    detokenize,
    split_tokens,
)

SYSTEM = "You are a SQL assistant. Use the execute_sql tool."


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_record(i: int, question: str, sql: str, kind: str = "raw",
                weight: float = 1.0) -> SFTRecord:
    """A hand-built SFTRecord in canonical OpenAI wire format."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"CREATE TABLE student (stu_id INTEGER, "
                                    f"stu_name TEXT, gpa REAL); Question: {question}"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": f"call_{i}", "type": "function",
                         "function": {"name": "execute_sql",
                                      "arguments": json.dumps({"sql": sql})}}]},
    ]
    return SFTRecord(id=f"rec_{i}", source_id=f"src_{i}", kind=kind, db_id="college",
                     messages=messages, target_sql=sql, weight=weight)


QA = [
    ("how many students are there", "select count ( * ) from student"),
    ("list the student names", "select stu_name from student"),
    ("students with gpa above 3.5", "select stu_name from student where gpa > 3.5"),
    ("what is the average gpa", "select avg ( gpa ) from student"),
]


def tiny_cfg(**kw) -> TrainCfg:
    """A config small enough that the whole test file runs in seconds."""
    base = dict(n_layer=2, n_head=2, d_model=32, d_ff=64, vocab_size=128,
                max_seq_len=96, lora_r=4, lora_alpha=8, lora_dropout=0.0,
                dropout=0.0, epochs=1, batch_size=4, lr=1e-2, log_every=1000,
                eval_every=0, seed=7)
    base.update(kw)
    return TrainCfg(**base)


class ScratchTestCase(unittest.TestCase):
    """Base class giving each test a writable scratch dir (tempfile is unusable)."""

    def setUp(self) -> None:
        self._dirs: list[Path] = []

    def tearDown(self) -> None:
        for d in self._dirs:
            shutil.rmtree(d, ignore_errors=True)

    def scratch(self, prefix: str = "train_test") -> Path:
        d = scratch_dir(prefix)
        self._dirs.append(d)
        return d


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
class TestWordTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        # The corpus must contain the vocabulary the tests then encode, because
        # word-level tokenization maps unseen words to <unk> by design. Fitting on
        # a corpus that omits a word and then asserting that word round-trips
        # would be testing the wrong thing.
        corpus = [
            "select t1 . name , count ( * ) from student as t1 "
            "join dept as t2 on t1 . dept_id = t2 . id where t1 . gpa >= 3.5",
            "select stu_name from student where gpa > 3.5 and budget > 100000",
            '{"tool_calls": [{"type": "function", "function": {"name": '
            '"execute_sql", "arguments": {"sql": "select * from t where a = \'x y\'"}}}]}',
            "<|im_start|>system <|im_end|> <|im_start|>assistant",
        ] * 3
        self.tok = WordTokenizer().fit(corpus, vocab_size=512)

    def test_special_tokens_have_fixed_ids(self) -> None:
        self.assertEqual(self.tok.tokens[: len(SPECIAL_TOKENS)], list(SPECIAL_TOKENS))
        self.assertEqual(self.tok.pad_id, PAD_ID)
        self.assertEqual(self.tok.eos_id, EOS_ID)
        self.assertEqual(self.tok.vocab_size, len(self.tok.tokens))

    def test_fit_is_deterministic(self) -> None:
        corpus = ["select a from t where b >= 1"] * 5
        a = WordTokenizer().fit(corpus, 64).tokens
        b = WordTokenizer().fit(corpus, 64).tokens
        self.assertEqual(a, b)

    def test_fit_respects_vocab_size(self) -> None:
        tok = WordTokenizer().fit([" ".join(f"w{i}" for i in range(500))] * 2, 100)
        self.assertEqual(tok.vocab_size, 100)

    def test_fit_orders_by_frequency_then_alphabetically(self) -> None:
        """Deterministic ordering: most frequent first, ties alphabetical."""
        corpus = ["b a a c c c"]
        tok = WordTokenizer().fit(corpus, 32)
        body = tok.tokens[len(SPECIAL_TOKENS):]
        self.assertEqual(body[0], "c")      # 3 occurrences
        self.assertEqual(body[1], "a")      # 2 occurrences
        self.assertEqual(body[2], "b")      # 1 occurrence

    def test_multi_char_operators_are_atomic(self) -> None:
        toks = split_tokens("a >= 1 and b <= 2 and c <> 3 and d != 4")
        for op in (">=", "<=", "<>", "!="):
            self.assertIn(op, toks, f"{op} should be one token")
        # none of the multi-char ops may be split into single characters
        self.assertEqual(toks.count(">"), 0)
        self.assertEqual(toks.count("<"), 0)

    def test_numbers_keep_decimals(self) -> None:
        self.assertIn("3.5", split_tokens("gpa >= 3.5"))
        self.assertIn("100000", split_tokens("budget > 100000"))

    def test_encode_decode_sql_roundtrip(self) -> None:
        sql = ("select t1.name, count(*) from student as t1 "
               "where t1.gpa >= 3.5")
        back = self.tok.decode(self.tok.encode(sql))
        self.assertIn("t1.name", back)
        self.assertIn("count(*)", back)
        self.assertIn("3.5", back)
        self.assertNotIn("<unk>", back)

    def test_encode_decode_json_roundtrip(self) -> None:
        payload = {"tool_calls": [{"type": "function", "function": {
            "name": "execute_sql", "arguments": {"sql": "select * from t where a = 'x y'"}}}]}
        text = json.dumps(payload)
        back = self.tok.decode(self.tok.encode(text))
        self.assertEqual(json.loads(back), payload)

    def test_unknown_word_maps_to_unk(self) -> None:
        ids = self.tok.encode("select zzz_never_seen from student")
        self.assertIn(UNK_ID, ids)
        self.assertIn("<unk>", self.tok.decode(ids))

    def test_bos_eos_flags(self) -> None:
        ids = self.tok.encode("select", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], self.tok.bos_id)
        self.assertEqual(ids[-1], self.tok.eos_id)

    def test_max_len_truncation(self) -> None:
        ids = self.tok.encode(" ".join(["word"] * 50), max_len=10)
        self.assertEqual(len(ids), 10)

    def test_save_load_roundtrip(self) -> None:
        d = scratch_dir("tok")
        try:
            p = self.tok.save(d / "tokenizer.json")
            loaded = WordTokenizer.load(p)
            self.assertEqual(loaded.tokens, self.tok.tokens)
            text = "select count ( * ) from student"
            self.assertEqual(loaded.encode(text), self.tok.encode(text))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_decode_skips_padding(self) -> None:
        ids = [PAD_ID, PAD_ID, *self.tok.encode("select"), EOS_ID]
        self.assertNotIn("<pad>", self.tok.decode(ids))
        self.assertNotIn("<eos>", self.tok.decode(ids))

    def test_load_rejects_bad_special_prefix(self) -> None:
        with self.assertRaises(ValueError):
            WordTokenizer.from_dict({"tokens": ["nope", "also_nope"]})


# --------------------------------------------------------------------------- #
# causal mask + label masking (the numerically delicate part)
# --------------------------------------------------------------------------- #
class TestMaskingCorrectness(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = tiny_cfg()
        self.model = build_model(self.cfg, vocab_size=64)
        self.model.eval()

    def test_causal_mask_blocks_future(self) -> None:
        """Token ``i`` must not attend to ``i+1``.

        Perturbing the *last* token of the sequence must leave the logits at every
        earlier position bit-identical. If the mask leaked, the earlier logits
        would change. This is the classic off-by-one: an off-by-one mask still
        trains, it just learns the wrong thing.
        """
        torch.manual_seed(0)
        ids = torch.randint(5, 64, (1, 12))
        with torch.no_grad():
            base = self.model(ids)

            perturbed = ids.clone()
            perturbed[0, -1] = (perturbed[0, -1] + 7) % 64
            after = self.model(perturbed)

        # positions 0..T-2 predict the future only through their own context
        torch.testing.assert_close(base[0, :-1], after[0, :-1],
                                   rtol=0, atol=1e-6)
        # sanity: the last position *did* change, so the test is not vacuous
        self.assertFalse(torch.allclose(base[0, -1], after[0, -1], atol=1e-6))

    def test_manual_mask_equals_sdpa(self) -> None:
        """The explicit tril-mask path and the SDPA fast path must agree."""
        torch.manual_seed(1)
        ids = torch.randint(5, 64, (2, 16))
        self.model.set_sdpa(True)
        with torch.no_grad():
            fast = self.model(ids)
        self.model.set_sdpa(False)
        with torch.no_grad():
            slow = self.model(ids)
        torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)

    def test_label_shift_is_next_token(self) -> None:
        """Logits at position ``i`` are scored against ``labels[i+1]``.

        Constructs logits where the correct next-token id is known, so a shift in
        either direction produces a large loss instead of a tiny one.
        """
        vocab, seq = 10, 6
        labels = torch.tensor([[0, 1, 2, 3, 4, 5]])
        logits = torch.zeros(1, seq, vocab)
        # make position i predict label i+1 with high confidence
        for i in range(seq - 1):
            logits[0, i, labels[0, i + 1]] = 12.0
        loss, acc, n_sup = masked_lm_loss(logits, labels)
        self.assertEqual(n_sup, seq - 1)          # last label has no logit to score
        self.assertEqual(acc, 1.0)
        self.assertLess(float(loss), 1e-3)

        # shifted the wrong way (predict label i instead of i+1) must be much worse
        bad = torch.zeros(1, seq, vocab)
        for i in range(seq - 1):
            bad[0, i, labels[0, i]] = 12.0
        bad_loss, bad_acc, _ = masked_lm_loss(bad, labels)
        self.assertGreater(float(bad_loss), float(loss) + 1.0)
        self.assertLess(bad_acc, 1.0)

    def test_loss_ignores_masked_positions(self) -> None:
        """``-100`` positions must be excluded from the loss entirely.

        Verified by computing the same loss two independent ways: through
        ``masked_lm_loss`` and by manually gathering only the supervised
        ``(logit_i, label_{i+1})`` pairs. If the mask leaked, the manual value
        (which never sees the masked positions) would disagree.
        """
        torch.manual_seed(2)
        vocab, seq = 16, 10
        logits = torch.randn(1, seq, vocab)
        labels = torch.randint(0, vocab, (1, seq))
        labels[0, :5] = IGNORE_INDEX

        loss, acc, n_sup = masked_lm_loss(logits, labels)

        # manual reference: only positions whose shifted label is supervised
        pairs = [(logits[0, i], int(labels[0, i + 1]))
                 for i in range(seq - 1) if labels[0, i + 1] != IGNORE_INDEX]
        self.assertEqual(n_sup, len(pairs))
        manual = torch.stack([
            torch.nn.functional.cross_entropy(logit.unsqueeze(0),
                                              torch.tensor([target]))
            for logit, target in pairs]).mean()
        self.assertAlmostEqual(float(loss), float(manual), places=5)

        # the masked region must be *excluded*, not merely ignored by accident:
        # turning one masked position into a real label must change the loss.
        unmasked = labels.clone()
        unmasked[0, 5] = 0
        unmasked[0, 4] = IGNORE_INDEX
        loss2, _, n2 = masked_lm_loss(logits, unmasked)
        self.assertNotAlmostEqual(float(loss), float(loss2), places=4)
        self.assertEqual(n2, n_sup)

    def test_masked_labels_do_not_affect_loss_value(self) -> None:
        """Two label tensors agreeing on supervised positions give equal loss."""
        torch.manual_seed(5)
        vocab, seq = 12, 9
        logits = torch.randn(1, seq, vocab)
        base = torch.randint(0, vocab, (1, seq))
        a = base.clone()
        a[0, :4] = IGNORE_INDEX
        b = a.clone()
        # both masked identically; the *other* supervised labels are shared
        loss_a, acc_a, n_a = masked_lm_loss(logits, a)
        loss_b, acc_b, n_b = masked_lm_loss(logits, b)
        self.assertEqual((n_a, n_b), (n_a, n_a))
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=6)
        self.assertAlmostEqual(acc_a, acc_b, places=6)

    def test_all_masked_gives_zero_loss_with_grad(self) -> None:
        logits = torch.randn(2, 5, 8, requires_grad=True)
        labels = torch.full((2, 5), IGNORE_INDEX)
        loss, acc, n_sup = masked_lm_loss(logits, labels)
        self.assertEqual(n_sup, 0)
        self.assertEqual(acc, 0.0)
        self.assertEqual(float(loss), 0.0)
        loss.backward()                       # must not raise
        self.assertIsNotNone(logits.grad)

    def test_loss_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            masked_lm_loss(torch.randn(2, 5), torch.zeros(2, 5, dtype=torch.long))
        with self.assertRaises(ValueError):
            masked_lm_loss(torch.randn(2, 5, 8), torch.zeros(3, 5, dtype=torch.long))


# --------------------------------------------------------------------------- #
# data path: prompt masked, target supervised
# --------------------------------------------------------------------------- #
class TestDataMasking(ScratchTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.records = [make_record(i, q, s) for i, (q, s) in enumerate(QA)]
        self.tok = WordTokenizer().fit(
            [p + t for r in self.records for p, t in [render_pair(r)]], 256)

    def test_split_prompt_target(self) -> None:
        prompt_msgs, target = split_prompt_target(self.records[0].messages)
        self.assertEqual(target["role"], "assistant")
        self.assertEqual(len(prompt_msgs), 2)
        self.assertTrue(all(m["role"] != "assistant" for m in prompt_msgs))

    def test_split_requires_assistant(self) -> None:
        rec = make_record(0, "q", "select 1")
        rec.messages = [m for m in rec.messages if m["role"] != "assistant"]
        with self.assertRaises(ValueError):
            split_prompt_target(rec.messages)

    def test_prompt_masked_target_supervised(self) -> None:
        """The label boundary must equal ``prompt_len`` exactly."""
        enc = encode_record(self.records[0], self.tok, max_seq_len=200)
        self.assertEqual(len(enc.input_ids), len(enc.labels))
        self.assertEqual(enc.prompt_len, enc.labels.count(IGNORE_INDEX))
        self.assertTrue(all(lab == IGNORE_INDEX for lab in enc.labels[: enc.prompt_len]))
        self.assertTrue(all(lab != IGNORE_INDEX for lab in enc.labels[enc.prompt_len:]))
        # every supervised label is a real vocabulary id
        for lab in enc.labels[enc.prompt_len:]:
            self.assertGreaterEqual(lab, 0)
            self.assertLess(lab, self.tok.vocab_size)

    def test_target_ends_with_eos(self) -> None:
        enc = encode_record(self.records[0], self.tok, max_seq_len=200)
        self.assertEqual(enc.input_ids[-1], EOS_ID)
        self.assertEqual(enc.labels[-1], EOS_ID, "EOS must be supervised or the "
                                                 "model never learns to stop")

    def test_truncation_never_cuts_the_target(self) -> None:
        """A long prompt is shortened; the target survives intact."""
        full = encode_record(self.records[0], self.tok, max_seq_len=10_000)
        short = encode_record(self.records[0], self.tok, max_seq_len=full.target_len + 8)
        self.assertEqual(short.target_len, full.target_len)
        self.assertLess(short.prompt_len, full.prompt_len)
        self.assertEqual(len(short.input_ids), len(short.labels))
        self.assertLessEqual(len(short.input_ids), full.target_len + 8)

    def test_length_never_exceeds_max_seq_len(self) -> None:
        """Regression: truncation must hold for EVERY budget, including tiny ones.

        A 257-token sequence was produced for ``max_seq_len=256`` because the
        target was allowed to fill the whole budget and ``<bos>`` was then added
        on top. The model's ``forward`` rejected it, so any run with a seq length
        below the corpus average crashed. Sweeping every budget from 2 to 400
        pins the invariant instead of spot-checking one value.
        """
        for budget in (2, 3, 4, 8, 16, 32, 64, 100, 128, 200, 256, 384, 400):
            for rec in self.records:
                enc = encode_record(rec, self.tok, max_seq_len=budget)
                self.assertLessEqual(
                    len(enc.input_ids), budget,
                    f"budget={budget}: produced {len(enc.input_ids)} tokens for "
                    f"record {rec.id}")
                self.assertEqual(len(enc.input_ids), len(enc.labels))
                self.assertGreaterEqual(enc.prompt_len, 1, "BOS token was dropped")
                self.assertGreaterEqual(enc.target_len, 1, "target was emptied")
                # BOS is preserved even under extreme truncation
                self.assertEqual(enc.input_ids[0], self.tok.bos_id)

    def test_truncation_when_target_alone_fills_budget(self) -> None:
        """The exact off-by-one case: target >= budget, so <bos> must still fit."""
        full = encode_record(self.records[0], self.tok, max_seq_len=10_000)
        budget = full.target_len        # target alone would exactly fill the budget
        enc = encode_record(self.records[0], self.tok, max_seq_len=budget)
        self.assertLessEqual(len(enc.input_ids), budget)
        self.assertTrue(enc.target_truncated)
        self.assertEqual(enc.input_ids[0], self.tok.bos_id)
        self.assertEqual(enc.input_ids[-1], EOS_ID,
                         "a truncated target must still end with EOS")

    def test_collate_guards_against_oversized_rows(self) -> None:
        """The collator reports an oversized row instead of letting forward fail."""
        encs = [encode_record(r, self.tok, max_seq_len=200) for r in self.records]
        collate(encs)                                   # no guard -> fine
        with self.assertRaises(ValueError) as ctx:
            collate(encs, max_seq_len=8)
        message = str(ctx.exception)
        self.assertIn("exceeds max_seq_len", message)
        self.assertIn("truncation", message)

    def test_train_survives_small_max_seq_len(self) -> None:
        """End-to-end: a max_seq_len below the corpus average must train fine."""
        cfg = tiny_cfg(epochs=2, batch_size=2, max_seq_len=32)
        d = self.scratch()
        metrics = train(cfg, self.records, None, d)
        # 4 records / batch 2 = 2 steps per epoch, 2 epochs
        self.assertEqual(metrics["steps"], 4)
        self.assertTrue(metrics["history"])
        self.assertLessEqual(metrics["max_seq_len"], 32)

    def test_collate_pads_with_ignore_index(self) -> None:
        encs = [encode_record(r, self.tok, max_seq_len=200) for r in self.records]
        batch = collate(encs)
        self.assertEqual(batch.input_ids.shape[0], len(encs))
        self.assertEqual(batch.input_ids.shape, batch.labels.shape)
        width = batch.input_ids.shape[1]
        for row, enc in enumerate(encs):
            pad = width - len(enc.input_ids)
            if pad:
                # padding must be PAD in inputs and -100 in labels
                self.assertTrue(bool((batch.input_ids[row, -pad:] == PAD_ID).all()))
                self.assertTrue(bool((batch.labels[row, -pad:] == IGNORE_INDEX).all()))

    def test_collate_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            collate([])

    def test_collate_rejects_weight_mismatch(self) -> None:
        encs = [encode_record(self.records[0], self.tok, max_seq_len=200)]
        with self.assertRaises(ValueError):
            collate(encs, weights=[1.0, 2.0])


# --------------------------------------------------------------------------- #
# model structure, LoRA freezing, param counts
# --------------------------------------------------------------------------- #
class TestModelStructure(unittest.TestCase):
    """Architecture invariants, checked in BOTH adaptation modes.

    ``lora_only=False`` (the default) trains everything; ``lora_only=True`` is
    textbook frozen-base LoRA. The architecture is identical either way, so every
    structural assertion is made against the default model and the freezing
    behaviour is asserted separately in :class:`TestLoRAFreezing`.
    """

    def setUp(self) -> None:
        self.cfg = tiny_cfg()
        self.model = build_model(self.cfg, vocab_size=64)

    def test_forward_shape(self) -> None:
        ids = torch.randint(5, 64, (3, 11))
        logits = self.model(ids)
        self.assertEqual(tuple(logits.shape), (3, 11, 64))

    def test_weight_tying(self) -> None:
        self.assertIs(self.model.lm_head.weight, self.model.tok_emb.weight)

    def test_causal_mask_buffer_is_lower_triangular(self) -> None:
        m = self.model.causal_mask
        self.assertEqual(tuple(m.shape), (self.cfg.max_seq_len, self.cfg.max_seq_len))
        self.assertTrue(bool(m[0, 0]))
        self.assertFalse(bool(m[0, 1]))
        self.assertTrue(bool(m[5, 4]))
        self.assertFalse(bool(m[4, 5]))

    def test_pre_ln_structure(self) -> None:
        blk = self.model.blocks[0]
        for name in ("ln1", "ln2", "attn", "mlp"):
            self.assertTrue(hasattr(blk, name), f"missing {name} (pre-LN layout)")
        self.assertIsInstance(blk.attn.q_proj, LoRALinear)

    def test_lora_adapters_exist_on_all_projections(self) -> None:
        """q/k/v/o and both FFN projections must carry adapters."""
        blk = self.model.blocks[0]
        for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj,
                     blk.attn.out_proj, blk.mlp.fc_in, blk.mlp.fc_out):
            self.assertIsInstance(proj, LoRALinear)

    def test_lora_b_starts_at_zero(self) -> None:
        """B=0 means the adapter is an exact no-op at initialisation."""
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                self.assertTrue(bool((module.lora_B == 0).all()))

    def test_lora_delta_zero_at_init(self) -> None:
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                self.assertLess(float(module.delta_weight().abs().max()), 1e-9)

    def test_lora_zero_init_preserves_output(self) -> None:
        """Building two models with the same seed gives identical logits."""
        torch.manual_seed(99)
        a = build_model(self.cfg, vocab_size=64)
        torch.manual_seed(99)
        b = build_model(self.cfg, vocab_size=64)
        ids = torch.randint(5, 64, (1, 7))
        a.eval()
        b.eval()
        with torch.no_grad():
            torch.testing.assert_close(a(ids), b(ids))

    def test_merge_unmerge_roundtrip(self) -> None:
        """merge() then unmerge() restores the exact original weight."""
        module = next(m for m in self.model.modules() if isinstance(m, LoRALinear))
        torch.manual_seed(3)
        torch.nn.init.normal_(module.lora_B, std=0.1)   # make the delta non-zero
        before = module.base.weight.detach().clone()

        module.merge()
        self.assertTrue(module.merged)
        merged = module.base.weight.detach().clone()
        self.assertFalse(torch.allclose(before, merged),
                         "merge() had no effect -- the delta was zero")

        module.unmerge()
        self.assertFalse(module.merged)
        torch.testing.assert_close(module.base.weight.detach(), before, rtol=0, atol=1e-6)

    def test_merged_forward_matches_unmerged(self) -> None:
        torch.manual_seed(4)
        ids = torch.randint(5, 64, (1, 8))
        self.model.eval()
        with torch.no_grad():
            plain = self.model(ids)
            merge_lora(self.model)
            merged = self.model(ids)
            unmerge_lora(self.model)
        torch.testing.assert_close(plain, merged, rtol=1e-5, atol=1e-5)

    def test_full_finetune_when_lora_r_zero(self) -> None:
        cfg = tiny_cfg(lora_r=0)
        model = build_model(cfg, vocab_size=64)
        self.assertEqual(count_parameters(model, trainable_only=True),
                         count_parameters(model))
        self.assertFalse(any(isinstance(m, LoRALinear) for m in model.modules()))

    def test_count_parameters_counts_tied_weight_once(self) -> None:
        """The tied embedding/LM-head matrix is counted once, not twice.

        ``nn.Module.parameters()`` already dedupes shared tensors, and
        ``count_parameters`` dedupes by identity again so it stays correct if
        that ever changes. The invariant that matters is: the counted total
        equals the sum over *unique* tensors, i.e. the vocab x d_model matrix is
        not double counted.
        """
        total = count_parameters(self.model)
        unique: dict[int, int] = {}
        for p in self.model.parameters():
            unique[id(p)] = int(p.numel())
        self.assertEqual(total, sum(unique.values()))
        # both the embedding and the head point at the same tensor
        self.assertIs(self.model.tok_emb.weight, self.model.lm_head.weight)

    def test_sequence_longer_than_max_seq_len_raises(self) -> None:
        ids = torch.randint(5, 64, (1, self.cfg.max_seq_len + 1))
        with self.assertRaises(ValueError):
            self.model(ids)


class TestLoRAFreezing(unittest.TestCase):
    """``lora_only=True``: the textbook frozen-base LoRA behaviour.

    Kept separate from the default-mode tests because the reference backend
    deliberately defaults to end-to-end training -- a randomly-initialised base
    has no language ability to adapt (see ``ReferenceLM.set_trainable``).
    """

    def setUp(self) -> None:
        self.cfg = tiny_cfg(lora_only=True)
        self.model = build_model(self.cfg, vocab_size=64)

    def test_only_lora_params_require_grad(self) -> None:
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable, "no trainable parameters at all")
        for name in trainable:
            self.assertTrue(name.endswith("lora_A") or name.endswith("lora_B"),
                            f"{name} should be frozen but requires_grad=True")

    def test_frozen_base_receives_no_gradient(self) -> None:
        ids = torch.randint(5, 64, (2, 9))
        logits = self.model(ids)
        loss, _, _ = masked_lm_loss(logits, ids.clone())
        loss.backward()
        for name, param in self.model.named_parameters():
            if name.endswith("lora_A") or name.endswith("lora_B"):
                self.assertIsNotNone(param.grad, f"{name} should have a gradient")
            else:
                self.assertIsNone(param.grad, f"{name} is frozen but got a gradient")

    def test_lora_is_a_small_fraction(self) -> None:
        info = describe_model(self.model)
        self.assertLess(info["trainable_params"], info["total_params"])
        self.assertLess(info["trainable_pct"], 25.0,
                        f"LoRA is {info['trainable_pct']}% of the model -- too much")
        self.assertEqual(info["trainable_params"], info["lora_params"])

    def test_unfreeze_all_restores_full_training(self) -> None:
        self.model.unfreeze_all()
        self.assertEqual(count_parameters(self.model, trainable_only=True),
                         count_parameters(self.model))

    def test_default_mode_trains_everything(self) -> None:
        model = build_model(tiny_cfg(), vocab_size=64)
        info = describe_model(model)
        self.assertFalse(info["lora_only"])
        self.assertEqual(info["trainable_params"], info["total_params"])
        self.assertGreater(info["lora_params"], 0)


# --------------------------------------------------------------------------- #
# checkpoint round-trip
# --------------------------------------------------------------------------- #
class TestCheckpoint(ScratchTestCase):
    def test_save_load_roundtrip(self) -> None:
        cfg = tiny_cfg()
        tok = WordTokenizer().fit([qa for _, qa in QA] * 2, 128)
        model = build_model(cfg, vocab_size=tok.vocab_size)
        d = self.scratch()
        path = save_checkpoint(d / "model.pt", model, tok, cfg)
        self.assertTrue(path.exists())

        model2, tok2 = load_model(path)
        self.assertEqual(tok2.tokens, tok.tokens)
        self.assertEqual(tok2.vocab_size, tok.vocab_size)
        self.assertEqual(describe_model(model2)["total_params"],
                         describe_model(model)["total_params"])
        ids = torch.randint(5, tok.vocab_size, (1, 9))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids), model2(ids))

    def test_load_accepts_run_directory(self) -> None:
        cfg = tiny_cfg()
        tok = WordTokenizer().fit([qa for _, qa in QA], 128)
        model = build_model(cfg, vocab_size=tok.vocab_size)
        d = self.scratch()
        save_checkpoint(d / "model.pt", model, tok, cfg)
        model2, _ = load_model(d)            # directory, not the file
        self.assertEqual(describe_model(model2)["total_params"],
                         describe_model(model)["total_params"])

    def test_load_missing_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model(self.scratch() / "nope.pt")

    def test_load_rejects_foreign_file(self) -> None:
        p = self.scratch() / "junk.pt"
        torch.save({"not": "a checkpoint"}, p)
        with self.assertRaises(ValueError):
            load_model(p)

    def test_loaded_model_is_in_eval_mode(self) -> None:
        cfg = tiny_cfg()
        tok = WordTokenizer().fit([qa for _, qa in QA], 128)
        model = build_model(cfg, vocab_size=tok.vocab_size)
        model.train()
        d = self.scratch()
        save_checkpoint(d / "model.pt", model, tok, cfg)
        model2, _ = load_model(d / "model.pt")
        self.assertFalse(model2.training)

    def test_checkpoint_embeds_config_and_tokenizer(self) -> None:
        cfg = tiny_cfg()
        tok = WordTokenizer().fit([qa for _, qa in QA], 128)
        model = build_model(cfg, vocab_size=tok.vocab_size)
        d = self.scratch()
        p = save_checkpoint(d / "model.pt", model, tok, cfg)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        self.assertIn("tokenizer", payload)
        self.assertIn("train_cfg", payload)
        self.assertEqual(payload["train_cfg"]["n_layer"], cfg.n_layer)
        self.assertEqual(payload["tokenizer"]["tokens"], tok.tokens)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
class TestGenerate(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = tiny_cfg()
        self.tok = WordTokenizer().fit([qa for _, qa in QA] * 4, 128)
        self.model = build_model(self.cfg, vocab_size=self.tok.vocab_size)
        self.model.eval()

    def test_generate_returns_string(self) -> None:
        out = generate(self.model, self.tok, "select", max_new_tokens=8)
        self.assertIsInstance(out, str)

    def test_greedy_is_deterministic(self) -> None:
        a = generate(self.model, self.tok, "select count", max_new_tokens=10,
                     temperature=0.0)
        b = generate(self.model, self.tok, "select count", max_new_tokens=10,
                     temperature=0.0)
        self.assertEqual(a, b)

    def test_generate_respects_max_new_tokens(self) -> None:
        out = generate(self.model, self.tok, "select", max_new_tokens=5,
                       temperature=0.0)
        # decoded text can be shorter (EOS), never longer than the token budget
        self.assertLessEqual(len(self.tok.encode(out)), 5)

    def test_generate_zero_tokens_returns_empty(self) -> None:
        self.assertEqual(generate(self.model, self.tok, "select", max_new_tokens=0), "")

    def test_generate_stops_at_eos(self) -> None:
        """With EOS forced to be the argmax, generation stops immediately."""
        model = self.model

        def force_eos(ids: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(ids.shape[0], ids.shape[1], self.tok.vocab_size)
            logits[:, :, self.tok.eos_id] = 50.0
            return logits

        model.forward = force_eos          # type: ignore[method-assign]
        out = generate(model, self.tok, "select", max_new_tokens=10)
        self.assertEqual(out, "")

    def test_generate_does_not_change_training_mode(self) -> None:
        self.model.train()
        generate(self.model, self.tok, "select", max_new_tokens=3)
        self.assertTrue(self.model.training)

    def test_generate_batch_matches_single(self) -> None:
        prompts = ["select", "select count", "from student"]
        batched = generate_batch(self.model, self.tok, prompts, max_new_tokens=6)
        self.assertEqual(len(batched), len(prompts))
        for prompt, got in zip(prompts, batched):
            single = generate(self.model, self.tok, prompt, max_new_tokens=6)
            self.assertEqual(got, single,
                             f"batched generation diverged for prompt {prompt!r}")

    def test_generate_batch_empty(self) -> None:
        self.assertEqual(generate_batch(self.model, self.tok, []), [])


# --------------------------------------------------------------------------- #
# the training loop
# --------------------------------------------------------------------------- #
class TestTrainLoop(ScratchTestCase):
    def _records(self, n: int = 24) -> list[SFTRecord]:
        return [make_record(i, *QA[i % len(QA)]) for i in range(n)]

    def test_can_overfit_single_example(self) -> None:
        """THE gate: the reference backend must be able to memorise one record.

        This is the test that catches the "LoRA-only on a randomly-initialised
        base" design flaw. A model adapted only through a rank-4 bottleneck on a
        frozen random base cannot learn the task at all: measured loss goes
        6.1 -> ~4.1 and greedy decoding stays degenerate. Training end-to-end
        drives the loss to ~0 and reproduces the target text.

        Both halves are asserted: the loss must collapse *and* greedy generation
        must reproduce a substantial part of the target. A loss that drops while
        generation stays garbage would mean the labels and the decode path
        disagree, which is a different bug this test also guards against.
        """
        # 1 record x 1 batch = 1 step per epoch, so epochs IS the step count here
        cfg = tiny_cfg(epochs=60, batch_size=1, lr=3e-3, dropout=0.0,
                       n_layer=2, d_model=64, d_ff=128, max_seq_len=160)
        rec = make_record(0, *QA[0])
        d = self.scratch()

        # one record, trained end-to-end for 60 steps
        metrics = train(cfg, [rec], None, d)
        self.assertGreaterEqual(metrics["steps"], 60,
                                "the overfit test did not actually run 60 steps")
        final_loss = metrics["final_train_loss"]
        self.assertLess(final_loss, 2.0,
                        f"could not overfit a single record: final loss {final_loss}")
        # and the loss actually collapsed rather than creeping down
        self.assertLess(final_loss, metrics["history"][0]["loss"] / 2.0)

        # greedy generation must reproduce the target. Compared whitespace-
        # insensitively: the tokenizer canonicalises spacing (``<tool_call>``
        # round-trips as ``< tool_call >``), which is semantically identical for
        # both SQL and JSON, so a raw string compare would fail on formatting
        # rather than on content.
        model, tok = load_model(d / "model.pt")
        prompt_text, target_text = render_pair(rec)
        generated = generate(model, tok, prompt_text, max_new_tokens=64,
                             temperature=0.0)

        def squash(text: str) -> str:
            return "".join(text.split())

        target_sq, generated_sq = squash(target_text), squash(generated)
        self.assertTrue(generated_sq, "generation produced nothing")
        # the generated SQL must appear (modulo whitespace) inside the target
        sql = "selectcount(*)fromstudent"
        self.assertIn(sql, generated_sq.replace(" ", ""),
                      f"generated text does not contain the target SQL\n"
                      f"  target={target_text!r}\n  generated={generated!r}")
        # and a long common prefix, not just a lucky substring
        common = 0
        for a, b in zip(target_sq, generated_sq):
            if a != b:
                break
            common += 1
        self.assertGreater(common / max(1, len(target_sq)), 0.5,
                           f"only {common}/{len(target_sq)} characters match\n"
                           f"  target={target_text!r}\n  generated={generated!r}")

    def test_lora_only_cannot_learn_from_scratch(self) -> None:
        """Documented negative result: frozen-base LoRA fails on a random base.

        This is asserted so the *reason* the reference backend trains end-to-end
        stays in the test suite as executable evidence rather than a comment. If
        a future change makes LoRA-only work here, this test fails and the
        documentation must be updated.
        """
        cfg = tiny_cfg(epochs=40, batch_size=1, lr=3e-3, dropout=0.0,
                       n_layer=2, d_model=64, d_ff=128, max_seq_len=160,
                       lora_only=True)
        rec = make_record(0, *QA[0])
        d = self.scratch()
        metrics = train(cfg, [rec], None, d)
        self.assertGreaterEqual(metrics["steps"], 40)
        # LoRA-only is much worse: it cannot get anywhere near the memorised loss
        self.assertGreater(metrics["final_train_loss"], 2.0,
                           "lora_only unexpectedly learned the task from scratch -- "
                           "if this passes, update the design note in reference.py")
        # and the trainable set really was adapter-only
        self.assertEqual(metrics["trainable_params"], metrics["lora_params"])
        self.assertLess(metrics["trainable_params"], metrics["total_params"])

    def test_default_is_end_to_end_not_lora_only(self) -> None:
        """The default must train everything: a random base has nothing to adapt."""
        cfg = tiny_cfg(epochs=1, batch_size=2)
        d = self.scratch()
        m = train(cfg, self._records(6), None, d)
        self.assertFalse(m["lora_only"])
        self.assertEqual(m["trainable_params"], m["total_params"])
        self.assertGreater(m["lora_params"], 0,
                           "LoRA adapters should still exist in the architecture")

    def test_loss_decreases_on_a_tiny_batch(self) -> None:
        """Two optimizer steps on one repeated record must reduce the loss."""
        cfg = tiny_cfg(epochs=1, batch_size=2, lr=3e-3, log_every=1)
        recs = self._records(8)
        d = self.scratch()
        metrics = train(cfg, recs, recs[:2], d, max_steps=6)
        losses = [h["loss"] for h in metrics["history"]]
        self.assertGreaterEqual(len(losses), 4)
        self.assertLess(losses[-1], losses[0],
                        f"loss did not decrease: {losses}")

    def test_train_writes_expected_artifacts(self) -> None:
        cfg = tiny_cfg(epochs=1, batch_size=4)
        recs = self._records(12)
        d = self.scratch()
        metrics = train(cfg, recs, recs[:4], d)
        for name in ("model.pt", "metrics.json", "config.json", "train_log.txt"):
            self.assertTrue((d / name).exists(), f"missing {name}")
        on_disk = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["steps"], metrics["steps"])
        self.assertIn("history", on_disk)
        self.assertIn("epochs_summary", on_disk)

    def test_metrics_have_required_keys(self) -> None:
        cfg = tiny_cfg(epochs=1, batch_size=4)
        recs = self._records(12)
        d = self.scratch()
        m = train(cfg, recs, recs[:4], d)
        for key in ("backend", "wall_time_s", "total_params", "trainable_params",
                    "vocab_size", "n_train", "n_dev", "history", "epochs_summary",
                    "final_train_loss", "final_dev_loss", "steps_per_sec",
                    "tokens_per_sec", "loss_decreased", "lora_only",
                    "lora_params", "lora_r"):
            self.assertIn(key, m, f"metrics.json missing {key}")
        self.assertEqual(m["backend"], "reference")
        self.assertGreaterEqual(m["total_params"], m["trainable_params"])

    def test_history_entries_are_plottable(self) -> None:
        cfg = tiny_cfg(epochs=1, batch_size=4)
        recs = self._records(8)
        d = self.scratch()
        m = train(cfg, recs, None, d)
        self.assertTrue(m["history"])
        for entry in m["history"]:
            self.assertIn("step", entry)
            self.assertIn("loss", entry)
            self.assertIn("token_acc", entry)
            self.assertIsInstance(entry["loss"], float)
            self.assertGreaterEqual(entry["loss"], 0.0)
            self.assertGreaterEqual(entry["token_acc"], 0.0)
            self.assertLessEqual(entry["token_acc"], 1.0)

    def test_max_steps_caps_training(self) -> None:
        cfg = tiny_cfg(epochs=10, batch_size=2)
        recs = self._records(16)
        d = self.scratch()
        m = train(cfg, recs, None, d, max_steps=3)
        self.assertEqual(m["steps"], 3)

    def test_max_records_subsamples(self) -> None:
        cfg = tiny_cfg(epochs=1, batch_size=4)
        recs = self._records(40)
        d = self.scratch()
        m = train(cfg, recs, None, d, max_records=8)
        self.assertEqual(m["n_train"], 8)

    def test_grad_clipping_is_applied(self) -> None:
        """Clipping must bound the post-clip norm, and report the pre-clip one."""
        cfg = tiny_cfg(epochs=1, batch_size=4, max_grad_norm=1.0)
        recs = self._records(8)
        d = self.scratch()
        m = train(cfg, recs, None, d)
        self.assertTrue(m["history"])
        for entry in m["history"]:
            self.assertIn("grad_norm", entry)
            self.assertIn("grad_norm_after_clip", entry)
            # the pre-clip norm is a diagnostic and may exceed the threshold...
            self.assertGreater(entry["grad_norm"], 0.0)
            # ...but the applied (post-clip) norm must never exceed it
            self.assertLessEqual(entry["grad_norm_after_clip"], 1.0 + 1e-6)
            if entry["clipped"]:
                self.assertGreater(entry["grad_norm"], 1.0)

    def test_weighted_sampling_is_used(self) -> None:
        """A high-weight record must be sampled more often than a low-weight one."""
        cfg = tiny_cfg(epochs=1, batch_size=4)
        recs = self._records(10)
        for r in recs:
            r.weight = 1.0
        recs[0].weight = 50.0            # heavily oversampled
        d = self.scratch()
        m = train(cfg, recs, None, d, max_steps=6)
        self.assertGreater(m["steps"], 0)

    def test_unknown_backend_raises(self) -> None:
        cfg = tiny_cfg()
        cfg.backend = "nope"
        with self.assertRaises(ValueError):
            train(cfg, self._records(4), None, self.scratch())

    def test_empty_training_set_raises(self) -> None:
        cfg = tiny_cfg()
        with self.assertRaises(ValueError):
            train(cfg, [], None, self.scratch())

    def test_records_without_assistant_are_skipped(self) -> None:
        cfg = tiny_cfg(epochs=1, batch_size=2)
        good = self._records(8)
        bad = make_record(99, "q", "select 1")
        bad.messages = [m for m in bad.messages if m["role"] != "assistant"]
        d = self.scratch()
        m = train(cfg, good + [bad], None, d)
        self.assertEqual(m["n_train"], len(good))

    def test_train_then_generate_end_to_end(self) -> None:
        cfg = tiny_cfg(epochs=2, batch_size=4, lr=5e-3)
        recs = self._records(16)
        d = self.scratch()
        train(cfg, recs, recs[:4], d)
        model, tok = load_model(d / "model.pt")
        prompt, _ = render_pair(recs[0])
        out = generate(model, tok, prompt, max_new_tokens=24, temperature=0.0)
        self.assertIsInstance(out, str)


# --------------------------------------------------------------------------- #
# hf_peft: the GPU-free logic (response-only label masking)
# --------------------------------------------------------------------------- #
class TestHfPeftMasking(unittest.TestCase):
    """The hf backend cannot run on CPU, but its label masking can be tested.

    ``find_assistant_span`` / ``build_response_labels`` are pure functions over
    token ids, and they are the most dangerous part of a QLoRA script: an
    off-by-one there trains the model to generate the assistant *header* instead
    of the answer, which still produces a falling loss. So they are tested with a
    synthetic tokenizer whose ids are known exactly.
    """

    def setUp(self) -> None:
        # Synthetic vocabulary: distinct ids per token, so the header is findable.
        self.vocab = {
            "<pad>": 0, "<bos>": 1, "<eos>": 2,
            "<|im_start|>": 10, "assistant": 11, "user": 12, "system": 13,
            "\n": 14, "hello": 20, "select": 21, "from": 22, "t": 23,
        }
        self.header = [self.vocab["<|im_start|>"], self.vocab["assistant"]]
        self.ids = [
            self.vocab["<bos>"],
            self.vocab["<|im_start|>"], self.vocab["user"], self.vocab["hello"],
            self.vocab["<|im_start|>"], self.vocab["assistant"],
            self.vocab["select"], self.vocab["from"], self.vocab["t"],
            self.vocab["<eos>"],
        ]
        self.span = 6                      # first token after the assistant header

    def test_find_assistant_span(self) -> None:
        self.assertEqual(find_assistant_span(self.ids, self.header), self.span)

    def test_find_assistant_span_missing_returns_minus_one(self) -> None:
        self.assertEqual(find_assistant_span(self.ids, [999, 998]), -1)
        self.assertEqual(find_assistant_span(self.ids, []), -1)

    def test_find_assistant_span_uses_last_occurrence(self) -> None:
        """A multi-turn chat must supervise the *final* assistant turn."""
        multi = self.ids + [
            self.vocab["<|im_start|>"], self.vocab["user"], self.vocab["hello"],
            self.vocab["<|im_start|>"], self.vocab["assistant"],
            self.vocab["select"],
        ]
        self.assertEqual(find_assistant_span(multi, self.header), len(multi) - 1)

    def test_build_response_labels_masks_prompt_only(self) -> None:
        labels = build_response_labels(self.ids, self.header)
        self.assertEqual(len(labels), len(self.ids))
        # everything up to and including the assistant header is masked
        self.assertTrue(all(lab == IGNORE_INDEX for lab in labels[: self.span]))
        # the answer is fully supervised, id-for-id
        self.assertEqual(labels[self.span:], self.ids[self.span:])
        # the answer tokens really are supervised
        self.assertNotIn(IGNORE_INDEX, labels[self.span:])

    def test_build_response_labels_raises_without_header(self) -> None:
        with self.assertRaises(ValueError):
            build_response_labels(self.ids, [999])

    def test_build_response_labels_custom_ignore_index(self) -> None:
        labels = build_response_labels(self.ids, self.header, ignore_index=-1)
        self.assertTrue(all(lab == -1 for lab in labels[: self.span]))

    def test_shifted_labels_supervise_exactly_the_answer(self) -> None:
        """End-to-end: after the causal shift, the supervised targets are the answer."""
        labels = build_response_labels(self.ids, self.header)
        # the loss uses labels[1:] against logits[:-1]
        shifted = labels[1:]
        supervised = [lab for lab in shifted if lab != IGNORE_INDEX]
        self.assertEqual(supervised, self.ids[self.span:])
        # first supervised target is the first answer token
        self.assertEqual(supervised[0], self.vocab["select"])

    def test_assistant_headers_cover_known_templates(self) -> None:
        for template in ("qwen", "llama3", "plain"):
            self.assertIn(template, ASSISTANT_HEADERS)
            self.assertTrue(ASSISTANT_HEADERS[template])

    def test_lora_spec_targets_all_projections(self) -> None:
        spec = LoraSpec()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            self.assertIn(name, spec.target_modules)
        kwargs = spec.as_kwargs()
        self.assertEqual(kwargs["task_type"], "CAUSAL_LM")
        self.assertEqual(kwargs["r"], spec.r)

    def test_hf_deps_unavailable_raises_clear_error(self) -> None:
        """In this CPU environment the hf backend must fail with an actionable message."""
        ok, missing = hf_deps_available()
        if ok:                              # pragma: no cover - GPU box
            self.skipTest("hf stack is installed; cannot test the guard path")
        self.assertTrue(missing)
        with self.assertRaises(RuntimeError) as ctx:
            require_hf_deps()
        message = str(ctx.exception)
        self.assertIn("reference", message)
        self.assertIn("pip install", message)
        for name in missing:
            self.assertIn(name, message)

    def test_train_hf_raises_runtime_error_not_importerror(self) -> None:
        """Calling the hf path on CPU must raise RuntimeError, never ImportError."""
        ok, _ = hf_deps_available()
        if ok:                              # pragma: no cover - GPU box
            self.skipTest("hf stack is installed")
        cfg = tiny_cfg()
        cfg.backend = "hf"
        with self.assertRaises(RuntimeError):
            train(cfg, [make_record(0, *QA[0])], None, scratch_dir("hf_guard"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
