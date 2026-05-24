#!/usr/bin/env python3
"""lm_eval CLI wrapper with PEFT monkey-patch for ExaoneModel and similar
custom architectures whose `get_input_embeddings` is not auto-handled.

Usage: python lm_eval_runner.py <lm_eval CLI args...>
  (즉 lm_eval CLI를 그대로 forward — 인자만 동일하게 전달)
"""
import sys


def _patch_peft_tied_check():
    """PEFT의 _get_module_names_tied_with_embedding이 NotImplementedError를 raise하면
    빈 set으로 fallback하도록 wrapping. tied embedding 검사 실패 = 안전한 가정 (no tied).
    """
    try:
        from peft.utils import other as peft_other
        orig = peft_other._get_module_names_tied_with_embedding

        def safe(model):
            try:
                return orig(model)
            except NotImplementedError as e:
                print(f"[patch] tied-modules check skipped (NotImplementedError: {e})", file=sys.stderr)
                return set()

        peft_other._get_module_names_tied_with_embedding = safe
        print("[patch] PEFT _get_module_names_tied_with_embedding wrapped (NotImplementedError -> empty set)", file=sys.stderr)
    except Exception as e:
        print(f"[patch] PEFT monkey-patch failed: {e}", file=sys.stderr)


_patch_peft_tied_check()

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    sys.exit(cli_evaluate())
