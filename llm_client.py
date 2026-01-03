# llm_client.py
# ✅ Option A (robust): Decision + Explanation are DECOUPLED
# + ✅ NEW: LLM Interpretation + LLM Verifier (self-check)
#
# Flow:
# 0) LLM interprets user feedback -> tiny intent JSON (no hard mapping)
# 1) LLM chooses candidate -> FINAL_CHOICE: <candidate>
# 1.5) LLM verifies choice against intent + metric_table -> may correct
# 2) LLM generates explanation (free-form)
#
# FIXES INCLUDED:
# 1) ✅ Normalize return/vol to DECIMALS (fixes 51% vs 5.1% scale confusion)
# 2) ✅ Prefer *_pct fields in explanation context (return_pct/vol_pct/max_weight_pct)
# 3) ✅ Infer a small structured hint from extra_notes → pain_points (not a rule tree)
# 4) ✅ Decision rubric handles "accept lower returns to reduce drawdowns"
# 5) ✅ Explanation call no longer receives reasoner_text
# 6) ✅ NEW: LLM interpretation + verifier + debug logging

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import requests
from dotenv import load_dotenv

load_dotenv()

Decision = Literal["accept"]

# =========================================================
# UI label constants (MUST match dashboard + portfolio_langgraph)
# =========================================================
PP_TOO_RISKY = "It feels too risky"
PP_TOO_CONSERVATIVE = "It feels too conservative"
PP_TOO_CONCENTRATED = "It’s too concentrated in a few assets"
PP_DISLIKE_ASSETS = "I don’t like some of the assets"
PP_NOT_SURE = "I’m not sure — I just want something safer/smoother"

_ALLOWED_CANDIDATES = {"maxsharpe", "minvar"}


# =========================================================
# Small helpers
# =========================================================
def _pref_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _has_meaningful_text(x: Any) -> bool:
    s = str(x or "").strip()
    return len(s) >= 8


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None


def _norm_pct_to_decimal(x: Any) -> Any:
    """
    Enforce return/vol as decimals.
    If value looks like percent-scale (e.g., 5.1 or 51.0), convert to decimal.
    Keep small decimals (<= ~1.5) as-is.
    """
    v = _safe_float(x)
    if v is None:
        return x
    if abs(v) > 1.5:
        return v / 100.0
    return v


def _infer_pain_points_from_notes(extra_notes: str, pain_points: List[str]) -> List[str]:
    """
    Minimal structured hint from free text (NOT a rule tree):
    If notes indicate 'smoother/avoid drawdowns/safer', add one label so the LLM
    doesn't miss the intent when pain_points UI is empty.
    """
    if not extra_notes:
        return pain_points

    s = extra_notes.lower()
    risk_words = [
        "smoother",
        "smooth",
        "drawdown",
        "drawdowns",
        "big drawdown",
        "avoid big drawdowns",
        "avoid drawdowns",
        "safer",
        "lower risk",
        "less risk",
        "downside",
        "avoid losses",
        "avoid loss",
    ]
    if any(w in s for w in risk_words):
        if PP_NOT_SURE not in pain_points and PP_TOO_RISKY not in pain_points:
            pain_points = list(pain_points) + [PP_NOT_SURE]
    return pain_points


def _extract_final_choice(text: str, available: List[str]) -> Optional[str]:
    """
    Parse a line like:
      FINAL_CHOICE: minvar
    Accepts casing/whitespace variants.
    """
    if not text:
        return None
    avail_set = set(a.lower().strip() for a in available)
    m = re.search(r"FINAL_CHOICE\s*:\s*([A-Za-z0-9_\-]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    cand = m.group(1).strip().lower()
    if cand in avail_set:
        return cand
    return None


def _compact_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    """
    candidates expected shape (from portfolio_langgraph):
      {
        "weights": {...},
        "metrics": {...}   # may include *_pct fields (recommended)
      }

    ✅ IMPORTANT:
    - Normalize return/vol so the LLM never sees mixed scales (0.51 vs 51.0 etc.)
    - Pass *_pct fields through so the LLM writes "10.4%" not "0.104%".
    """
    m = _safe_dict(c.get("metrics"))
    w = _safe_dict(c.get("weights"))

    # decimals (for safe comparisons / fallback logic)
    ret = _norm_pct_to_decimal(m.get("return"))
    vol = _norm_pct_to_decimal(m.get("vol"))
    sharpe = _safe_float(m.get("sharpe"))

    # normalized display (preferred for explanation text)
    ret_pct = _safe_float(m.get("return_pct"))
    vol_pct = _safe_float(m.get("vol_pct"))
    max_w_pct = _safe_float(m.get("max_weight_pct"))

    top_w = sorted([(k, float(v)) for k, v in w.items()], key=lambda x: x[1], reverse=True)[:5]
    return {
        "metrics": {
            # raw decimals (still useful)
            "return": ret,
            "vol": vol,
            "sharpe": sharpe,
            "max_weight": m.get("max_weight"),
            "effective_n": m.get("effective_n"),
            "active_assets": m.get("active_assets"),
            # ✅ preferred for natural-language
            "return_pct": ret_pct,
            "vol_pct": vol_pct,
            "max_weight_pct": max_w_pct,
        },
        "top_weights": top_w,
    }


def validate_choice(choice: str, available: List[str]) -> Tuple[bool, str]:
    if not choice:
        return False, "choice missing"
    c = choice.lower().strip()
    if c not in [a.lower().strip() for a in available]:
        return False, f"choice must be one of {available}"
    return True, "ok"


def _sort_available(candidates: Dict[str, Any]) -> List[str]:
    """
    Stable ordering to reduce tiny-model randomness:
    - prefer known keys order: maxsharpe then minvar
    - else fallback to sorted keys
    """
    keys = list(candidates.keys())
    ordered = [k for k in ["maxsharpe", "minvar"] if k in keys]
    rest = sorted([k for k in keys if k not in ordered])
    return ordered + rest


def _extract_metric_table(ctx: Dict[str, Any], available: List[str]) -> Dict[str, Any]:
    cand_map = (ctx.get("candidates") or {})
    return {k: ((cand_map.get(k) or {}).get("metrics") or {}) for k in available}


# =========================================================
# Configs
# =========================================================
@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:3b-instruct"
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_s: float = 60.0


@dataclass(frozen=True)
class HFConfig:
    base_url: str = "https://router.huggingface.co/hf-inference/models"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_s: float = 60.0


class LLMClient:
    def __init__(self):
        self.provider = (os.getenv("LLM_PROVIDER", "ollama") or "ollama").lower().strip()

        self.ollama_cfg = OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("OLLAMA_TOP_P", "1.0")),
            timeout_s=float(os.getenv("OLLAMA_TIMEOUT_S", "60.0")),
        )

        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_cfg = HFConfig(
            base_url=os.getenv("HF_BASE_URL", "https://router.huggingface.co/hf-inference/models"),
            model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            temperature=float(os.getenv("HF_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("HF_TOP_P", "1.0")),
            max_tokens=int(os.getenv("HF_MAX_TOKENS", "512")),
            timeout_s=float(os.getenv("HF_TIMEOUT_S", "60.0")),
        )

        print(f"[LLMClient] provider={self.provider}")
        print(f"[LLMClient] ollama_model={self.ollama_cfg.model} base_url={self.ollama_cfg.base_url}")
        print(f"[LLMClient] hf_model={self.hf_cfg.model}")

    # ----------------------------
    # Transport
    # ----------------------------
    def _chat_ollama(self, system: str, user: str) -> str:
        url = f"{self.ollama_cfg.base_url}/api/chat"
        payload = {
            "model": self.ollama_cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self.ollama_cfg.temperature,
                "top_p": self.ollama_cfg.top_p,
            },
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.ollama_cfg.timeout_s)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise RuntimeError(f"Unexpected Ollama response shape: keys={list(data.keys())}")
        return msg

    def _chat_hf(self, system: str, user: str) -> str:
        if not self.hf_token or not self.hf_token.strip():
            raise RuntimeError("HF_TOKEN missing but LLM_PROVIDER=hf")

        url = f"{self.hf_cfg.base_url}/{self.hf_cfg.model}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.hf_cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.hf_cfg.temperature,
            "top_p": self.hf_cfg.top_p,
            "max_tokens": self.hf_cfg.max_tokens,
            "stream": False,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.hf_cfg.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"HF inference error {r.status_code}: {r.text}")

        data = r.json()
        try:
            msg = data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected HF response shape: keys={list(data.keys())}, body={data}")
        if not isinstance(msg, str):
            raise RuntimeError("HF returned non-string message content.")
        return msg

    def chat(self, system: str, user: str) -> str:
        if self.provider == "hf":
            return self._chat_hf(system, user)
        return self._chat_ollama(system, user)

    # =========================================================
    # NEW: Interpret user feedback (LLM) -> tiny intent JSON
    # =========================================================
    def _interpret_feedback(self, pain_points: List[str], extra_notes: str) -> Dict[str, Any]:
        """
        Returns a tiny JSON intent. No hard mapping.
        This lets us LOG what the LLM thinks "too risky" means.
        """
        system = (
            "You interpret portfolio feedback into a tiny structured intent.\n"
            "Return ONLY valid JSON (no markdown).\n"
            "Schema:\n"
            "{\n"
            '  "risk_aversion": "low"|"medium"|"high",\n'
            '  "return_seeking": "low"|"medium"|"high",\n'
            '  "prefers_diversification": true|false,\n'
            '  "notes_summary": string\n'
            "}\n"
            "Rules:\n"
            "- If pain_points include 'It feels too risky' or notes mention drawdowns/smoother/safer -> risk_aversion=high.\n"
            "- If pain_points include 'It feels too conservative' -> return_seeking=high.\n"
            "- Use notes_summary to paraphrase user intent briefly.\n"
        )
        user = json.dumps({"pain_points": pain_points, "extra_notes": extra_notes}, ensure_ascii=False)
        text = self.chat(system=system, user=user).strip()

        # best-effort JSON parse
        try:
            j = json.loads(text)
            if not isinstance(j, dict):
                raise ValueError("intent not dict")
        except Exception:
            # fallback intent (still not hard mapping; just safe default)
            j = {
                "risk_aversion": "medium",
                "return_seeking": "medium",
                "prefers_diversification": False,
                "notes_summary": (extra_notes or "")[:160],
            }

        # debug log
        if os.getenv("LLM_DEBUG_INTENT", "0") == "1":
            print("\n===== LLM DEBUG: INTERPRETED INTENT =====")
            print(json.dumps(j, indent=2))
            print("========================================\n")

        return j

    # =========================================================
    # Candidate Selection (Decision) + Verification + Explanation
    # =========================================================
    def select_candidate(
        self,
        *,
        mode: str,
        objective_key: str,
        rf: float,
        w_max: float,
        lambda_l2: float,
        selected_tickers: List[str],
        candidates: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]],
        current_metrics: Optional[Dict[str, Any]],
        preferences: Dict[str, Any],
        news_signals: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "decision": "accept",
            "chosen_candidate": "maxsharpe" | "minvar",
            "rationale": "..."
          }
        """

        # ---- hard guards ----
        if mode == "base":
            chosen0 = str(objective_key or "maxsharpe").lower().strip() or "maxsharpe"
            if chosen0 not in candidates:
                chosen0 = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
            return {
                "decision": "accept",
                "chosen_candidate": chosen0,
                "rationale": "Base mode: candidate selection disabled.",
            }

        prefs = preferences or {}
        satisfaction = str(prefs.get("satisfaction") or "").lower().strip()

        # satisfaction yes -> keep current objective (no selection)
        if satisfaction == "yes":
            chosen = str(objective_key or "maxsharpe").lower().strip() or "maxsharpe"
            if chosen not in candidates:
                chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
            return {
                "decision": "accept",
                "chosen_candidate": chosen,
                "rationale": "User satisfaction=yes; skipping candidate comparison.",
            }

        # only run selection if explicit dissatisfaction
        if satisfaction != "no":
            chosen = str(objective_key or "maxsharpe").lower().strip() or "maxsharpe"
            if chosen not in candidates:
                chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
            return {
                "decision": "accept",
                "chosen_candidate": chosen,
                "rationale": "No explicit dissatisfaction; defaulting to the current objective.",
            }

        # availability (stable order)
        available = _sort_available(candidates)
        available = [k for k in available if k in _ALLOWED_CANDIDATES] or available
        if not available:
            return {"decision": "accept", "chosen_candidate": "maxsharpe", "rationale": "No candidates provided."}

        # compact payload (normalized + includes *_pct fields)
        candidates_summary = {k: _compact_candidate(_safe_dict(v)) for k, v in candidates.items()}

        pain_points = _pref_list(prefs.get("pain_points"))
        extra_notes = str(prefs.get("extra_notes") or "").strip()

        # ✅ infer one structured hint from free-text notes (helps when UI pain_points empty)
        pain_points = _infer_pain_points_from_notes(extra_notes, pain_points)

        # ✅ NEW: LLM interprets feedback -> intent (and we can log it)
        intent = self._interpret_feedback(pain_points, extra_notes)

        ctx = {
            "objective_key_current": objective_key,
            "rf": rf,
            "w_max": w_max,
            "lambda_l2": lambda_l2,
            "n_universe": len(selected_tickers),
            "preferences": {
                **prefs,
                "satisfaction": "no",
                "pain_points": pain_points,
                "extra_notes": extra_notes,
                "extra_notes_present": _has_meaningful_text(extra_notes),
            },
            "intent": intent,  # ✅ NEW
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            "news_signals": news_signals,
            "candidates": candidates_summary,
            "available_candidates": available,
        }

        # =========================================================
        # STEP 1: DECISION (no JSON required)
        # =========================================================
        decision_system = (
            "You are a portfolio candidate comparison assistant.\n"
            "You will be given multiple candidate portfolios produced by a deterministic optimizer.\n"
            "Choose exactly ONE candidate from available_candidates that best matches the user's intent.\n"
            "\n"
            "Use the provided 'intent' as the main interpretation of the feedback.\n"
            "Decision rubric:\n"
            "- If intent.risk_aversion is high -> prefer LOWER volatility and more diversification.\n"
            "- If intent.return_seeking is high -> prefer HIGHER Sharpe ratio and/or higher return.\n"
            "- Use ONLY the provided metrics; do not invent anything.\n"
            "\n"
            "Output format:\n"
            "FINAL_CHOICE: <candidate>\n"
        )

        decision_user = "Context JSON:\n" + json.dumps(ctx, ensure_ascii=False)
        reasoner_text = self.chat(system=decision_system, user=decision_user)

        chosen = _extract_final_choice(reasoner_text, available)
        if chosen is None:
            retry_system = (
                "Return ONLY the final line:\n"
                "FINAL_CHOICE: <candidate>\n"
                f"Candidate must be one of: {available}\n"
            )
            retry_text = self.chat(system=retry_system, user=reasoner_text)
            chosen = _extract_final_choice(retry_text, available)

        # fallback if model fails formatting
        if chosen is None:
            safer_intent = str(intent.get("risk_aversion", "medium")).lower() == "high"
            if safer_intent:
                vol_map: Dict[str, float] = {}
                for k in available:
                    m = (candidates_summary.get(k) or {}).get("metrics") or {}
                    v = _safe_float(m.get("vol"))
                    if v is not None:
                        vol_map[k] = float(v)
                chosen = min(vol_map.keys(), key=lambda x: vol_map[x]) if vol_map else available[0]
            else:
                sharpe_map: Dict[str, float] = {}
                for k in available:
                    m = (candidates_summary.get(k) or {}).get("metrics") or {}
                    s = _safe_float(m.get("sharpe"))
                    if s is not None:
                        sharpe_map[k] = float(s)
                chosen = max(sharpe_map.keys(), key=lambda x: sharpe_map[x]) if sharpe_map else available[0]

        ok, _ = validate_choice(chosen, available)
        if not ok:
            chosen = available[0]

        # =========================================================
        # STEP 1.5: VERIFIER (LLM self-check)  ✅
        # =========================================================
        metric_table = _extract_metric_table(ctx, available)

        verify_system = (
            "You are a strict verifier of a portfolio choice.\n"
            "Check whether the chosen_candidate contradicts the user's intent.\n"
            "Use ONLY intent + metric_table.\n"
            "\n"
            "If intent.risk_aversion is high, choosing the higher-volatility candidate is likely a contradiction.\n"
            "If intent.return_seeking is high, choosing the clearly lower-sharpe candidate is likely a contradiction.\n"
            "\n"
            "Output ONLY one line:\n"
            "FINAL_CHOICE: <candidate>\n"
        )

        verify_payload = {
            "intent": intent,
            "chosen_candidate": chosen,
            "available_candidates": available,
            "metric_table": metric_table,
        }

        try:
            verify_text = self.chat(system=verify_system, user=json.dumps(verify_payload, ensure_ascii=False))
            verified = _extract_final_choice(verify_text, available)
            if verified and verified != chosen:
                if os.getenv("LLM_DEBUG_VERIFIER", "0") == "1":
                    print(f"[LLM Verifier] corrected choice: {chosen} -> {verified}")
                chosen = verified
        except Exception as e:
            if os.getenv("LLM_DEBUG_VERIFIER", "0") == "1":
                print(f"[LLM Verifier] skipped due to error: {e}")

        # =========================================================
        # STEP 2: EXPLANATION (free-form text)
        # =========================================================
        rationale = self.generate_candidate_explanation(
            chosen_candidate=chosen,
            available_candidates=available,
            ctx=ctx,
        )

        return {
            "decision": "accept",
            "chosen_candidate": chosen,
            "rationale": rationale,
        }

    def generate_candidate_explanation(
        self,
        *,
        chosen_candidate: str,
        available_candidates: List[str],
        ctx: Dict[str, Any],
    ) -> str:
        """
        Separate interpretability call (no JSON).
        Produces the user-facing explanation of why chosen_candidate was selected.
        """
        explain_system = (
            "You are writing a short user-facing explanation for a portfolio selection decision.\n"
            "Write 3-5 sentences.\n"
            "Requirements:\n"
            "- Mention the user's feedback (pain_points and extra_notes).\n"
            "- Compare chosen candidate vs alternatives using ONLY provided metrics.\n"
            "- Prefer *_pct fields when available and express them as percentages.\n"
            "- Do NOT contradict the metrics.\n"
            "- If assets were explicitly excluded by the user, acknowledge this clearly in the explanation.\n"
            "- Keep it clear and non-technical.\n"
        )

        payload = {
            "chosen_candidate": chosen_candidate,
            "available_candidates": available_candidates,
            "preferences": (ctx.get("preferences") or {}),
            "intent": (ctx.get("intent") or {}),
            "candidates": (ctx.get("candidates") or {}),
        }
        cand_map = (ctx.get("candidates") or {})
        payload["metric_table"] = {k: ((cand_map.get(k) or {}).get("metrics") or {}) for k in available_candidates}
        payload["excluded_assets"] = ctx["preferences"].get("excluded_assets", [])

        if os.getenv("LLM_DEBUG_METRICS", "0") == "1":
            print("\n===== LLM DEBUG: METRICS PASSED TO EXPLANATION =====")
            print(json.dumps(payload.get("metric_table"), indent=2))
            print("===================================================\n")

        text = self.chat(system=explain_system, user="Context:\n" + json.dumps(payload, ensure_ascii=False))
        text = (text or "").strip()

        if len(text) > 900:
            text = text[:900].rsplit(" ", 1)[0] + "…"

        return text or "Selected the most preference-aligned candidate based on the provided metrics and your feedback."

    # =========================================================
    # Backward compatibility (optional)
    # =========================================================
    def decide_refine_actions(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            "decision": "accept",
            "rationale": "Legacy refine-actions API disabled in A/B selection mode.",
            "proposed_actions": [],
        }

    # =========================================================
    # ✅ NEW: Insight Generator call (JSON output)  [ADD ONLY]
    # =========================================================
    def _parse_json_best_effort(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Tries to parse JSON even if the model wraps it with extra text.
        Returns (json_dict_or_none, parse_mode_string).
        """
        if not text or not isinstance(text, str):
            return None, "empty"

        s = text.strip()

        # 1) direct parse
        try:
            j = json.loads(s)
            if isinstance(j, dict):
                return j, "direct"
        except Exception:
            pass

        # 2) find first {...} block
        #    (naive but effective for LLM leakage)
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            chunk = s[start : end + 1].strip()
            try:
                j = json.loads(chunk)
                if isinstance(j, dict):
                    return j, "brace_slice"
            except Exception:
                pass

        return None, "failed"

    def _verify_insight_output_light(self, insight: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight deterministic verifier (no imports, no circular deps).
        - Ensures risk_drivers tickers are in allowed set (top_risk_drivers from payload).
        - Ensures base_vs_refine.metric_deltas exists (fills from payload.delta.metrics if missing).
        Returns {ok, issues, cleaned}
        """
        issues: List[str] = []
        cleaned = dict(insight or {})

        allowed = set()
        for side in ("base", "refine"):
            for item in ((payload.get(side) or {}).get("top_risk_drivers") or []):
                t = str((item or {}).get("ticker") or "").strip()
                if t:
                    allowed.add(t)

        rd = cleaned.get("risk_drivers")
        if isinstance(rd, list):
            kept = []
            for item in rd:
                if not isinstance(item, dict):
                    issues.append("risk_driver_item_not_dict")
                    continue
                t = str(item.get("ticker") or "").strip()
                if t and t in allowed:
                    kept.append(item)
                else:
                    issues.append(f"risk_driver_ticker_not_allowed: {t}")
            cleaned["risk_drivers"] = kept

        # Ensure metric_deltas exists
        delta_metrics = ((payload.get("delta") or {}).get("metrics") or {})
        bvr = cleaned.get("base_vs_refine")
        if not isinstance(bvr, dict):
            bvr = {}
        if "metric_deltas" not in bvr:
            bvr["metric_deltas"] = delta_metrics
        cleaned["base_vs_refine"] = bvr

        ok = len(issues) == 0
        return {"ok": ok, "issues": issues, "cleaned": cleaned}

    def generate_portfolio_insights(
        self,
        *,
        prompts: Dict[str, Any],   # artık Any çünkü içinde narrative/json olabilir
        payload: Dict[str, Any],
        mode: str = "json",        # "narrative" | "json"
        max_chars: int = 8000,
    ) -> Dict[str, Any]:
        """
        Insight Generator:
        - If mode="narrative": returns plain text for UI (no JSON parse)
        - If mode="json": parses JSON best-effort + verifies

        prompts can be either:
        A) single pack: {"system":..., "developer":..., "user":...}
        B) two packs: {"narrative": {...}, "json": {...}}
        """

        # --- pick correct prompt pack ---
        pack = prompts
        if isinstance(prompts, dict) and ("narrative" in prompts or "json" in prompts):
            pack = prompts.get(mode, {}) if isinstance(prompts.get(mode, {}), dict) else {}

        system = (pack or {}).get("system", "")
        developer = (pack or {}).get("developer", "")
        user = (pack or {}).get("user", "")

        system_full = (system.rstrip() + "\n\n" + developer.strip()).strip()
        user_text = (user or "").strip()
        if len(user_text) > max_chars:
            user_text = user_text[:max_chars] + "\n\n[TRUNCATED]\n"

        raw = self.chat(system=system_full, user=user_text)
        raw = (raw or "").strip()

        # ----------------------------
        # NARRATIVE MODE: return text directly
        # ----------------------------
        if mode == "narrative":
            if not raw:
                raw = "Insights not available (empty LLM response)."
            return {
                "ok": True,
                "text": raw,
                "issues": [],
                "raw_text": raw,
                "parse_mode": "text",
            }

        # ----------------------------
        # JSON MODE: parse + verify
        # ----------------------------
        j, parse_mode = self._parse_json_best_effort(raw)

        if j is None:
            fallback = {
                "headline": "Insights not available (LLM returned non-JSON).",
                "portfolio_story": [],
                "risk_drivers": [],
                "diversification_read": {"max_weight": None, "effective_n": None, "comment": "not provided"},
                "base_vs_refine": {"key_changes": [], "metric_deltas": ((payload.get("delta") or {}).get("metrics") or {})},
                "news_overlay": [],
                "action_suggestions_optional": [],
            }
            issues = [f"json_parse_failed(mode={parse_mode})"]
            return {"ok": False, "insight": fallback, "issues": issues, "raw_text": raw, "parse_mode": parse_mode}

        verified = self._verify_insight_output_light(j, payload)
        ok = bool(verified.get("ok"))
        issues = list(verified.get("issues") or [])
        cleaned = verified.get("cleaned") or j

        return {"ok": ok, "insight": cleaned, "issues": issues, "raw_text": raw, "parse_mode": parse_mode}

