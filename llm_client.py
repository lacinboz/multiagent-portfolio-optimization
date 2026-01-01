 # llm_client.py
# ✅ Updated to support REAL LLM value + new action: set_objective_key
# - satisfaction == "yes" => MUST accept, no actions
# - satisfaction == "no"  => MAY refine
# - Actions NOW: set_objective_key, set_w_max, set_lambda_l2, exclude_assets
# - Server-side post-guards enforce safety + prevent nonsense:
#     * If user wants diversification => do NOT increase w_max
#     * exclude_assets only if user explicitly indicated dislike-assets OR explicitly gave excluded_assets
#     * conflict (too risky + too conservative) => only mild changes (lambda_l2 and optional objective switch)
# - IMPORTANT CHANGE for your thesis goal:
#     * If extra_notes exists (and pain_points empty), LLM is allowed to interpret and can switch objective to minvar.
#     * This is where the LLM “adds value” without you hardcoding every phrase.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import requests
from dotenv import load_dotenv

load_dotenv()

Decision = Literal["accept", "refine"]

# =========================================================
# UI label constants (MUST match dashboard + portfolio_langgraph)
# =========================================================
PP_TOO_RISKY = "It feels too risky"
PP_TOO_CONSERVATIVE = "It feels too conservative"
PP_TOO_CONCENTRATED = "It’s too concentrated in a few assets"
PP_DISLIKE_ASSETS = "I don’t like some of the assets"
PP_NOT_SURE = "I’m not sure — I just want something safer/smoother"

# =========================================================
# Small helpers
# =========================================================
def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Accepts:
      - pure JSON dict
      - JSON dict wrapped in fences
      - text that contains a first {...} JSON dict
    """
    text = _strip_code_fences(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pref_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]


def _safe_bool(x: Any) -> bool:
    return bool(x) is True


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _has_meaningful_text(x: Any) -> bool:
    s = str(x or "").strip()
    return len(s) >= 8


# =========================================================
# Output schema validation (+ SAFETY LIMITS)
# =========================================================
# ✅ allow objective switching (your new portfolio_langgraph supports it)
_ALLOWED_ACTION_TYPES = {"set_objective_key", "set_w_max", "set_lambda_l2", "exclude_assets"}

_MAX_ACTIONS = 2
_MAX_EXCLUDE_TICKERS = 10

_MIN_W_MAX = 0.05
_MAX_W_MAX = 1.0

_MIN_L2 = 0.0
_MAX_L2 = 1.0


def validate_llm_decision_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not a dict"

    decision = payload.get("decision")
    if decision not in ("accept", "refine"):
        return False, "decision must be 'accept' or 'refine'"

    if "rationale" in payload and not isinstance(payload["rationale"], str):
        return False, "rationale must be a string"

    actions = payload.get("proposed_actions", [])
    if actions is None:
        actions = []
    if not isinstance(actions, list):
        return False, "proposed_actions must be a list"

    if len(actions) > _MAX_ACTIONS:
        return False, f"too many actions (max {_MAX_ACTIONS})"

    for a in actions:
        if not isinstance(a, dict):
            return False, "each action must be an object/dict"

        t = a.get("type")
        if t not in _ALLOWED_ACTION_TYPES:
            return False, f"unknown action type: {t}"

        if t == "set_objective_key":
            v = str(a.get("value") or "").strip().lower()
            if v not in ("maxsharpe", "minvar"):
                return False, "set_objective_key.value must be 'maxsharpe' or 'minvar'"

        if t in ("set_w_max", "set_lambda_l2"):
            try:
                float(a.get("value"))
            except Exception:
                return False, f"{t}.value must be numeric"

        if t == "exclude_assets":
            tickers = a.get("tickers", [])
            if not isinstance(tickers, list) or any(not isinstance(x, str) for x in tickers):
                return False, "exclude_assets.tickers must be a list[str]"
            if len(tickers) > _MAX_EXCLUDE_TICKERS:
                return False, f"exclude_assets.tickers too long (max {_MAX_EXCLUDE_TICKERS})"

    return True, "ok"


def normalize_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clamp + de-dupe by type (keep last)."""
    last: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        t = a.get("type")
        if not t:
            continue

        if t == "set_objective_key":
            v = str(a.get("value") or "").strip().lower()
            if v in ("maxsharpe", "minvar"):
                a = {**a, "value": v}
            else:
                continue

        if t == "set_w_max":
            v = float(a.get("value"))
            a = {**a, "value": _clamp_float(v, _MIN_W_MAX, _MAX_W_MAX)}

        if t == "set_lambda_l2":
            v = float(a.get("value"))
            a = {**a, "value": _clamp_float(v, _MIN_L2, _MAX_L2)}

        last[t] = a

    return list(last.values())


# =========================================================
# Deterministic “notes flags” (optional, not required)
# NOTE: We keep this, but it should NOT be your main logic anymore.
# The main value path is: LLM reads extra_notes and chooses objective/actions.
# =========================================================
def _actions_from_extra_note_flags(
    *,
    flags: Dict[str, Any],
    w_max_current: float,
    lambda_l2_current: float,
    wants_diversification: bool,
) -> List[Dict[str, Any]]:
    if not flags:
        return []

    actions: List[Dict[str, Any]] = []

    avoid_drawdowns = _safe_bool(flags.get("avoid_drawdowns"))
    safer_smoother = _safe_bool(flags.get("safer_smoother"))
    prefer_div = _safe_bool(flags.get("prefer_diversification"))
    avoid_big_positions = _safe_bool(flags.get("avoid_big_positions"))
    still_want_growth = _safe_bool(flags.get("still_want_growth"))

    # Smoothness / drawdowns: nudge towards minvar + slightly higher L2
    if avoid_drawdowns or safer_smoother:
        actions.append({"type": "set_objective_key", "value": "minvar"})
        new_l2 = float(lambda_l2_current) + 0.001
        if float(lambda_l2_current) < 0.001:
            new_l2 = 0.003
        actions.append({"type": "set_lambda_l2", "value": new_l2})

    # Diversification / avoid big positions: reduce w_max a bit (never increase)
    if (prefer_div or avoid_big_positions) and (not still_want_growth):
        target = float(w_max_current) - 0.05
        actions.append({"type": "set_w_max", "value": target})

    # Post-guards later will enforce no w_max increase if wants_diversification.
    return actions


def _merge_actions_priority(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge actions with priority: primary wins on same type."""
    by_type: Dict[str, Dict[str, Any]] = {}
    for a in secondary:
        t = a.get("type")
        if t:
            by_type[t] = a
    for a in primary:
        t = a.get("type")
        if t:
            by_type[t] = a
    return list(by_type.values())


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

    # ----------------------------
    # Decision Agent
    # ----------------------------
    def decide_refine_actions(
        self,
        *,
        mode: str,
        iteration: int,
        max_iterations: int,
        objective_key: str,
        rf: float,
        w_max: float,
        lambda_l2: float,
        selected_tickers: List[str],
        optimized_metrics: Dict[str, Any],
        optimized_weights: Dict[str, float],
        baseline_metrics: Optional[Dict[str, Any]],
        current_metrics: Optional[Dict[str, Any]],
        preferences: Dict[str, Any],
        news_signals: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        # --- hard guards first ---
        if mode == "base":
            return {"decision": "accept", "rationale": "Base mode: no refinement.", "proposed_actions": []}
        if iteration >= max_iterations:
            return {"decision": "accept", "rationale": "Max iterations reached.", "proposed_actions": []}

        prefs = preferences or {}
        satisfaction = str(prefs.get("satisfaction") or "").lower().strip()

        # satisfaction == yes => MUST accept
        if satisfaction == "yes":
            return {
                "decision": "accept",
                "rationale": "User indicated the portfolio looks good (satisfaction=yes).",
                "proposed_actions": [],
            }

        # STRICT: only refine if explicitly "no"
        if satisfaction != "no":
            return {
                "decision": "accept",
                "rationale": "No explicit dissatisfaction (satisfaction!='no'). Skipping refinement.",
                "proposed_actions": [],
            }

        pain_points = _pref_list(prefs.get("pain_points"))
        concentration = prefs.get("concentration")

        wants_div = (concentration == "low") or (PP_TOO_CONCENTRATED in pain_points)
        conflict_risk = (PP_TOO_RISKY in pain_points) and (PP_TOO_CONSERVATIVE in pain_points)

        # Allow exclude_assets ONLY if user explicitly said dislike-assets OR provided excluded_assets list
        excluded_assets_ui = prefs.get("excluded_assets") or []
        if not isinstance(excluded_assets_ui, list):
            excluded_assets_ui = []
        excluded_assets_ui = [str(x) for x in excluded_assets_ui if isinstance(x, (str, int, float))]
        user_signaled_dislike_assets = (PP_DISLIKE_ASSETS in pain_points) or (len(excluded_assets_ui) > 0)

        # extra_notes: raw user text (THIS is what you want the LLM to interpret)
        extra_notes = str(prefs.get("extra_notes") or "").strip()
        has_extra_notes = _has_meaningful_text(extra_notes)

        # Optional: extra flags from UI (if you have them)
        extra_note_flags = _safe_dict(prefs.get("extra_note_flags"))
        notes_tickers = prefs.get("notes_tickers", [])
        if not isinstance(notes_tickers, list):
            notes_tickers = []
        notes_tickers = [str(t) for t in notes_tickers if isinstance(t, (str, int, float))]
        # keep only tickers in current universe
        universe = set(map(str, selected_tickers))
        notes_tickers = [t for t in notes_tickers if t in universe][:_MAX_EXCLUDE_TICKERS]

        # Deterministic hint actions (secondary): flags -> small safe actions
        flag_actions = _actions_from_extra_note_flags(
            flags=extra_note_flags,
            w_max_current=float(w_max),
            lambda_l2_current=float(lambda_l2),
            wants_diversification=wants_div,
        )

        # If UI explicitly excluded, include it as deterministic exclusions
        if excluded_assets_ui and user_signaled_dislike_assets:
            # keep within universe
            ex = [t for t in excluded_assets_ui if t in universe][:_MAX_EXCLUDE_TICKERS]
            if ex:
                flag_actions.append({"type": "exclude_assets", "tickers": ex})

        # If notes mention tickers, treat as exclusion ONLY if user signaled dislike-assets
        if user_signaled_dislike_assets and notes_tickers:
            flag_actions.append({"type": "exclude_assets", "tickers": notes_tickers})

        constraints_hint = {
            "user_wants_diversification": wants_div,
            "do_not_increase_w_max": wants_div,
            "allow_exclude_assets": user_signaled_dislike_assets,
            "conflict_risky_and_conservative": conflict_risk,
            "extra_notes_present": has_extra_notes,
        }

        system = (
            "You are a portfolio refinement Decision Agent.\n"
            "Return ONLY a single JSON object. No markdown, no extra text.\n"
            "\n"
            "Your job:\n"
            "1) Decide if the CURRENT portfolio matches the user's feedback.\n"
            "2) If satisfied => decision='accept'.\n"
            "3) If NOT satisfied => decision='refine' and propose ONLY small safe adjustments.\n"
            "\n"
            "Allowed actions (ONLY these):\n"
            "- set_objective_key: {\"type\":\"set_objective_key\",\"value\":\"maxsharpe\"|\"minvar\"}\n"
            "- set_w_max:         {\"type\":\"set_w_max\",\"value\": number}\n"
            "- set_lambda_l2:     {\"type\":\"set_lambda_l2\",\"value\": number}\n"
            "- exclude_assets:    {\"type\":\"exclude_assets\",\"tickers\": [string,...]}\n"
            f"- Propose 0-{_MAX_ACTIONS} actions max.\n"
            f"- Never exclude more than {_MAX_EXCLUDE_TICKERS} assets.\n"
            "\n"
            "Safety rules (MUST follow):\n"
            "- If user wants diversification => do NOT propose increasing w_max.\n"
            "- Propose exclude_assets ONLY if allow_exclude_assets is true.\n"
            "- If feedback is conflicting (too risky + too conservative) => keep changes mild (prefer lambda_l2; objective switch optional).\n"
            "\n"
            "How to use extra_notes:\n"
            "- If user asks for smoother ride / avoid big drawdowns => prefer objective_key='minvar'.\n"
            "- If user says keep best returns / maximize Sharpe => keep 'maxsharpe'.\n"
            "- Use w_max + lambda_l2 to control concentration/diversification.\n"
            "\n"
            "Schema:\n"
            "{\n"
            '  \"decision\": \"accept\" | \"refine\",\n'
            '  \"rationale\": \"short reason\",\n'
            '  \"proposed_actions\": [ ... ]\n'
            "}\n"
        )

        summary = {
            "mode": mode,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "objective_key_current": objective_key,
            "rf": rf,
            "w_max_current": w_max,
            "lambda_l2_current": lambda_l2,
            "n_universe": len(selected_tickers),
            "top_weights": sorted(optimized_weights.items(), key=lambda x: x[1], reverse=True)[:5],
            "optimized_metrics": {
                "return": optimized_metrics.get("return"),
                "vol": optimized_metrics.get("vol"),
                "sharpe": optimized_metrics.get("sharpe"),
                # (optional) any extra metrics you already compute
                "max_drawdown": optimized_metrics.get("max_drawdown"),
            },
            "baseline_metrics": None
            if not baseline_metrics
            else {"return": baseline_metrics.get("return"), "vol": baseline_metrics.get("vol")},
            "current_metrics": None
            if not current_metrics
            else {"return": current_metrics.get("return"), "vol": current_metrics.get("vol")},
            "preferences": {
                **prefs,
                # make sure extra_notes is visible (this is the key)
                "extra_notes": extra_notes,
            },
            "news_signals": news_signals,
            "constraints_hint": constraints_hint,
            # This is just a hint; LLM can override. Priority merge below keeps them safe.
            "deterministic_suggestion_from_flags": flag_actions,
        }

        raw = self.chat(system=system, user="Context:\n" + json.dumps(summary, ensure_ascii=False))
        obj = _extract_first_json_object(raw)
        if obj is None:
            raise ValueError(f"LLM did not return valid JSON. Raw:\n{raw}")

        ok, reason = validate_llm_decision_payload(obj)
        if not ok:
            raise ValueError(f"Invalid LLM decision payload: {reason}. Raw:\n{raw}")

        llm_actions = normalize_actions(obj.get("proposed_actions", []) or [])
        flag_actions = normalize_actions(flag_actions)

        # Priority merge: deterministic flags win if same type (keeps stability)
        actions = _merge_actions_priority(flag_actions, llm_actions)
        actions = normalize_actions(actions)

        # Limit number of actions after merge: keep most impactful/safe
        # preference: objective switch > lambda_l2 > w_max > exclude (but exclude only when allowed)
        def _score(a: Dict[str, Any]) -> int:
            t = a.get("type")
            if t == "set_objective_key":
                return 4
            if t == "set_lambda_l2":
                return 3
            if t == "set_w_max":
                return 2
            if t == "exclude_assets":
                return 1
            return 0

        actions = sorted(actions, key=_score, reverse=True)[:_MAX_ACTIONS]

        # ----------------------------
        # Post-guards (server authority)
        # ----------------------------
        # 1) If wants_div -> no w_max increase
        if wants_div:
            filtered: List[Dict[str, Any]] = []
            for a in actions:
                if a.get("type") == "set_w_max":
                    try:
                        if float(a.get("value")) > float(w_max):
                            continue
                    except Exception:
                        continue
                filtered.append(a)
            actions = filtered

        # 2) Exclude only if user allowed it
        if not user_signaled_dislike_assets:
            actions = [a for a in actions if a.get("type") != "exclude_assets"]

        # 3) Conflict case: mild changes only (lambda_l2 and optional objective switch)
        if conflict_risk:
            actions = [a for a in actions if a.get("type") in ("set_lambda_l2", "set_objective_key")]

        # 4) If decision refine but no actions -> accept
        decision = str(obj.get("decision", "accept")).lower().strip()

        # If we have actions, refine regardless of model's decision
        if actions:
            return {
                "decision": "refine",
                "rationale": str(obj.get("rationale") or "Applying small adjustments based on feedback/notes."),
                "proposed_actions": actions,
            }

        # otherwise accept
        return {
            "decision": "accept",
            "rationale": str(obj.get("rationale") or "Looks aligned / no safe actions proposed."),
            "proposed_actions": [],
        }