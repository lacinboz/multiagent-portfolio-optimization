# llm_client.py
# ✅ Option A (robust): Decision + Explanation are DECOUPLED
# + ✅ NEW: LLM Interpretation + LLM Verifier (self-check)
# + ✅ NEW (ADDITIVE): News Snapshot + News Risk Check (LLM) + Deterministic verifier
#
# Flow (refine):
# 0) LLM interprets user feedback -> tiny intent JSON (no hard mapping)
# 1) LLM chooses candidate -> FINAL_CHOICE: <candidate>
# 1.5) LLM verifies choice against intent + metric_table -> may correct
# 2) LLM generates explanation (free-form)
# 3) ✅ (optional) LLM summarizes recent news -> snapshot + risk flags (verifier cleans output)
# 4) ✅ (optional) LLM proposes news actions -> deterministic cleaner -> optional LLM verifier -> deterministic cleaner

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal
from evidence_utils import assign_evidence_ids_and_map
from probabilistic_news_integration import build_article_signals_with_finbert

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
# NEWS actions schema (shared by generator + verifier + deterministic cleaner)
# =========================================================
_ALLOWED_NEWS_ACTION_TYPES = {
    "exclude_ticker",
    "set_w_max",
    "shift_objective",
    "reduce_exposure",
    "hedge",
}
# =========================================================
# CHATBOT command intents (dashboard action layer)
# =========================================================
_ALLOWED_CHAT_INTENTS = {
    "run_base_portfolio",
    "run_news_overview",
    "generate_news_actions",
    "run_refine_candidate_selection",
    "show_final_portfolio_insight",
    "clarify_news_usage_mode",
    "compare_base_refined_metrics",

}


# =========================================================
# Small helpers
# =========================================================
from datetime import datetime, timezone



def _fmt_news_dt(x: Any) -> str:
    # epoch seconds -> ISO string
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(float(x), tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return "unknown"
    # already string?
    s = str(x or "").strip()
    if not s:
        return "unknown"
    # if ISO-like, keep only date
    try:
        if s.endswith("Z"):
            s = s[:-1]
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "unknown"

def _format_evidence_snapshot_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    # Header yoksa ekle
    if not s.startswith("EVIDENCE SNAPSHOT"):
        s = "EVIDENCE SNAPSHOT\n\n" + s

    # "EVIDENCE SNAPSHOT Action:" gibi yapışmaları düzelt
    s = re.sub(r"^EVIDENCE SNAPSHOT\s+Action:", "EVIDENCE SNAPSHOT\n\nAction:", s)

    # Başlıkları normalize et: Action / Why / Evidence hep satır başında olsun
    s = re.sub(r"\s+Action:", "\n\nAction:", s)
    s = re.sub(r"\s+Why:", "\nWhy:", s)
    s = re.sub(r"\s+Evidence:", "\nEvidence:", s)

    # Action ... Why: aynı satırdaysa -> paragraf kır
    s = re.sub(r"(Action:[^\n]*?)\s+Why:", r"\1\n\nWhy:", s)

    # Why ... Evidence: aynı satırdaysa -> paragraf kır
    s = re.sub(r"(Why:[^\n]*?)\s+Evidence:", r"\1\n\nEvidence:", s)

    # Evidence satırlarını bullet yap:
    # [GOOGL_xxx] ile başlayan parçayı yeni satıra al ve "- " ekle
    s = re.sub(r"\s*\[([A-Za-z0-9_:\-]+)\]\s*", r"\n - [\1] ", s)

    # Double bullet temizliği
    s = re.sub(r"\n\s*-\s*-\s*", "\n - ", s)

    # Çoklu boşluk/boş satır normalize
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s
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


def _clamp01(x: Any) -> Optional[float]:
    v = _safe_float(x)
    if v is None:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


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
    Minimal structured hint from free text (NOT a rule tree).
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
    ✅ Normalize return/vol so the LLM never sees mixed scales
    ✅ Pass *_pct fields so the LLM writes "10.4%" not "0.104%".
    """
    m = _safe_dict(c.get("metrics"))
    w = _safe_dict(c.get("weights"))

    ret = _norm_pct_to_decimal(m.get("return"))
    vol = _norm_pct_to_decimal(m.get("vol"))
    sharpe = _safe_float(m.get("sharpe"))

    ret_pct = _safe_float(m.get("return_pct"))
    vol_pct = _safe_float(m.get("vol_pct"))
    max_w_pct = _safe_float(m.get("max_weight_pct"))

    top_w = sorted([(k, float(v)) for k, v in w.items()], key=lambda x: x[1], reverse=True)[:5]
    return {
        "metrics": {
            "return": ret,
            "vol": vol,
            "sharpe": sharpe,
            "max_weight": m.get("max_weight"),
            "effective_n": m.get("effective_n"),
            "active_assets": m.get("active_assets"),
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
    keys = list(candidates.keys())
    ordered = [k for k in ["maxsharpe", "minvar"] if k in keys]
    rest = sorted([k for k in keys if k not in ordered])
    return ordered + rest


def _extract_metric_table(ctx: Dict[str, Any], available: List[str]) -> Dict[str, Any]:
    cand_map = (ctx.get("candidates") or {})
    return {k: ((cand_map.get(k) or {}).get("metrics") or {}) for k in available}

def _extract_excluded_assets_from_text(message: str, selected_tickers: List[str]) -> List[str]:
    if not message:
        return []
    text_u = str(message).upper()
    found = []
    for t in selected_tickers or []:
        tt = str(t).upper().strip()
        if tt and tt in text_u and tt not in found:
            found.append(tt)
    return found


def _infer_chat_pain_points(message: str) -> List[str]:
    s = str(message or "").lower()
    out: List[str] = []

    risk_words = [
        "too risky", "risky", "safer", "more safe", "lower risk",
        "less risk", "smoother", "smooth", "avoid drawdown",
        "avoid drawdowns", "drawdown", "drawdowns", "defensive",
    ]
    conservative_words = [
        "too conservative", "more return", "higher return",
        "more aggressive", "aggressive", "higher risk",
    ]
    concentrated_words = [
        "too concentrated", "concentrated", "more diversified",
        "diversified", "better diversified", "spread out",
    ]
    dislike_words = [
        "exclude", "remove", "do not include", "don't include",
        "i don't like", "i do not like",
    ]
    unsure_words = [
        "not sure", "unsure", "something safer", "something smoother",
    ]

    if any(w in s for w in risk_words):
        out.append(PP_TOO_RISKY)
    if any(w in s for w in conservative_words):
        out.append(PP_TOO_CONSERVATIVE)
    if any(w in s for w in concentrated_words):
        out.append(PP_TOO_CONCENTRATED)
    if any(w in s for w in dislike_words):
        out.append(PP_DISLIKE_ASSETS)
    if any(w in s for w in unsure_words):
        out.append(PP_NOT_SURE)

    # de-dup preserve order
    seen = set()
    clean = []
    for x in out:
        if x not in seen:
            seen.add(x)
            clean.append(x)
    return clean


def _sanitize_chat_command_result(
    result: Dict[str, Any],
    *,
    selected_tickers: List[str],
    original_message: str,
) -> Dict[str, Any]:
    supported = bool(result.get("supported"))
    intent = str(result.get("intent") or "").strip()
    params = result.get("parameters") if isinstance(result.get("parameters"), dict) else {}

    if intent == "unsupported" or (not supported):
        return {
            "supported": False,
            "intent": "unsupported",
            "parameters": {},
            "reply": (
                "This assistant is limited to portfolio optimization, portfolio insights, "
                "news-based analysis, and refinement tasks inside this dashboard."
            ),
        }

    if intent not in _ALLOWED_CHAT_INTENTS:
        return {
            "supported": False,
            "intent": "unsupported",
            "parameters": {},
            "reply": (
                "This assistant is limited to portfolio optimization, news-based analysis, "
                "and refinement tasks inside this dashboard."
            ),
        }

    if intent == "run_refine_candidate_selection":
        pain_points = params.get("pain_points")
        if not isinstance(pain_points, list):
            pain_points = []
        pain_points = [str(x) for x in pain_points if str(x).strip()]

        inferred_pp = _infer_chat_pain_points(original_message)
        for pp in inferred_pp:
            if pp not in pain_points:
                pain_points.append(pp)

        excluded_assets = params.get("excluded_assets")
        if not isinstance(excluded_assets, list):
            excluded_assets = []
        excluded_assets = [str(x).upper().strip() for x in excluded_assets if str(x).strip()]

        inferred_excluded = _extract_excluded_assets_from_text(original_message, selected_tickers)
        for t in inferred_excluded:
            if t not in excluded_assets:
                excluded_assets.append(t)

        extra_notes = str(params.get("extra_notes") or "").strip()
        if not extra_notes:
            extra_notes = str(original_message or "").strip()

        params = {
            "satisfaction": "no",
            "pain_points": pain_points,
            "excluded_assets": excluded_assets,
            "extra_notes": extra_notes,
        }

    return {
        "supported": True,
        "intent": intent,
        "parameters": params,
        "reply": str(result.get("reply") or "").strip(),
    }

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
    _ALLOWED_FLAGS = {
        "none",
        "event_risk",
        "earnings_uncertainty",
        "regulatory",
        "litigation",
        "product_issue",
        "macro",
    }

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
    # NEW: Dashboard chat command interpreter
    # =========================================================
    def interpret_dashboard_chat_command(
            self,
            *,
            message: str,
            selected_tickers: Optional[List[str]] = None,
        ) -> Dict[str, Any]:

            """
            Interprets a dashboard chat message into a supported dashboard action.

            Output schema:
            {
            "supported": bool,
            "intent": "run_base_portfolio" | "run_news_overview" | "generate_news_actions" | "clarify_news_usage_mode" | "run_refine_candidate_selection" | "show_final_portfolio_insight" | "unsupported",
            "parameters": {...},
            "reply": "optional assistant reply"
            }
            """
            
            selected_tickers = selected_tickers or []
            msg = str(message or "").strip()

            if not msg:
                return {
                    "supported": False,
                    "intent": "unsupported",
                    "parameters": {},
                    "reply": "Please enter a portfolio-related command.",
                }

            system = (
                "You are a command interpreter for a portfolio dashboard chatbot.\n"
                "You are NOT a general assistant.\n"
                "Your job is to map the user's message into ONE supported dashboard intent.\n"
                "\n"
                "Supported intents:\n"
                "- run_base_portfolio\n"
                "- run_news_overview\n"
                "- generate_news_actions\n"
                "- clarify_news_usage_mode\n"
                "- run_refine_candidate_selection\n"
                "- show_final_portfolio_insight\n"
                "- compare_base_refined_metrics\n"
                "- unsupported\n"
                "Rules:\n"
                "- If the message is not about portfolio optimization, portfolio refinement, final portfolio insight, news overview, or news-based actions, return unsupported.\n"
                "- Do NOT answer the user's question directly.\n"
                "- Return ONLY valid JSON.\n"
                "- No markdown.\n"
                "- No extra text.\n"
                "\n"
                "Critical mapping rules:\n"
                "- If the user says exclude, remove, drop, without, do not include, or don't include, "
                "and mentions one or more tickers from selected_tickers, you MUST return intent='run_refine_candidate_selection'.\n"
                "- In that case, parameters.excluded_assets MUST include the mentioned ticker symbols.\n"
                "- In that case, parameters.pain_points SHOULD include 'I don’t like some of the assets'.\n"
                "- Do NOT map explicit asset exclusion requests to run_news_overview.\n"
                "- Do NOT map explicit asset exclusion requests to generate_news_actions.\n"
                "- Do NOT map explicit asset exclusion requests to run_base_portfolio.\n"
                "- If the user asks for a news snapshot, news risk check, news summary, or news overview without asking to change the portfolio, "
                "return intent='run_news_overview'.\n"
                "- If the user explicitly asks to generate actions, proposals, or suggestions from news, "
                "return intent='generate_news_actions'.\n"
                "- If the user asks to use, include, apply, or consider news in the portfolio, refinement, or decision, "
                "but does NOT clearly ask for a news overview and does NOT clearly ask to generate actions from news, "
                "return intent='clarify_news_usage_mode'.\n"
                "- If the user asks to make the portfolio safer, smoother, less risky, more defensive, or lower risk, "
                "return intent='run_refine_candidate_selection'.\n"
                "- If the user asks to compare base and refined portfolios, asks what changed after refinement, asks for before/after metrics, or says 'show me what changed', "
                "return intent='compare_base_refined_metrics'.\n"
                "- If the user asks for insight, explanation, interpretation, or a description of the current, final, or active portfolio without asking to change it, "
                "return intent='show_final_portfolio_insight'.\n"
                "- If the user asks to create or run the first portfolio, return intent='run_base_portfolio'.\n"
                "Schema:\n"
                "{\n"
                '  "supported": true|false,\n'
                '  "intent": "run_base_portfolio"|"run_news_overview"|"generate_news_actions"|"clarify_news_usage_mode"|"run_refine_candidate_selection"|"show_final_portfolio_insight"|"compare_base_refined_metrics"|"unsupported",\n'
                '  "parameters": {\n'
                '     "pain_points": [string],\n'
                '     "excluded_assets": [string],\n'
                '     "extra_notes": string\n'
                "  },\n"
                '  "reply": string\n'
                "}\n"
                "\n"
                "Intent guidance:\n"
                "- run_base_portfolio: user wants to create/run/generate the initial/base portfolio.\n"
                "- run_news_overview: user wants a news snapshot, news risk check, news summary, or news overview without changing the portfolio.\n"
                "- generate_news_actions: user explicitly wants actions/proposals/suggestions generated from the news.\n"
                "- run_refine_candidate_selection: user is unhappy with the portfolio and wants refinement, adjustments, safer/smoother portfolio, exclusions, or changes.\n"
                "- clarify_news_usage_mode: user wants news to be used in the portfolio/refinement/decision, but it is unclear whether they want mathematical integration or LLM-generated news actions.\n"
                "- show_final_portfolio_insight: user wants the current/final/active portfolio to be explained, interpreted, or summarized without changing it.\n"
                "- unsupported: anything outside this dashboard scope.\n"
            )

            user = json.dumps(
                {
                    "message": msg,
                    "selected_tickers": [str(t) for t in selected_tickers],
                },
                ensure_ascii=False,
            )

            raw = (self.chat(system=system, user=user) or "").strip()
            j, parse_mode = self._parse_json_best_effort(raw)

            if not isinstance(j, dict):
                # deterministic fallback
                s = msg.lower()

                if any(x in s for x in ["refine", "adjust", "change this portfolio", "make it safer", "make it smoother", "exclude"]):
                    j = {"supported": True, "intent": "run_refine_candidate_selection", "parameters": {}, "reply": ""}
                elif any(x in s for x in ["news action", "actions from news", "generate actions"]):
                    j = {"supported": True, "intent": "generate_news_actions", "parameters": {}, "reply": ""}
                elif any(x in s for x in ["news overview", "news snapshot", "news risk", "analyze news", "show news"]):
                    j = {"supported": True, "intent": "run_news_overview", "parameters": {}, "reply": ""}

                elif any(x in s for x in [
                    "insight", "show insight", "portfolio insight", "final portfolio insight",
                    "explain this portfolio", "explain the portfolio", "describe this portfolio",
                    "describe the portfolio", "summarize this portfolio", "summarise this portfolio",  "current portfolio explanation", "active portfolio explanation"
                ]):
                     j = {"supported": True, "intent": "show_final_portfolio_insight", "parameters": {}, "reply": ""}

                elif any(x in s for x in [
                    "compare base and refined",
                    "compare base vs refined",
                    "base vs refined",
                    "before after",
                    "before/after",
                    "show me what changed",
                    "what changed after refine",
                    "what changed after refinement",
                    "compare the portfolios",
                    "metric comparison",
                    "compare metrics",
                ]):
                    j = {"supported": True, "intent": "compare_base_refined_metrics", "parameters": {}, "reply": ""}
                elif any(x in s for x in [
                    "use news",
                    "include news",
                    "apply news",
                    "consider news",
                    "use recent news",
                    "include recent news",
                    "consider recent news",
                    "apply recent news",
                ]):
                    j = {"supported": True, "intent": "clarify_news_usage_mode", "parameters": {}, "reply": ""}
                elif any(x in s for x in ["run base", "base portfolio", "initial portfolio", "create portfolio", "generate portfolio"]):
                    j = {"supported": True, "intent": "run_base_portfolio", "parameters": {}, "reply": ""}
                else:
                    j = {"supported": False, "intent": "unsupported", "parameters": {}, "reply": ""}

            out = _sanitize_chat_command_result(
                j,
                selected_tickers=selected_tickers,
                original_message=msg,
            )

            if os.getenv("LLM_DEBUG_CHAT_INTENT", "0") == "1":
                print("\n===== LLM DEBUG: CHAT COMMAND =====")
                print("raw:", raw)
                print("parse_mode:", parse_mode)
                print(json.dumps(out, indent=2))
                print("===================================\n")

            return out

    # =========================================================
    # NEW: Interpret user feedback (LLM) -> tiny intent JSON
    # =========================================================
    def _has_meaningful_news_overlay(self, payload: Dict[str, Any]) -> bool:
        news = payload.get("news_signals") or {}
        if not isinstance(news, dict):
            return False

        by_ticker = news.get("by_ticker")
        if isinstance(by_ticker, dict):
            for _, v in by_ticker.items():
                if not isinstance(v, dict):
                    continue
                flag = str(v.get("risk_flag") or "none").strip().lower()
                conf = _safe_float(v.get("confidence"))
                if flag != "none" and (conf is None or conf > 0):
                    return True

        glob = news.get("global")
        if isinstance(glob, dict):
            rf = glob.get("risk_flags")
            if isinstance(rf, list) and len(rf) > 0:
                return True
            vr = str(glob.get("vol_regime") or "normal").strip().lower()
            if vr == "high":
                return True

        summary = str(news.get("summary") or "").strip()
        return bool(summary)
    def _build_portfolio_insight_prompt_pack(
            self,
            *,
            payload: Dict[str, Any],
            mode: str = "narrative",
        ) -> Dict[str, str]:
            has_news = self._has_meaningful_news_overlay(payload)

            if mode == "narrative":
                system = (
                    "You are the final Portfolio Insight writer for a portfolio decision product.\n"
                    "Your job is to explain the FINAL selected portfolio to a non-expert user.\n"
                    "Return ONLY plain text. No JSON. No markdown headings.\n"
                    "Use ONLY the provided payload. Do NOT invent numbers, facts, or tickers.\n"
                )

                developer = (
                    "Write a clear final portfolio explanation for a non-expert user.\n"
                    "\n"
                    "Core purpose:\n"
                    "- Explain the FINAL selected portfolio.\n"
                    "- The selected final objective is provided in the payload. You MUST describe only that final selected objective.\n"
                    "- Do NOT describe Max Sharpe unless the final selected objective is maxsharpe.\n"
                    "- Do NOT describe Min Variance unless the final selected objective is minvar.\n"
                    "- Do NOT use metrics from a non-selected candidate as if they belong to the final portfolio.\n"
                    "- If base and refine are compared, keep the comparison separate from the final selected portfolio description.\n"
                    "- This is NOT a news snapshot.\n"
                    "- This is NOT an evidence snapshot.\n"
                    "- This is the final user-facing explanation of the selected portfolio.\n"
                    "\n"
                    "Required content:\n"
                    "- Explain the portfolio objective in plain English.\n"
                    "- If you mention Max Sharpe, explain immediately that it tries to maximize return per unit of risk.\n"
                    "- If you mention Min Variance, explain immediately that it tries to reduce ups and downs (volatility).\n"
                    "- Mention at least 3 exact metrics from the payload.\n"
                    "- When mentioning return, volatility, Sharpe, max_weight, or effective_n, explain what each means for the user in simple language.\n"
                    "- Explain whether the portfolio looks more aggressive, more defensive, more concentrated, or more diversified.\n"
                    "- If base vs refine information exists, explain what changed in practice.\n"
                    "- If no meaningful base-vs-refine comparison exists, focus only on the final portfolio.\n"
                    "- Do NOT call expected return a 'target return'. Use 'expected annual return'.\n"
                    "- Do NOT interpret Sharpe ratio as a percentage return. A Sharpe ratio is not '2% excess return'.\n"
                    "- Explain Sharpe as return per unit of risk, not as a direct return percentage.\n"
                    "- If volatility delta is very small, say volatility stayed almost unchanged instead of increased/decreased.\n"
                    "- For news-integrated portfolios, say news adjusted the expected return/risk inputs before optimization; do not imply the LLM directly picked stocks from news.\n"
                    "\n"
                    "News overlay rule:\n"
                    f"- Meaningful news overlay present: {'yes' if has_news else 'no'}.\n"
                    "- If meaningful news overlay is present, add only a short final note about it.\n"
                    "- If news signals are empty or all risk flags are none, do NOT force a news paragraph.\n"
                    "\n"
                    "Style rules:\n"
                    "- Write in natural, user-friendly language.\n"
                    "- Do not sound academic.\n"
                    "- Do not just repeat numbers; interpret them.\n"
                    "- Keep it around 8 to 14 sentences.\n"
                )

                refine_obj = str(((payload.get("refine") or {}).get("objective")) or "").strip()
                base_obj = str(((payload.get("base") or {}).get("objective")) or "").strip()

                refine_metrics = _safe_dict((payload.get("refine") or {}).get("metrics"))
                base_metrics = _safe_dict((payload.get("base") or {}).get("metrics"))

                def _pct_str(m: Dict[str, Any], key_pct: str, key_dec: str) -> str:
                    v = _safe_float(m.get(key_pct))
                    if v is not None:
                        return f"{v:.2f}%"
                    v = _safe_float(m.get(key_dec))
                    if v is not None:
                        return f"{v * 100:.2f}%"
                    return "unknown"

                final_return_str = _pct_str(refine_metrics, "return_pct", "return")
                final_vol_str = _pct_str(refine_metrics, "vol_pct", "vol")
                final_sharpe_str = str(_safe_float(refine_metrics.get("sharpe")) or "unknown")
                final_maxw_str = _pct_str(refine_metrics, "max_weight_pct", "max_weight")

                base_return_str = _pct_str(base_metrics, "return_pct", "return")
                base_vol_str = _pct_str(base_metrics, "vol_pct", "vol")
                base_sharpe_str = str(_safe_float(base_metrics.get("sharpe")) or "unknown")

                user = (
                    f"THE FINAL SELECTED PORTFOLIO (this is what you must describe as 'this portfolio' / 'the final portfolio'):\n"
                    f"- objective: {refine_obj}\n"
                    f"- expected annual return: {final_return_str}\n"
                    f"- volatility: {final_vol_str}\n"
                    f"- Sharpe ratio: {final_sharpe_str}\n"
                    f"- max weight: {final_maxw_str}\n"
                    "\n"
                    f"THE BASE PORTFOLIO (only use these numbers when explicitly comparing 'compared to the base portfolio'):\n"
                    f"- objective: {base_obj}\n"
                    f"- expected annual return: {base_return_str}\n"
                    f"- volatility: {base_vol_str}\n"
                    f"- Sharpe ratio: {base_sharpe_str}\n"
                    "\n"
                    "CRITICAL: Every sentence that describes 'this portfolio', 'the final portfolio', or "
                    "'the selected portfolio' MUST use ONLY the FINAL SELECTED PORTFOLIO numbers above. "
                    "Never use the BASE PORTFOLIO numbers except inside an explicit before/after comparison sentence.\n"
                    "\n"
                    "Important wording rules:\n"
                    "- Use 'expected annual return', not 'target return'.\n"
                    "- Do not describe Sharpe as a percent return.\n"
                    "- If volatility change is near zero, say it stayed almost unchanged.\n"
                    "\n"
                    "Here is the full payload as JSON for additional context (holdings, risk drivers, news overlay):\n\n"
                    + json.dumps(payload, ensure_ascii=False)
                )
                return {"system": system, "developer": developer, "user": user}

            system = (
                "You are the final Portfolio Insight writer for a portfolio decision product.\n"
                "Return ONLY valid JSON. No markdown. No extra text.\n"
                "Use ONLY the provided payload. Do NOT invent numbers, facts, or tickers.\n"
            )

            developer = (
                "Create a structured final portfolio insight for a non-expert user.\n"
                "The JSON must explain the FINAL selected portfolio.\n"
                "\n"
                "Output schema:\n"
                "{\n"
                '  "headline": string,\n'
                '  "portfolio_story": [string, ...],\n'
                '  "risk_drivers": [{"ticker": string, "reason": string, "rc_pct": number|null}],\n'
                '  "diversification_read": {"max_weight": number|null, "effective_n": number|null, "comment": string},\n'
                '  "base_vs_refine": {"key_changes": [string, ...], "metric_deltas": object},\n'
                '  "news_overlay": [string, ...],\n'
                '  "action_suggestions_optional": [string, ...]\n'
                "}\n"
                "\n"
                "Rules:\n"
                "- All 7 keys must be present.\n"
                "- risk_drivers.ticker must come only from payload.base.top_risk_drivers or payload.refine.top_risk_drivers.\n"
                "- portfolio_story should explain the final portfolio in user-friendly language.\n"
                "- diversification_read must interpret concentration/diversification.\n"
                "- base_vs_refine should explain practical change if available.\n"
                "- news_overlay should be empty if there is no meaningful news signal.\n"
                "- action_suggestions_optional may stay empty.\n"
            )

            user = (
                "Here is the final portfolio insight payload as JSON.\n"
                "Return the structured insight JSON now.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            )

            return {"system": system, "developer": developer, "user": user}
    def _interpret_feedback(self, pain_points: List[str], extra_notes: str) -> Dict[str, Any]:
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

        try:
            j = json.loads(text)
            if not isinstance(j, dict):
                raise ValueError("intent not dict")
        except Exception:
            j = {
                "risk_aversion": "medium",
                "return_seeking": "medium",
                "prefers_diversification": False,
                "notes_summary": (extra_notes or "")[:160],
            }

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

        if satisfaction == "yes":
            chosen = str(objective_key or "maxsharpe").lower().strip() or "maxsharpe"
            if chosen not in candidates:
                chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
            return {
                "decision": "accept",
                "chosen_candidate": chosen,
                "rationale": "User satisfaction=yes; skipping candidate comparison.",
            }

        if satisfaction != "no":
            chosen = str(objective_key or "maxsharpe").lower().strip() or "maxsharpe"
            if chosen not in candidates:
                chosen = "maxsharpe" if "maxsharpe" in candidates else next(iter(candidates.keys()))
            return {
                "decision": "accept",
                "chosen_candidate": chosen,
                "rationale": "No explicit dissatisfaction; defaulting to the current objective.",
            }

        available = _sort_available(candidates)
        available = [k for k in available if k in _ALLOWED_CANDIDATES] or available
        if not available:
            return {"decision": "accept", "chosen_candidate": "maxsharpe", "rationale": "No candidates provided."}

        candidates_summary = {k: _compact_candidate(_safe_dict(v)) for k, v in candidates.items()}

        pain_points = _pref_list(prefs.get("pain_points"))
        extra_notes = str(prefs.get("extra_notes") or "").strip()

        pain_points = _infer_pain_points_from_notes(extra_notes, pain_points)

        # ✅ NEW: LLM interprets feedback -> intent
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
            "intent": intent,
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            # NOTE: news_signals intentionally excluded from candidate selection context.
            # News is handled in a separate stage (snapshot/actions) and must not influence this decision.
            "candidates": candidates_summary,
            "available_candidates": available,
        }

        decision_system = (
            "You are a portfolio candidate comparison assistant.\n"
            "Choose exactly ONE candidate from available_candidates that best matches the user's intent.\n"
            "Use the provided 'intent' as the main interpretation of the feedback.\n"
            "Decision rubric:\n"
            "- If intent.risk_aversion is high -> prefer LOWER volatility and more diversification.\n"
            "- If intent.return_seeking is high -> prefer HIGHER Sharpe ratio and/or higher return.\n"
            "Use ONLY the provided metrics; do not invent anything.\n"
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
        # STEP 1.5: VERIFIER (LLM self-check)
        # =========================================================
        metric_table = _extract_metric_table(ctx, available)

        verify_system = (
            "You are a strict verifier of a portfolio choice.\n"
            "Check whether the chosen_candidate contradicts the user's intent.\n"
            "Use ONLY intent + metric_table.\n"
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
        explain_system = (
            "You are writing a short user-facing explanation for a portfolio selection decision.\n"
            "Write 3-5 sentences.\n"
            "Requirements:\n"
            "- Mention the user's feedback (pain_points and extra_notes).\n"
            "- Compare chosen candidate vs alternatives using ONLY provided metrics.\n"
            "- Prefer *_pct fields when available and express them as percentages.\n"
            "- Do NOT contradict the metrics.\n"
            "- If assets were explicitly excluded by the user, acknowledge this clearly.\n"
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
    # ✅ NEW: Robust JSON parsing helper (used by insights + news)
    # =========================================================
    def _parse_json_best_effort(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        if not text or not isinstance(text, str):
            return None, "empty"

        s = text.strip()

        try:
            j = json.loads(s)
            if isinstance(j, dict):
                return j, "direct"
        except Exception:
            pass

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

    # =========================================================
    # ✅ NEW: Insight Generator call (supports "narrative" + "json")
    # =========================================================
    def _verify_insight_output_light(self, insight: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
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
        payload: Dict[str, Any],
        mode: str = "narrative",  # "narrative" | "json"
        max_chars: int = 8000,
    ) -> Dict[str, Any]:
        """
        Final Portfolio Insight (LLM)
        Purpose:
        - Explain the FINAL selected portfolio to the user
        - Works for both base and refine outputs
        - News is only an optional overlay, not the main content

        Returns:
        narrative mode:
            { ok, text, issues, raw_text, parse_mode }

        json mode:
            { ok, insight, issues, raw_text, parse_mode }
        """
        mode = str(mode or "narrative").strip().lower()
        if mode not in ("narrative", "json"):
            mode = "narrative"

        pack = self._build_portfolio_insight_prompt_pack(payload=payload, mode=mode)

        system = (pack.get("system") or "").strip()
        developer = (pack.get("developer") or "").strip()
        user_text = (pack.get("user") or "").strip()

        system_full = (system + "\n\n" + developer).strip()

        if len(user_text) > max_chars:
            user_text = user_text[:max_chars] + "\n\n[TRUNCATED]\n"

        raw = (self.chat(system=system_full, user=user_text) or "").strip()

        # -------------------------------------------------
        # narrative mode
        # -------------------------------------------------
        if mode == "narrative":
            if not raw:
                fallback = "Portfolio insight unavailable (empty LLM response)."
                return {
                    "ok": False,
                    "text": fallback,
                    "issues": ["empty_llm_response"],
                    "raw_text": raw,
                    "parse_mode": "text",
                }

            issues = []
            refine_obj = str(((payload.get("refine") or {}).get("objective")) or "").lower().strip()
            raw_lower = raw.lower()

            if refine_obj == "minvar":
                if (
                    "maximize return per unit of risk" in raw_lower
                    or "max sharpe" in raw_lower
                ):
                    issues.append("objective_mismatch_minvar_written_as_maxsharpe")

            elif refine_obj == "maxsharpe":
                if (
                    "min variance" in raw_lower
                    or "reducing volatility" in raw_lower
                    or "reduce ups and downs" in raw_lower
                ):
                    issues.append("objective_mismatch_maxsharpe_written_as_minvar")

            return {
                "ok": len(issues) == 0,
                "text": raw,
                "issues": issues,
                "raw_text": raw,
                "parse_mode": "text",
            }
        # -------------------------------------------------
        # json mode
        # -------------------------------------------------
        j, parse_mode = self._parse_json_best_effort(raw)

        if j is None:
            repair_system = (
                "You are a JSON repair tool.\n"
                "Convert the given text into VALID JSON that matches the schema.\n"
                "Return ONLY JSON. No markdown. No extra text.\n"
                "The output MUST start with '{' and end with '}'.\n"
            )

            schema_hint = (
                '{'
                '"headline":"string",'
                '"portfolio_story":["string"],'
                '"risk_drivers":[{"ticker":"string","reason":"string","rc_pct":null}],'
                '"diversification_read":{"max_weight":null,"effective_n":null,"comment":"string"},'
                '"base_vs_refine":{"key_changes":["string"],"metric_deltas":{}},'
                '"news_overlay":["string"],'
                '"action_suggestions_optional":["string"]'
                '}'
            )

            repair_user = json.dumps(
                {"schema": schema_hint, "text": raw},
                ensure_ascii=False,
            )

            repaired = (self.chat(system=repair_system, user=repair_user) or "").strip()
            j2, parse_mode2 = self._parse_json_best_effort(repaired)

            if j2 is not None:
                j = j2
                parse_mode = f"repair:{parse_mode2}"
            else:
                fallback = {
                    "headline": "Insights not available (LLM returned non-JSON).",
                    "portfolio_story": [],
                    "risk_drivers": [],
                    "diversification_read": {
                        "max_weight": None,
                        "effective_n": None,
                        "comment": "not provided",
                    },
                    "base_vs_refine": {
                        "key_changes": [],
                        "metric_deltas": ((payload.get("delta") or {}).get("metrics") or {}),
                    },
                    "news_overlay": [],
                    "action_suggestions_optional": [],
                }
                return {
                    "ok": False,
                    "insight": fallback,
                    "issues": [
                        f"json_parse_failed(mode={parse_mode})",
                        f"json_repair_failed(mode={parse_mode2})",
                    ],
                    "raw_text": raw,
                    "parse_mode": parse_mode2,
                }

        verified = self._verify_insight_output_light(j, payload)
        ok = bool(verified.get("ok"))
        issues = list(verified.get("issues") or [])
        cleaned = verified.get("cleaned") or j

        return {
            "ok": ok,
            "insight": cleaned,
            "issues": issues,
            "raw_text": raw,
            "parse_mode": parse_mode,
        }


    def _clean_evidence_list(self, ev: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(ev, list):
            return out
        for x in ev:
            if not isinstance(x, dict):
                continue
            headline = str(x.get("headline") or "").strip()
            if not headline:
                continue
            if len(headline) > 180:
                headline = headline[:177] + "..."
            date = str(x.get("date") or "").strip() or None
            source = str(x.get("source") or "").strip() or None
            e = {"headline": headline, "date": date, "source": source}
            url = str(x.get("url") or "").strip()
            if url:
                e["url"] = url
            out.append(e)
            if len(out) >= 3:
                break
        return out

    # =========================================================
    # ✅ NEW: News Snapshot + News Risk Check (ADD ONLY)
    # =========================================================
    def _collect_allowed_evidence_ids(
        self,
        *,
        snapshot_text: Optional[str],
        risk_json: Optional[Dict[str, Any]],
    ) -> set[str]:
        """
        Allowed evidence IDs are:
        - IDs appearing in snapshot_text bullet tags: ([ID] ...)
        - IDs listed in risk_json.by_ticker[*].evidence_ids
        """
        allowed: set[str] = set()

        # From snapshot_text: match "([ID]" where ID is inside [ ... ]
        st = str(snapshot_text or "")
        # bullet format: " - ([ID] DATE | SOURCE) ..."
        for m in re.finditer(r"\(\[([^\]]+)\]", st):
            _id = m.group(1).strip()
            if _id:
                allowed.add(_id)

        # From risk_json.by_ticker.*.evidence_ids
        rj = risk_json if isinstance(risk_json, dict) else {}
        bt = rj.get("by_ticker")
        if isinstance(bt, dict):
            for _, v in bt.items():
                if not isinstance(v, dict):
                    continue
                ev = v.get("evidence_ids")
                if isinstance(ev, list):
                    for x in ev:
                        sx = str(x).strip()
                        if sx:
                            allowed.add(sx)

        return allowed

    def _verify_news_risk_json(
        self,
        j: Dict[str, Any],
        tickers: List[str],
        *,
        allowed_evidence_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic cleaner:
        - keeps keys: summary, by_ticker, global
        - enforces ticker whitelist
        - clamps confidence to [0,1]
        - ✅ filters evidence_ids against allowed_evidence_ids (if provided)
        """
        issues: List[str] = []
        allowed = set(str(t).upper() for t in (tickers or []))

        cleaned: Dict[str, Any] = {
            "summary": "",
            "by_ticker": {},
            "global": {"risk_flags": [], "vol_regime": "normal"},
        }

        if not isinstance(j, dict):
            return {"ok": False, "issues": ["news_not_dict"], "cleaned": cleaned}

        if isinstance(j.get("summary"), str):
            cleaned["summary"] = j["summary"].strip()

        bt = j.get("by_ticker")
        if isinstance(bt, dict):
            for k, v in bt.items():
                t = str(k).upper().strip()
                if t not in allowed:
                    issues.append(f"ticker_not_allowed:{t}")
                    continue
                if not isinstance(v, dict):
                    issues.append(f"ticker_value_not_dict:{t}")
                    continue

                risk_flag = str(v.get("risk_flag") or "none").strip().lower()
                if risk_flag not in self._ALLOWED_FLAGS:
                    risk_flag = "none"
                conf = _clamp01(v.get("confidence"))

                ev = v.get("evidence_ids")
                ev_ids: List[str] = []
                if isinstance(ev, list):
                    tmp = [str(x).strip() for x in ev if str(x).strip()]
                    if allowed_evidence_ids is not None:
                        tmp = [x for x in tmp if x in allowed_evidence_ids]
                    ev_ids = tmp

                cleaned["by_ticker"][t] = {
                    "risk_flag": risk_flag,
                    "confidence": conf,
                    "evidence_ids": ev_ids,
                }

        glob = j.get("global")
        if isinstance(glob, dict):
            vr = str(glob.get("vol_regime") or "normal").strip().lower()
            if vr not in ("normal", "high"):
                vr = "normal"
            cleaned["global"]["vol_regime"] = vr

            rf = glob.get("risk_flags")
            if isinstance(rf, list):
                kept_flags = []
                for item in rf:
                    if not isinstance(item, dict):
                        continue
                    t = str(item.get("ticker") or "").upper().strip()
                    flag = str(item.get("flag") or "").strip()
                    if t and (t in allowed) and flag:
                        kept_flags.append({"ticker": t, "flag": flag})
                cleaned["global"]["risk_flags"] = kept_flags

        ok = len(issues) == 0
        return {"ok": ok, "issues": issues, "cleaned": cleaned}
    def format_news_snapshot_strict(
        self,
        *,
        tickers: List[str],
        news_items: List[Dict[str, Any]],   # items (evidence_id'li)
        draft_snapshot_text: str,
        draft_risk_json: Dict[str, Any],
        max_bullets_per_ticker: int = 6,
    ) -> Dict[str, Any]:
        """
        LLM-2 formatter:
        - snapshot_text'i kanonik formata sokar ve evidence_id ekler
        - risk_json.by_ticker.*.evidence_ids üretir (sadece verilen evidence_id'lerden)
        - ÇIKTI: {"snapshot_text": str, "risk_json": dict}
        """
        tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]

        # küçük, güvenli paket
        compact_items = []
        for it in (news_items or []):
            compact_items.append(
                {
                    "evidence_id": it.get("evidence_id"),
                    "ticker": str(it.get("ticker") or "").upper().strip(),
                    "date": it.get("date"),
                    "source": it.get("source"),
                    "headline": it.get("headline"),
                    "url": it.get("url"),
                }
            )

        system = (
            "You are a strict NEWS snapshot formatter.\n"
            "Your job: produce a canonical snapshot_text and a risk_json that references evidence_ids.\n"
            "Return ONLY valid JSON (no markdown, no extra text). Output must start with '{' and end with '}'.\n"
            "\n"
            "Hard rules:\n"
            f"- Allowed tickers: {tickers_u}\n"
            "- You may use ONLY evidence_id values that appear in news_items.\n"
            f"- Max {max_bullets_per_ticker} bullets per ticker.\n"
            "- Do NOT invent any events or companies.\n"
            "\n"
            "snapshot_text rules:\n"
            "- Must be ticker-by-ticker.\n"
            "- Each bullet MUST be anchored to exactly ONE news item.\n"
            "- Bullet format MUST be exactly:\n"
            " - ([EVIDENCE_ID] DATE | SOURCE) HEADLINE: <headline> — <one short explanation>\n"
            "- DATE must come from news_items.date (or 'unknown').\n"
            "- SOURCE must come from news_items.source (or 'unknown').\n"
            "\n"
            "risk_json schema:\n"
            "{\n"
            '  \"summary\": string,\n'
            '  \"by_ticker\": {\n'
            '     \"TICKER\": {\"risk_flag\": \"none|event_risk|earnings_uncertainty|regulatory|litigation|product_issue|macro\", \"confidence\": number, \"evidence_ids\": [\"TICKER_01\"]}\n'
            "  },\n"
            '  \"global\": {\"risk_flags\": [{\"ticker\":\"TICKER\",\"flag\":\"string\"}], \"vol_regime\": \"normal|high\"}\n'
            "}\n"
        )

        user = json.dumps(
            {
                "tickers": tickers_u,
                "news_items": compact_items,
                "draft_snapshot_text": draft_snapshot_text or "",
                "draft_risk_json": draft_risk_json or {},
            },
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()
        j, _mode = self._parse_json_best_effort(raw)

        # fail-safe: draft'ı bozma
        if not isinstance(j, dict):
            return {
                "ok": False,
                "snapshot_text": draft_snapshot_text or "",
                "risk_json": draft_risk_json or {},
                "raw_text": raw,
            }

        out_snapshot = j.get("snapshot_text")
        out_risk = j.get("risk_json")

        if not isinstance(out_snapshot, str):
            out_snapshot = draft_snapshot_text or ""
        if not isinstance(out_risk, dict):
            out_risk = draft_risk_json or {}

        return {"ok": True, "snapshot_text": out_snapshot.strip(), "risk_json": out_risk, "raw_text": raw}

    def generate_evidence_snapshot_from_actions(
        self,
        *,
        actions: List[Dict[str, Any]],
        news_items: List[Dict[str, Any]],
        max_items: int = 12,
    ) -> Dict[str, Any]:
        """
        Evidence Snapshot (LLM) — ACTION-CENTRIC + NATURAL LANGUAGE
        - Collect evidence_ids from actions
        - Pull matching news_items (headline + summary)
        - Ask LLM to write a short 'Why' narrative per action + list evidence used
        Output: { ok, evidence_snapshot_text, used_evidence_ids, issues, raw_text }
        """
        issues: List[str] = []

        # 1) collect evidence_ids from actions (preserve order)
        eids: List[str] = []
        for a in (actions or []):
            if not isinstance(a, dict):
                continue
            ev = a.get("evidence_ids")
            if isinstance(ev, list):
                for x in ev:
                    sx = str(x).strip()
                    if sx:
                        eids.append(sx)

        seen = set()
        eids_u: List[str] = []
        for x in eids:
            if x not in seen:
                seen.add(x)
                eids_u.append(x)

        if not eids_u:
            return {
                "ok": False,
                "evidence_snapshot_text": "No evidence_ids found in actions.",
                "used_evidence_ids": [],
                "issues": ["no_evidence_ids_in_actions"],
                "raw_text": "",
            }

        # 2) map news_items by evidence_id
        by_id: Dict[str, Dict[str, Any]] = {}
        for it in (news_items or []):
            if not isinstance(it, dict):
                continue
            _id = str(it.get("evidence_id") or "").strip()
            if _id:
                by_id[_id] = it

        matched: List[Dict[str, Any]] = []
        for eid in eids_u:
            it = by_id.get(eid)
            if not it:
                continue
            headline = str(it.get("headline") or "").strip()
            if not headline:
                continue

            matched.append(
                {
                    "evidence_id": eid,
                    "ticker": str(it.get("ticker") or "").upper().strip(),
                    "date": str(it.get("date") or "unknown"),
                    "source": str(it.get("source") or it.get("provider") or "unknown"),
                    "headline": headline[:180],
                    # ✅ NEW: give the LLM material to write a narrative
                    "summary": (str(it.get("summary") or "").strip() or None),
                    "url": (str(it.get("url") or "").strip() or None),
                }
            )

        if not matched:
            return {
                "ok": False,
                "evidence_snapshot_text": "Evidence ids were present but no matching news_items were found.",
                "used_evidence_ids": eids_u[: max(1, int(max_items))],
                "issues": ["no_matching_news_items"],
                "raw_text": "",
            }

        matched = matched[: max(1, int(max_items))]

        # 3) Compact actions for LLM (so it writes per-action narrative)
        compact_actions: List[Dict[str, Any]] = []
        for a in (actions or []):
            if not isinstance(a, dict):
                continue
            out = {"type": str(a.get("type") or "").strip(), "reason": str(a.get("reason") or "").strip()}
            if "ticker" in a:
                out["ticker"] = str(a.get("ticker") or "").upper().strip()
            if "intensity" in a:
                out["intensity"] = str(a.get("intensity") or "").strip()
            if "value" in a:
                out["value"] = a.get("value")
            if "to" in a:
                out["to"] = str(a.get("to") or "").strip()
            if "hedge_hint" in a:
                out["hedge_hint"] = str(a.get("hedge_hint") or "").strip()

            ev = a.get("evidence_ids")
            if isinstance(ev, list):
                out["evidence_ids"] = [str(x).strip() for x in ev if str(x).strip()]
            else:
                out["evidence_ids"] = []
            compact_actions.append(out)

        # 4) LLM renders ONLY the "Why" narrative per action.
        # The "Evidence:" lines are NOT generated by the LLM — they are
        # injected deterministically from `matched` below, so the headline
        # text shown to the user is always exactly what was fetched,
        # never a paraphrase or hallucination.
        system = (
            "You write short, user-friendly explanations for WHY each portfolio action was proposed.\n"
            "Return ONLY valid JSON (no markdown, no extra text). Output must start with '[' and end with ']'.\n"
            "\n"
            "Hard rules:\n"
            "- Use ONLY the provided evidence_items (headline + summary). Do NOT invent facts.\n"
            "- If evidence summary is missing, rely only on the headline and explicitly say 'headline-only'.\n"
            "- NEVER expand, translate, or reinterpret a ticker symbol into a company name.\n"
            "  Use the ticker symbol exactly as given in the action (e.g. write 'MSTR', not\n"
            "  'Marshall & Sterling' or any other guessed expansion). If you don't know what a\n"
            "  ticker stands for, just use the symbol as-is.\n"
            "\n"
            "CRITICAL QUALITY RULES (must follow):\n"
            "- Do NOT copy the action's original reason sentence verbatim.\n"
            "- Each 'why' must be 2-4 sentences.\n"
            "- At least ONE sentence must reference specifics from the evidence headlines/summaries\n"
            "  (e.g., mention the topic/theme of the headline, not generic wording).\n"
            "- If the evidence is weak/indirect, say so explicitly and justify the action as precautionary.\n"
            "- Do NOT include evidence_id, headline, date, or source text in your output — only the 'why' explanation.\n"
            "\n"
            "Output schema (JSON array, one object per action, in the given order):\n"
            '[{"action_index": 0, "why": "..."}, {"action_index": 1, "why": "..."}, ...]\n'
        )
        user = json.dumps(
            {
                "actions": compact_actions,
                "evidence_items": matched,
            },
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()

        # parse the JSON array of {action_index, why}
        why_by_index: Dict[int, str] = {}
        try:
            parsed_arr = json.loads(raw)
            if isinstance(parsed_arr, list):
                for item in parsed_arr:
                    if not isinstance(item, dict):
                        continue
                    idx = item.get("action_index")
                    why_text = str(item.get("why") or "").strip()
                    if isinstance(idx, int) and why_text:
                        why_by_index[idx] = why_text
        except Exception:
            # best-effort brace/bracket slice fallback
            try:
                start = raw.find("[")
                end = raw.rfind("]")
                if start >= 0 and end > start:
                    parsed_arr = json.loads(raw[start : end + 1])
                    if isinstance(parsed_arr, list):
                        for item in parsed_arr:
                            if not isinstance(item, dict):
                                continue
                            idx = item.get("action_index")
                            why_text = str(item.get("why") or "").strip()
                            if isinstance(idx, int) and why_text:
                                why_by_index[idx] = why_text
            except Exception:
                pass

        # ✅ Deterministically build the final text: LLM supplies only "why",
        # everything else (Action header + Evidence lines) comes straight
        # from `compact_actions` / `matched`, never from the LLM.
        lines: List[str] = ["EVIDENCE SNAPSHOT"]
        for i, a in enumerate(compact_actions):
            a_type = a.get("type", "")
            details = ""
            if a_type == "exclude_ticker":
                details = f"{a.get('ticker','')}"
            elif a_type == "reduce_exposure":
                details = f"{a.get('ticker','')} ({a.get('intensity','medium')})"
            elif a_type == "set_w_max":
                details = f"{a.get('value')}"
            elif a_type == "shift_objective":
                details = f"to {a.get('to')}"
            elif a_type == "hedge":
                details = f"{a.get('hedge_hint','')}"

            why_text = why_by_index.get(i) or "Derived from the linked evidence headlines/summaries below."

            lines.append("")
            lines.append(f"Action: {a_type} {details}".strip())
            lines.append(f"Why: {why_text}")
            lines.append("Evidence:")
            for eid in (a.get("evidence_ids") or [])[:3]:
                it = by_id.get(eid) or {}
                lines.append(
                    f" - [{eid}] {str(it.get('ticker') or '').upper()} | {str(it.get('date') or 'unknown')} | "
                    f"{str(it.get('source') or it.get('provider') or 'unknown')} — {str(it.get('headline') or '')[:180]}"
                )

        raw = "\n".join(lines)
        raw = _format_evidence_snapshot_text(raw)

        # 5) Fail-safe fallback if LLM empty
        if not raw:
            used_ids = [x["evidence_id"] for x in matched]
            # deterministic tiny fallback
            lines = ["EVIDENCE SNAPSHOT"]
            for a in compact_actions:
                a_type = a.get("type", "")
                details = ""
                if a_type == "exclude_ticker":
                    details = f"{a.get('ticker','')}"
                elif a_type == "reduce_exposure":
                    details = f"{a.get('ticker','')} ({a.get('intensity','medium')})"
                elif a_type == "set_w_max":
                    details = f"{a.get('value')}"
                elif a_type == "shift_objective":
                    details = f"to {a.get('to')}"
                elif a_type == "hedge":
                    details = f"{a.get('hedge_hint','')}"
                lines.append(f"Action: {a_type} {details}".strip())
                lines.append("Why: Derived from the linked evidence headlines/summaries.")
                lines.append("Evidence:")
                for eid in (a.get("evidence_ids") or [])[:3]:
                    it = by_id.get(eid) or {}
                    lines.append(
                        f" - [{eid}] {str(it.get('ticker') or '').upper()} | {str(it.get('date') or 'unknown')} | "
                        f"{str(it.get('source') or it.get('provider') or 'unknown')} — {str(it.get('headline') or '')[:180]}"
                    )
            raw = "\n".join(lines)

            return {
                "ok": False,
                "evidence_snapshot_text": raw,
                "used_evidence_ids": used_ids,
                "issues": ["empty_llm_response_fallback_used"],
                "raw_text": "",
            }

        return {
            "ok": True,
            "evidence_snapshot_text": raw,
            "used_evidence_ids": [x["evidence_id"] for x in matched],
            "issues": issues,
            "raw_text": raw,
        }

    def generate_news_snapshot(
        self,
        *,
        tickers: List[str],
        news_raw: List[Dict[str, Any]],
        lookback_days: int = 7,
        max_items_total: int = 60,
    ) -> Dict[str, Any]:
        """
        Produces:
        - snapshot_text_raw: UI’da göstereceğin LLM-1 “güzel” metin
        - snapshot_text: canonical (evidence_id’li) metin (actions + allowed_eids için)
        - risk_json: structured {summary, by_ticker, global} (verified/cleaned)
        """
        tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
        allowed = set(tickers_u)

        # ----------------------------
        # 0) Build items (grounding)
        # ----------------------------
        items: List[Dict[str, Any]] = []
        for it in (news_raw or []):
            t = str(it.get("ticker") or "").upper().strip()
            if t and t in allowed:
                items.append(
                    {
                        "id": it.get("id"),
                        "ticker": t,
                        "evidence_id": it.get("evidence_id"),
                        "date": it.get("date") or "unknown",
                        "headline": it.get("headline"),
                        "summary": it.get("summary"),
                        "source": it.get("source") or it.get("provider"),
                        "url": it.get("url"),
                    }
                )
        items = items[: max(0, int(max_items_total))]

        has_eids = any(isinstance(it, dict) and it.get("evidence_id") for it in (items or []))
        if not has_eids:
            items, _evidence_map = assign_evidence_ids_and_map(items)

        # ✅ DEBUG: show assigned evidence_ids
        if os.getenv("LLM_DEBUG_NEWS_EVIDENCE_IDS", "0") == "1":
            print("\n===== DEBUG: evidence_id assignment (input news_items) =====")
            by_t: Dict[str, List[Dict[str, Any]]] = {}
            for it2 in items:
                by_t.setdefault(it2.get("ticker", "UNK"), []).append(it2)

            for t2 in sorted(by_t.keys()):
                print(f"Ticker={t2} count={len(by_t[t2])}")
                for x in by_t[t2][:5]:
                    print(
                        f"  {x.get('evidence_id')} | finnhub_id={x.get('id')} | date={x.get('date')} | "
                        f"source={x.get('source')} | headline={(x.get('headline') or '')[:70]}"
                    )
            print("===========================================================\n")

        # ----------------------------
        # 1) LLM-1: snapshot + risk (no evidence_id requirement)
        # ----------------------------
        system = (
            "You summarize recent market/company news for a portfolio risk overlay.\n"
            "Return ONLY valid JSON (no markdown, no extra text).\n"
            "The output MUST start with '{' and end with '}'.\n"
            "Do NOT wrap the JSON in ``` fences.\n"
            "Do NOT use trailing commas.\n"
            "Do NOT use NaN/Infinity; use null instead.\n"
            "Use ONLY the provided news items. Do NOT invent events.\n"
            "\n"
            "CRITICAL HARD RULES:\n"
            f"- You may mention ONLY these tickers: {tickers_u}\n"
            "- Do NOT mention any other company/ticker names even as examples.\n"
            "- Do NOT create extra sections like 'General Market Context' or 'Partnership'.\n"
            "- Only output ticker sections for the allowed tickers.\n"
            "\n"
            "Schema:\n"
            "{\n"
            '  \"snapshot_text\": string,\n'
            '  \"risk_json\": {\n'
            '    \"summary\": string,\n'
            '    \"by_ticker\": { \"TICKER\": {\"risk_flag\": string, \"confidence\": number}, ... },\n'
            '    \"global\": { \"risk_flags\": [{\"ticker\": string, \"flag\": string}, ...], \"vol_regime\": \"normal\"|\"high\" }\n'
            "  }\n"
            "}\n"
            "\n"
            "Guidance:\n"
            "- risk_flag examples: none, event_risk, earnings_uncertainty, regulatory, litigation, product_issue, macro\n"
            "- confidence must be between 0 and 1.\n"
            "- vol_regime='high' only if multiple strong risk items appear.\n"
            "\n"
            "snapshot_text formatting:\n"
            "- MUST be ticker-by-ticker.\n"
            "- For EACH ticker: include UP TO 6 bullet points (0 to 6). If there are not enough relevant items, write fewer.\n"
            "- Do NOT create filler bullets.\n"
            "- If a ticker has no relevant items, output that ticker with NO bullets.\n"
            "- IMPORTANT: NEVER exceed the number of provided items for that ticker.\n"
            "- Each bullet MUST be anchored to ONE provided news item.\n"
            "- Bullet format MUST be exactly:\n"
            " - (DATE | SOURCE) HEADLINE: <headline> — <one short explanation>\n"
            "- DATE must use news_items[i].date. If missing use 'unknown'.\n"
            "- SOURCE must use news_items[i].source. If missing use 'unknown'.\n"
        )

        user = json.dumps(
            {
                "tickers": tickers_u,
                "lookback_days": int(lookback_days),
                "news_items": items,
            },
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()
        j, parse_mode = self._parse_json_best_effort(raw)

        # ----------------------------
        # 1.1) JSON repair if needed
        # ----------------------------
        if j is None:
            repair_system = (
                "You are a JSON repair tool.\n"
                "Convert the given text into VALID JSON that matches the schema.\n"
                "Return ONLY JSON. No markdown. No extra text.\n"
                "The output MUST start with '{' and end with '}'.\n"
            )
            schema_hint = (
                '{ "snapshot_text": "string", "risk_json": { "summary": "string", '
                '"by_ticker": { "TICKER": {"risk_flag": "string", "confidence": "number", "evidence_ids": ["string"]} }, '
                '"global": { "risk_flags": [{"ticker": "string", "flag": "string"}], "vol_regime": "normal|high" } } }'
            )
            repair_user = json.dumps(
                {"schema": schema_hint, "text": raw, "tickers": tickers_u},
                ensure_ascii=False,
            )
            repaired = (self.chat(system=repair_system, user=repair_user) or "").strip()
            j2, parse_mode2 = self._parse_json_best_effort(repaired)
            if j2 is not None:
                j, parse_mode = j2, f"repair:{parse_mode2}"
            else:
                return {
                    "ok": False,
                    "snapshot_text": "News snapshot unavailable (LLM returned non-JSON).",
                    "snapshot_text_raw": "",
                    "risk_json": {"summary": "", "by_ticker": {}, "global": {"risk_flags": [], "vol_regime": "normal"}},
                    "issues": [
                        f"news_json_parse_failed(mode={parse_mode})",
                        f"news_json_repair_failed(mode={parse_mode2})",
                    ],
                    "raw_text": raw,
                    "parse_mode": parse_mode2,
                }

        snapshot_text = ""
        if isinstance(j.get("snapshot_text"), str):
            snapshot_text = j["snapshot_text"].strip()
        risk_json = j.get("risk_json") if isinstance(j.get("risk_json"), dict) else {}

        # ✅ UI RAW (LLM-1’in güzel metni) — bunu dashboard’da göster
        snapshot_text_raw = snapshot_text

        # ----------------------------
        # 2) Fixer gating: LLM-2 sadece gerekiyorsa çalışsın
        # ----------------------------
        def _needs_fix_snapshot(st: str, tickers_local: List[str], max_bullets: int = 6) -> bool:
            s = st or ""
            # hiç ticker görünmüyorsa muhtemelen format bozuk
            if tickers_local and (not any(t in s for t in tickers_local)):
                return True
            # hiç bullet yoksa ( - () bekliyorsun)
            if " - (" not in s:
                return True
            # kaba “çok bullet” check (aşırı saçmalamayı yakalar)
            if s.count(" - (") > max(1, len(tickers_local)) * max_bullets:
                return True
            return False

        run_formatter2 = _needs_fix_snapshot(snapshot_text, tickers_u, 6)

        # ----------------------------
        # 3) LLM-2: canonicalize (evidence_id ekle) — sadece gerekirse
        # ----------------------------
        if run_formatter2:
            fmt = self.format_news_snapshot_strict(
                tickers=tickers_u,
                news_items=items,  # evidence_id’li items
                draft_snapshot_text=snapshot_text,
                draft_risk_json=risk_json,
                max_bullets_per_ticker=6,
            )

            if os.getenv("LLM_DEBUG_NEWS_FORMATTER2", "0") == "1":
                print("\n===== DEBUG: Formatter-2 RAW =====")
                print((fmt.get("raw_text") or "")[:2000])
                print("===== DEBUG: Formatter-2 snapshot_text preview =====")
                print((fmt.get("snapshot_text") or "")[:800])
                print("===== DEBUG: Formatter-2 risk_json keys =====")
                rj = fmt.get("risk_json") if isinstance(fmt.get("risk_json"), dict) else {}
                print(list(rj.keys()) if isinstance(rj, dict) else type(rj))
                bt = rj.get("by_ticker") if isinstance(rj, dict) else None
                print("by_ticker keys:", list(bt.keys()) if isinstance(bt, dict) else bt)
                print("==================================\n")

            snapshot_text_canonical = (fmt.get("snapshot_text") or snapshot_text).strip()
            risk_json = fmt.get("risk_json") if isinstance(fmt.get("risk_json"), dict) else risk_json
        else:
            # formatter2 gerekmediyse, canonical = LLM-1
            snapshot_text_canonical = (snapshot_text or "").strip()

        # ----------------------------
        # 4) allowed_eids çıkar + (opsiyonel) fallback
        # ----------------------------
        allowed_eids = self._collect_allowed_evidence_ids(
            snapshot_text=snapshot_text_canonical,
            risk_json=risk_json,
        )

        # ✅ Fallback: bazen model evidence_id yerine "AI report" gibi şey basıyor.
        # allowed çok küçükse bütün item evidence_id’lerini kabul et ki actions pipeline kilitlenmesin.
        if len(allowed_eids) < 5:
            allowed_eids = {
                str(it.get("evidence_id")).strip()
                for it in (items or [])
                if isinstance(it, dict) and str(it.get("evidence_id") or "").strip()
            }

        if os.getenv("LLM_DEBUG_NEWS_ALLOWED_EIDS", "0") == "1":
            print("\n===== DEBUG: allowed_eids extraction =====")
            print("items_count:", len(items))
            print("missing_evidence_id_count:", sum(1 for it in items if not it.get("evidence_id")))
            print("evidence_id_sample:", [it.get("evidence_id") for it in items[:8]])
            print("--- snapshot_text_canonical head ---")
            print((snapshot_text_canonical or "")[:600])
            print("--- allowed_eids ---")
            print("allowed_eids_count:", len(allowed_eids))
            print("allowed_eids_sample:", sorted(list(allowed_eids))[:20])
            print("========================================\n")

        # ----------------------------
        # 5) deterministic risk cleaner (evidence_ids filtered)
        # ----------------------------
        verified = self._verify_news_risk_json(
            risk_json if isinstance(risk_json, dict) else {},
            tickers_u,
            allowed_evidence_ids=allowed_eids,
        )

        return {
            "ok": bool(verified.get("ok")),
            # ✅ canonical: actions + allowed_eids için
            "snapshot_text": snapshot_text_canonical or "News snapshot generated.",
            # ✅ UI’da bunu göster (RAW / güzel metin)
            "snapshot_text_raw": snapshot_text_raw or "",
            "risk_json": verified.get("cleaned"),
            "issues": list(verified.get("issues") or []),
            "raw_text": raw,               # LLM-1 raw json text
            "parse_mode": parse_mode,
            # debug/telemetry (istersen dashboard’da gizli tut)
            "used_formatter2": bool(run_formatter2),
            "allowed_eids_count": int(len(allowed_eids)),
        }
    # ============================
# ✅ NEW: News Actions (LINE)
# ============================

    def _parse_actions_lines(
        self,
        text: str,
        *,
        tickers: List[str],
        allowed_evidence_ids: set[str],
        max_actions: int = 8,
    ) -> Dict[str, Any]:
        """
        Parses canonical action lines into list[dict].
        Drops any action that has:
        - invalid type
        - invalid ticker (if required)
        - missing/invalid evidence_ids (must be in allowed_evidence_ids)
        """
        issues: List[str] = []
        actions: List[Dict[str, Any]] = []

        tickers_u = {str(t).upper().strip() for t in (tickers or []) if str(t).strip()}
        allowed_types = set(_ALLOWED_NEWS_ACTION_TYPES)
        # ✅ 0) Extract only the ACTION BLOCK if present
        # We ignore any text outside the block.
        m = re.search(r"BEGIN_ACTIONS\s*(.*?)\s*END_ACTIONS", text or "", flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = (m.group(1) or "").strip()
        else:
            # If block missing, keep original behavior (fallback to full text)
            text = (text or "").strip()
        if os.getenv("LLM_DEBUG_ACTIONS_PARSE", "0") == "1":
            print("\n===== DEBUG: _parse_actions_lines INPUT (post-block-extract) =====")
            print("tickers_u:", sorted(list(tickers_u)))
            print("allowed_evidence_ids_count:", len(allowed_evidence_ids))
            print("text_len:", len(text))
            print("--- text head (800) ---")
            print((text or "")[:800])
            print("--- text lines ---")
            for idx, ln in enumerate((text or "").splitlines()[:20]):
                print(f"{idx:02d}: {ln}")
            print("===============================================================\n")

        # ✅ Special explicit empty case
        if text.strip().upper() == "NO_ACTIONS":
            return {"ok": True, "actions": [], "issues": ["no_actions_returned"]}


        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return {"ok": False, "actions": [], "issues": ["empty_lines"]}

        for i, ln in enumerate(lines):
            if len(actions) >= max_actions:
                break

            # allow comments or headers safely
            if ln.startswith("#"):
                continue

            parts = [p.strip() for p in ln.split("|")]
            if len(parts) < 4:
                issues.append(f"line_{i}_too_few_fields")
                continue

            a_type = parts[0]
            if a_type not in allowed_types:
                issues.append(f"line_{i}_bad_type:{a_type}")
                continue

            # evidence ids are always the second last field
            # reason is always the last field
            reason = parts[-1].strip()
            eids_raw = parts[-2].strip()

            eids = [x.strip() for x in eids_raw.split(",") if x.strip()]
            eids = [x for x in eids if x in allowed_evidence_ids]

            # ✅ Ticker-evidence mismatch: reduce_exposure|AMD için AAPL_xxx geçersiz
            _TICKER_REQUIRE_MATCH = {"reduce_exposure", "exclude_ticker"}
            if a_type in _TICKER_REQUIRE_MATCH and len(parts) > 1:
                _action_ticker = parts[1].upper()
                eids = [x for x in eids if not (
                    "_" in x and x.split("_", 1)[0].upper() != _action_ticker
                )]

            eids = eids[:6]

            if not eids:
                issues.append(f"line_{i}_missing_or_invalid_eids")
                continue

            reason_stripped = reason.strip()
            looks_truncated = bool(reason_stripped) and reason_stripped[-1] not in ".!?\"')"
            if reason_stripped.upper() in ("REASON", "") or len(reason_stripped) < 15 or looks_truncated:
                issues.append(f"line_{i}_placeholder_reason")

            out: Dict[str, Any] = {"type": a_type, "reason": reason or "Derived from recent news.", "evidence_ids": eids}

            # parse payload fields per type
            if a_type == "exclude_ticker":
                # exclude_ticker|TICKER|EIDS|REASON
                ticker = parts[1].upper()
                if ticker not in tickers_u:
                    issues.append(f"line_{i}_exclude_bad_ticker:{ticker}")
                    continue
                out["ticker"] = ticker

            elif a_type == "reduce_exposure":
                # reduce_exposure|TICKER|INTENSITY|EIDS|REASON
                if len(parts) < 5:
                    issues.append(f"line_{i}_reduce_missing_fields")
                    continue
                ticker = parts[1].upper()
                intensity = parts[2].lower().strip() or "medium"
                if ticker not in tickers_u:
                    issues.append(f"line_{i}_reduce_bad_ticker:{ticker}")
                    continue
                if intensity not in ("low", "medium", "high"):
                    intensity = "medium"
                out["ticker"] = ticker
                out["intensity"] = intensity

            elif a_type == "set_w_max":
                # set_w_max|0.30|EIDS|REASON
                try:
                    v = float(parts[1])
                except Exception:
                    issues.append(f"line_{i}_wmax_not_float")
                    continue
                if not (0.05 <= v <= 0.50):
                    issues.append(f"line_{i}_wmax_out_of_range:{v}")
                    continue
                out["value"] = v

            elif a_type == "shift_objective":
                # shift_objective|minvar|maxsharpe|EIDS|REASON
                to = parts[1].lower().strip()
                if to not in ("minvar", "maxsharpe"):
                    issues.append(f"line_{i}_shift_bad_to:{to}")
                    continue
                out["to"] = to

            elif a_type == "hedge":
                # hedge|HEDGE_HINT|EIDS|REASON
                hedge_hint = parts[1].strip()
                if not hedge_hint:
                    issues.append(f"line_{i}_hedge_missing_hint")
                    continue
                out["hedge_hint"] = hedge_hint

            actions.append(out)

        ok = len(actions) > 0 and len(issues) == 0
        return {"ok": ok, "actions": actions, "issues": issues}
    def _filter_news_actions_with_finbert(
            self,
            *,
            actions: List[Dict[str, Any]],
            news_items: List[Dict[str, Any]],
            universe_tickers: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """
            Quality filter applied AFTER all batches are generated, BEFORE round-robin merge.
            Drops actions that are:
            1) Grounded in placeholder/empty reasons.
            2) Logically inconsistent with the FinBERT sentiment of their own evidence
            (e.g. a bearish action like exclude_ticker/reduce_exposure backed by
            clearly positive news, or a risk-seeking shift_objective backed by
            clearly negative news).
            Returns: {"kept": [...], "dropped": [...], "issues": [...]}
            """
            issues: List[str] = []
            dropped: List[Dict[str, Any]] = []
            if not actions:
                        return {"kept": [], "dropped": [], "issues": []}

            debug_on = os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1"
            if debug_on:
                        print(f"\n[FINBERT FILTER] ENTER: total_actions_in={len(actions)}")

            # 1) Drop placeholder/empty reasons first (cheap, no model call needed)
            survivors: List[Dict[str, Any]] = []
            for a in actions:
                reason = str(a.get("reason") or "").strip().upper()
                if reason in ("REASON", "") or len(reason) < 15:
                    dropped.append(a)
                    issues.append(f"dropped_placeholder_reason: {a.get('type')} {a.get('ticker', '')}")
                    continue
                survivors.append(a)

            if not survivors:
                return {"kept": [], "dropped": dropped, "issues": issues}

            # 2) Build evidence_id -> news_item lookup
            by_id: Dict[str, Dict[str, Any]] = {}
            for it in (news_items or []):
                if not isinstance(it, dict):
                    continue
                eid = str(it.get("evidence_id") or "").strip()
                if eid:
                    by_id[eid] = it

            if debug_on:
                sample_keys = list(by_id.keys())[:3]
                for sk in sample_keys:
                    print(f"[DEBUG related_check] {sk} -> related={by_id[sk].get('related')!r}")

            # 2.5) Tag (do NOT drop) actions whose evidence looks like a broad
            # "listicle"/sector-roundup article rather than company-specific
            # news — i.e. the headline+summary mentions several OTHER
            # universe tickers besides the action's own ticker. This is a
            # ticker-name-independent signal (works the same for MSTR,
            # AAPL, or any of the other ~100 tickers without needing a
            # per-ticker name dictionary), and reuses the same "count how
            # many universe tickers are mentioned" approach as the
            # dashboard's _detect_mentioned_tickers. Tagging only — never
            # drops the action.
            universe_u = [str(t).upper().strip() for t in (universe_tickers or []) if str(t).strip()]

            def _count_other_universe_mentions(text: str, own_ticker: str) -> int:
                if not text or not universe_u:
                    return 0
                text_u = text.upper()
                count = 0
                for t in universe_u:
                    if t and t != own_ticker and t in text_u:
                        count += 1
                return count

            WEAK_MENTION_THRESHOLD = 3  # headline mentions 3+ other tickers -> likely a roundup/listicle

            for a in survivors:
                a_ticker = str(a.get("ticker") or "").upper().strip()
                eids = a.get("evidence_ids") or []
                weak_flags = []

                for eid in eids:
                    it = by_id.get(eid) or {}
                    headline = str(it.get("headline") or "")
                    summary = str(it.get("summary") or "")
                    combined_text = f"{headline} {summary}"

                    other_count = _count_other_universe_mentions(combined_text, a_ticker)
                    if other_count >= WEAK_MENTION_THRESHOLD:
                        weak_flags.append(eid)

                if weak_flags:
                    a["weak_evidence"] = True
                    a["weak_evidence_ids"] = weak_flags
                    if debug_on:
                        print(f"[TICKER GUARD] ⚠️ TAGGED weak_evidence: {a.get('type')} {a_ticker} | eids={weak_flags} (headline mentions {WEAK_MENTION_THRESHOLD}+ other universe tickers)")
            # 3) Score only the unique headlines actually referenced by actions (cheap)
            eids_needed: List[str] = []
            for a in survivors:
                for eid in (a.get("evidence_ids") or []):
                    if eid in by_id and eid not in eids_needed:
                        eids_needed.append(eid)

            if not eids_needed:
                # no evidence to check sentiment against -> keep as-is (can't judge)
                return {"kept": survivors, "dropped": dropped, "issues": issues}

            raw_for_finbert = [
                {
                    "ticker": by_id[eid].get("ticker"),
                    "headline": by_id[eid].get("headline"),
                    "summary": by_id[eid].get("summary") or "",
                    "source": by_id[eid].get("source") or by_id[eid].get("provider"),
                    "datetime": by_id[eid].get("datetime"),
                    "url": by_id[eid].get("url"),
                    "id": by_id[eid].get("id"),
                }
                for eid in eids_needed
            ]

            tickers_needed = sorted({str(r.get("ticker") or "").upper() for r in raw_for_finbert if r.get("ticker")})

            try:
                article_signals = build_article_signals_with_finbert(
                    raw_news=raw_for_finbert,
                    tickers=tickers_needed,
                )
            except Exception as e:
                # FinBERT unavailable/failed -> don't block the pipeline, keep actions as-is
                issues.append(f"finbert_unavailable: {e}")
                return {"kept": survivors, "dropped": dropped, "issues": issues}

# eid -> sentiment in [-1, 1] (use headline match since we passed 1:1)
            eid_sentiment: Dict[str, float] = {}
            for eid, sig in zip(eids_needed, article_signals):
                eid_sentiment[eid] = float(sig.article_sentiment)

            if debug_on:
                print("\n===== DEBUG: FinBERT evidence sentiment scores =====")
                for eid in eids_needed:
                    it = by_id.get(eid, {})
                    print(f"  {eid} | sentiment={eid_sentiment.get(eid):.3f} | headline={(it.get('headline') or '')[:90]}")
                print("====================================================\n")

            BEARISH_TYPES = {"exclude_ticker", "reduce_exposure"}
            POSITIVE_THRESHOLD = 0.25   # clearly positive
            NEGATIVE_THRESHOLD = -0.25  # clearly negative

            kept: List[Dict[str, Any]] = []
            for a in survivors:
                a_type = a.get("type")
                eids = a.get("evidence_ids") or []
                sentiments = [eid_sentiment[e] for e in eids if e in eid_sentiment]

                if not sentiments:
                    kept.append(a)  # no scorable evidence -> keep, can't judge
                    if debug_on:
                        print(f"[FINBERT FILTER] KEEP (no scorable evidence): {a_type} {a.get('ticker', a.get('to', ''))}")
                    continue

                avg_sentiment = sum(sentiments) / len(sentiments)

                mismatch = False
                mismatch_reason = ""
                if a_type in BEARISH_TYPES and avg_sentiment >= POSITIVE_THRESHOLD:
                    mismatch = True
                    mismatch_reason = f"bearish action ({a_type}) but evidence is positive"
                elif a_type == "shift_objective" and a.get("to") == "maxsharpe" and avg_sentiment <= NEGATIVE_THRESHOLD:
                    mismatch = True
                    mismatch_reason = "shift to maxsharpe but evidence is negative"
                elif a_type == "shift_objective" and a.get("to") == "minvar" and avg_sentiment >= POSITIVE_THRESHOLD:
                    mismatch = True
                    mismatch_reason = "shift to minvar but evidence is positive"
                elif a_type == "hedge" and avg_sentiment >= POSITIVE_THRESHOLD:
                    mismatch = True
                    mismatch_reason = "hedge action but evidence is positive"

                label = a.get("ticker") or a.get("to") or a.get("hedge_hint") or ""

                if mismatch:
                    dropped.append(a)
                    issues.append(
                        f"dropped_sentiment_mismatch: {a_type} {label} "
                        f"avg_sentiment={avg_sentiment:.2f}"
                    )
                    if debug_on:
                        print(f"[FINBERT FILTER] ❌ DROP: {a_type} {label} | avg_sentiment={avg_sentiment:.3f} | reason: {mismatch_reason}")
                        print(f"    evidence_ids={eids} reason_text={(a.get('reason') or '')[:100]}")
                else:
                    kept.append(a)
                    if debug_on:
                        print(f"[FINBERT FILTER] ✅ KEEP: {a_type} {label} | avg_sentiment={avg_sentiment:.3f}")

            if debug_on:
                print(f"[FINBERT FILTER] DONE: kept={len(kept)} dropped={len(dropped)} (out of {len(survivors)} survivors)\n")

            return {"kept": kept, "dropped": dropped, "issues": issues}

    def generate_news_actions_lines(
        self,
        *,
        tickers: List[str],
        news_items: List[Dict[str, Any]],
        max_actions: int = 8,
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Batched version.
        Splits news_items into ticker-based batches (~25 tickers per batch)
        and calls the single-batch generator for each, then merges results.
        This avoids overwhelming the model's context and ensures all
        tickers get a chance to be evaluated, not just the first ~35 items.
        """
        tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
        max_actions = max(1, min(int(max_actions), 12))
        # group news_items by ticker, preserving order of tickers_u
        by_ticker: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tickers_u}
        for it in (news_items or []):
            if not isinstance(it, dict):
                continue
            t = str(it.get("ticker") or "").upper().strip()
            if t in by_ticker:
                by_ticker[t].append(it)

        # ✅ NEW: build batches by ITEM COUNT (not ticker count) to keep
        # each prompt small enough for the model to follow format reliably,
        # while still allowing each ticker's full news history to be seen.
        ITEMS_PER_BATCH = 35
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_item_count = 0

        for t in tickers_u:
            t_items = by_ticker.get(t, [])
            t_count = len(t_items)
            if t_count == 0:
                continue

            if current_batch and (current_item_count + t_count) > ITEMS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_item_count = 0

            current_batch.append(t)
            current_item_count += t_count

        if current_batch:
            batches.append(current_batch)

        if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
            print(f"\n[DEBUG] generate_news_actions_lines: total_tickers={len(tickers_u)} batches={len(batches)} target_items_per_batch={ITEMS_PER_BATCH}")
            for bi, bt in enumerate(batches):
                bcount = sum(len(by_ticker.get(t, [])) for t in bt)
                print(f"[DEBUG]   batch {bi+1}: tickers={len(bt)} items={bcount}")


        # ✅ Run every batch independently (no early break) — each batch
        # generates as many grounded actions as it finds evidence for,
        # capped at a reasonable per-batch ceiling to avoid runaway output.
        PER_BATCH_CEILING = 8

        batch_results: List[List[Dict[str, Any]]] = []
        all_issues: List[str] = []
        all_raw_texts: List[str] = []
        any_fixer_used = False

        for bi, batch_tickers in enumerate(batches):
            batch_items: List[Dict[str, Any]] = []
            for t in batch_tickers:
                batch_items.extend(by_ticker.get(t, []))

            if not batch_items:
                batch_results.append([])
                continue

            out = self._generate_news_actions_lines_single_batch(
                tickers=batch_tickers,
                news_items=batch_items,
                max_actions=PER_BATCH_CEILING,
            )

            batch_actions = out.get("actions") or []
            batch_results.append(batch_actions)
            all_issues.extend(out.get("issues") or [])
            all_raw_texts.append(f"--- batch {bi+1} ---\n{out.get('raw_text') or ''}")
            if out.get("used_fixer"):
                any_fixer_used = True

            if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
                print(f"[DEBUG] Batch {bi+1} produced {len(batch_actions)} actions")

# ✅ NEW: quality filter BEFORE round-robin — drop placeholder reasons
        # and actions whose evidence sentiment contradicts the action direction
        flat_pool: List[Dict[str, Any]] = [a for batch_list in batch_results for a in batch_list]

        filter_out = self._filter_news_actions_with_finbert(
            actions=flat_pool,
            news_items=news_items,
            universe_tickers=tickers_u,
        )
        kept_actions = filter_out.get("kept") or []
        all_issues.extend(filter_out.get("issues") or [])

        if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
            print(f"[DEBUG] FinBERT quality filter: {len(flat_pool)} -> {len(kept_actions)} kept, "
                  f"{len(filter_out.get('dropped') or [])} dropped")
            for iss in (filter_out.get("issues") or [])[:20]:
                print(f"[DEBUG]   {iss}")

        # ✅ Round-robin merge across batches using the FILTERED pool,
        # rebuilt per-batch so diversity is preserved after filtering.
        kept_ids = {id(a) for a in kept_actions}
        filtered_batch_results = [
            [a for a in batch_list if id(a) in kept_ids] for batch_list in batch_results
        ]

        all_actions: List[Dict[str, Any]] = []
        idx = 0
        while len(all_actions) < max_actions:
            added_this_round = False
            for batch_list in filtered_batch_results:
                if idx < len(batch_list):
                    all_actions.append(batch_list[idx])
                    added_this_round = True
                    if len(all_actions) >= max_actions:
                        break
            if not added_this_round:
                break
            idx += 1

        if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
            print(f"[DEBUG] Total actions after round-robin merge: {len(all_actions)} (from {len(filtered_batch_results)} batches)")
    
        return {
            "ok": len(all_actions) > 0,
            "actions": all_actions,
            "issues": all_issues,
            "raw_text": "\n\n".join(all_raw_texts),
            "parse_mode": "lines:batched",
            "used_fixer": any_fixer_used,
        }

    def _generate_news_actions_lines_single_batch(
        self,
        *,
        tickers: List[str],
        news_items: List[Dict[str, Any]],
        max_actions: int = 8,
    ) -> Dict[str, Any]:
        """
        Original single-call logic, now operating on ONE batch of tickers/items.
        """
        tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
        max_actions = max(1, min(int(max_actions), 12))

        if max_actions <= 0:
            return {"ok": True, "actions": [], "issues": [], "raw_text": "", "used_fixer": False}

        # compact news items for grounding (LLM sees only these)
        compact_items = []
        for it in (news_items or []):
            if not isinstance(it, dict):
                continue
            compact_items.append({
                "evidence_id": it.get("evidence_id"),
                "ticker": str(it.get("ticker") or "").upper().strip(),
                "date": it.get("date"),
                "source": it.get("source") or it.get("provider"),
                "headline": it.get("headline"),
            })

        # ✅ FIX: allowed_ids must come from compact_items in THIS batch only
        allowed_ids = sorted(
            {str(it.get("evidence_id")).strip() for it in compact_items if it.get("evidence_id")}
        )
        allowed_set = set(allowed_ids)
        allowed_ids_block = "\n".join(allowed_ids)

        system = (
            "CRITICAL: You are a MACHINE that outputs ONLY structured action lines.\n"
            "You are NOT a chat assistant. Do NOT summarize news. Do NOT analyze themes.\n"
            "Do NOT write any prose, headers, or bullet points outside the ACTION BLOCK.\n"
            "If you write anything other than the ACTION BLOCK, your output is REJECTED.\n"
            "\n"
            "You are a portfolio NEWS actions generator.\n"
            "You may include a brief summary, BUT you MUST include exactly one ACTION BLOCK.\n"
            "Only lines inside the ACTION BLOCK will be parsed.\n"
            "\n"
            "ACTION BLOCK format (MANDATORY):\n"
            "BEGIN_ACTIONS\n"
            "<one action line per row OR a single line: NO_ACTIONS>\n"
            "END_ACTIONS\n"
            "\n"
            "CRITICAL RULES (must follow):\n"
            "- Inside the ACTION BLOCK: output ONLY action lines OR NO_ACTIONS.\n"
            "- No markdown, no bullets, no headers inside the block.\n"
            "- evidence_ids MUST be copied EXACTLY from the Allowed evidence_ids list below.\n"
            "- NEVER output placeholders like EID1, EID2, ID1, NEWS1, etc.\n"
            "- Use 1 or 2 evidence_ids per line (comma-separated). Not more.\n"
            "- REASON must be a real, specific sentence explaining the action using the evidence headline. NEVER write the literal word REASON.\n"
            "- If you cannot copy real evidence_ids, output NO_ACTIONS.\n"
            "\n"
            "Line formats (one per line):\n"
            "exclude_ticker|TICKER|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "reduce_exposure|TICKER|low|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "reduce_exposure|TICKER|medium|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "reduce_exposure|TICKER|high|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "set_w_max|0.30|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON   (value must be 0.05-0.50)\n"
            "shift_objective|minvar|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "shift_objective|maxsharpe|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "hedge|HEDGE_HINT|<EVIDENCE_ID_1>,<EVIDENCE_ID_2>|REASON\n"
            "\n"
            "Example of a CORRECT line (note REASON is replaced with a real sentence):\n"
            "reduce_exposure|NVDA|medium|NVDA_ab12cd34|Recent earnings guidance cut signals near-term downside risk.\n"
            "\n"
            "Allowed evidence_ids (one per line):\n"
            f"{allowed_ids_block}\n"
        )

        user = json.dumps(
            {"tickers": tickers_u, "news_items": compact_items},
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()
        used_fixer = False

        if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
            print("\n===== DEBUG: news_actions_lines (single batch) =====")
            print("tickers:", tickers_u)
            print("allowed_set_count:", len(allowed_set))
            print("--- raw actions head ---")
            print((raw or "")[:1200])
            print("====================================\n")

        parsed = self._parse_actions_lines(
            raw,
            tickers=tickers_u,
            allowed_evidence_ids=allowed_set,
            max_actions=max_actions,
        )

        need_fix = (
            (not (parsed.get("actions") or [])) or
            any("too_few_fields" in str(x) for x in (parsed.get("issues") or [])) or
            any("bad_type" in str(x) for x in (parsed.get("issues") or [])) or
            any("missing_or_invalid_eids" in str(x) for x in (parsed.get("issues") or []))
            or any("exclude_bad_ticker" in str(x) for x in (parsed.get("issues") or []))
            or any("placeholder_reason" in str(x) for x in (parsed.get("issues") or []))
        )

        if need_fix:
            used_fixer = True

            # ✅ NEW: only give the fixer the headlines it actually needs —
            # i.e. evidence_ids referenced in its own broken output — not the
            # full batch. This keeps the fixer's context small (so it doesn't
            # regress into the same "too much context -> goes off-script"
            # failure mode as the main call) while still grounding it in
            # real headlines instead of guessing from ticker names alone.
            referenced_eids: set = set()
            for ln in (raw or "").splitlines():
                parts = [p.strip() for p in ln.split("|")]
                if len(parts) >= 4:
                    eid_field = parts[-2]
                    for e in eid_field.split(","):
                        e = e.strip()
                        if e in allowed_set:
                            referenced_eids.add(e)

            fixer_news_items = [
                it for it in compact_items
                if it.get("evidence_id") in referenced_eids
            ]

            if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
                print(f"[DEBUG] Fixer context: referenced_eids={len(referenced_eids)} fixer_news_items={len(fixer_news_items)} (vs full batch={len(compact_items)})")

            fix_system = (
                "CRITICAL: You are a MACHINE. You do NOT summarize, analyze, or explain news.\n"
                "Any text outside BEGIN_ACTIONS/END_ACTIONS is INVALID and REJECTED.\n"
                "\n"
                "You are a STRICT parser-compatible action-line generator.\n"
                "Return ONLY the ACTION BLOCK. No prose. No explanation. No markdown.\n"
                "\n"
                "Your entire output MUST be exactly one of these two forms:\n"
                "\n"
                "BEGIN_ACTIONS\n"
                "NO_ACTIONS\n"
                "END_ACTIONS\n"
                "\n"
                "OR:\n"
                "\n"
                "BEGIN_ACTIONS\n"
                "reduce_exposure|TICKER|medium|EVIDENCE_ID|short reason\n"
                "END_ACTIONS\n"
                "\n"
                "Allowed action line formats:\n"
                "exclude_ticker|TICKER|EVIDENCE_ID|REASON\n"
                "reduce_exposure|TICKER|low|EVIDENCE_ID|REASON\n"
                "reduce_exposure|TICKER|medium|EVIDENCE_ID|REASON\n"
                "reduce_exposure|TICKER|high|EVIDENCE_ID|REASON\n"
                "set_w_max|0.30|EVIDENCE_ID|REASON\n"
                "shift_objective|minvar|EVIDENCE_ID|REASON\n"
                "shift_objective|maxsharpe|EVIDENCE_ID|REASON\n"
                "hedge|HEDGE_HINT|EVIDENCE_ID|REASON\n"
                "\n"
                "Rules:\n"
                "- Use ONLY allowed tickers from allowed_tickers.\n"
                "- Use ONLY evidence_ids from allowed_evidence_ids.\n"
                "- Copy evidence_ids EXACTLY.\n"
                "- Use 1 evidence_id per action line.\n"
                "- CRITICAL: The word 'REASON' must NEVER appear in your output, not even once.\n"
                "- The broken_output_DO_NOT_COPY_REASON_FIELD you are fixing likely contains the literal placeholder word 'REASON' instead of a real sentence.\n"
                "- For EVERY line, write a NEW specific sentence (8+ words) describing what the evidence headline says and why it implies risk or opportunity.\n"
                "- The news_items field contains ONLY the real headlines relevant to the lines you are fixing. You MUST base your reason on the actual headline text for that evidence_id, not on the ticker name alone.\n"
                "- Do NOT invent facts, numbers, or events that are not present in the headline.\n"
                "- WRONG example: reduce_exposure|NVDA|medium|NVDA_ab12cd34|REASON\n"
                "- CORRECT example: reduce_exposure|NVDA|medium|NVDA_ab12cd34|Recent earnings guidance cut signals near-term downside risk for chip demand.\n"
                "- Do NOT write EID1, ID1, NEWS1, placeholders, headlines, dates, or ticker names in the evidence field.\n"
                "- Do NOT write numbered lists.\n"
                "- Do NOT write sentences outside BEGIN_ACTIONS and END_ACTIONS.\n"
                "- If no valid action can be produced, output NO_ACTIONS inside the block.\n"
            )
            fix_user = json.dumps(
                {
                    "allowed_tickers": tickers_u,
                    "max_actions": max_actions,
                    "news_items": fixer_news_items,
                    "broken_output_DO_NOT_COPY_REASON_FIELD": raw,
                    "parser_issues": list(parsed.get("issues") or [])[:60],
                    "instruction": "Rewrite every line with a brand new specific reason sentence based on the actual headline text in news_items for that evidence_id. Do not copy the word REASON from broken_output_DO_NOT_COPY_REASON_FIELD. Do not invent facts not present in the headline.",
                },
                ensure_ascii=False,
            )
            raw2 = (self.chat(system=fix_system, user=fix_user) or "").strip()

            if os.getenv("LLM_DEBUG_NEWS_ACTIONS_LINES", "0") == "1":
                print("\n===== DEBUG: news_actions_lines FIXER raw2 =====")
                print((raw2 or "")[:1200])
                print("================================================\n")

            if raw2.strip() == "NO_ACTIONS":
                parsed2 = {"ok": True, "actions": [], "issues": ["no_actions_returned"]}
            else:
                parsed2 = self._parse_actions_lines(
                    raw2,
                    tickers=tickers_u,
                    allowed_evidence_ids=allowed_set,
                    max_actions=max_actions,
                )

            if parsed2.get("actions"):
                return {
                    "ok": bool(parsed2.get("ok")),
                    "actions": parsed2.get("actions") or [],
                    "issues": parsed2.get("issues") or [],
                    "raw_text": raw2,
                    "used_fixer": used_fixer,
                }
            return {
                "ok": bool(parsed.get("ok")),
                "actions": parsed.get("actions") or [],
                "issues": parsed.get("issues") or [],
                "raw_text": raw,
                "used_fixer": used_fixer,
            }

        return {
            "ok": bool(parsed.get("ok")),
            "actions": parsed.get("actions") or [],
            "issues": parsed.get("issues") or [],
            "raw_text": raw,
            "used_fixer": used_fixer,
        }


    # =========================================================
    # ✅ NEW: News Actions (LLM) + Verifier (LLM) + Deterministic Cleaner
    # =========================================================
    def _verify_news_actions_json(
        self,
        j: Dict[str, Any],
        tickers: List[str],
        *,
        allowed_evidence_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic cleaner for actions JSON:
        Output shape:
          {"ok": bool, "issues": [...], "cleaned": {"actions": [...]}}
        ✅ Additionally filters evidence_ids against allowed_evidence_ids (if provided).
        """
        issues: List[str] = []
        allowed = set(str(t).upper().strip() for t in (tickers or []))

        cleaned_actions: List[Dict[str, Any]] = []

        if not isinstance(j, dict):
            return {"ok": False, "issues": ["actions_not_dict"], "cleaned": {"actions": []}}

        actions = j.get("actions")
        if not isinstance(actions, list):
            return {"ok": False, "issues": ["actions_missing_or_not_list"], "cleaned": {"actions": []}}

        for a in actions:
            if not isinstance(a, dict):
                issues.append("action_item_not_dict")
                continue

            t = str(a.get("type") or "").strip()
            if t not in _ALLOWED_NEWS_ACTION_TYPES:
                issues.append(f"unsupported_action_type:{t}")
                continue

            out: Dict[str, Any] = {"type": t, "reason": str(a.get("reason") or "").strip()}

            out["evidence"] = self._clean_evidence_list(a.get("evidence"))
    
            ev = a.get("evidence_ids")
            ev_ids: List[str] = []
            if isinstance(ev, list):
                tmp = [str(x).strip() for x in ev if str(x).strip()]
                if allowed_evidence_ids is not None:
                    tmp = [x for x in tmp if x in allowed_evidence_ids]
                ev_ids = tmp[:2]

            out["evidence_ids"] = ev_ids

            # ✅ require evidence_ids (no ungrounded actions)
            if not out["evidence_ids"]:
                issues.append("missing_evidence_ids")
                continue



            if t == "exclude_ticker":
                ticker = str(a.get("ticker") or "").upper().strip()
                if not ticker or ticker not in allowed:
                    issues.append(f"exclude_outside_universe:{ticker}")
                    continue
                out["ticker"] = ticker

            elif t == "set_w_max":
                try:
                    v = float(a.get("value"))
                except Exception:
                    issues.append("w_max_not_float")
                    continue
                if not (0.05 <= v <= 0.50):
                    issues.append(f"w_max_out_of_range:{v}")
                    continue
                out["value"] = v

            elif t == "shift_objective":
                to = str(a.get("to") or "").lower().strip()
                if to not in ("minvar", "maxsharpe"):
                    issues.append(f"invalid_shift_objective:{to}")
                    continue
                out["to"] = to

            elif t == "reduce_exposure":
                ticker = str(a.get("ticker") or "").upper().strip()
                if not ticker or ticker not in allowed:
                    issues.append(f"reduce_exposure_outside_universe:{ticker}")
                    continue
                intensity = str(a.get("intensity") or "medium").lower().strip()
                if intensity not in ("low", "medium", "high"):
                    intensity = "medium"
                out["ticker"] = ticker
                out["intensity"] = intensity

            elif t == "hedge":
                out["hedge_hint"] = str(a.get("hedge_hint") or "").strip()

            cleaned_actions.append(out)

        ok = len(issues) == 0
        return {"ok": ok, "issues": issues, "cleaned": {"actions": cleaned_actions}}

    def generate_news_actions(
        self,
        *,
        tickers: List[str],
        snapshot: Optional[Dict[str, Any]] = None,
        risk_json: Optional[Dict[str, Any]] = None,
        max_actions: int = 8,
        news_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        LLM proposes actionable portfolio changes based on news.
        Returns:
          { ok, actions, issues, raw_text, parse_mode }
        """
        tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]
        max_actions = max(1, min(int(max_actions), 12))
        allowed_ids = sorted(
            {str(it.get("evidence_id")).strip() for it in (news_items or []) if isinstance(it, dict) and it.get("evidence_id")}
        )
        # prompt çok uzamasın diye (opsiyonel)
        allowed_ids = allowed_ids[:120]
        allowed_ids_block = "\n".join(allowed_ids)
        system = (
            "You are a portfolio NEWS actions generator.\n"
            "Your goal: propose practical portfolio adjustment actions driven by recent news.\n"
            "Return ONLY valid JSON (no markdown, no extra text).\n"
            "Do NOT invent news. Use ONLY snapshot + risk_json + news_items.\n"
            "\n"
            "Allowed action types:\n"
            "- exclude_ticker: {type, ticker, reason, evidence_ids, evidence}\n"
            "- set_w_max: {type, value, reason, evidence_ids, evidence}\n"
            "- shift_objective: {type, to, reason, evidence_ids, evidence}\n"
            "- reduce_exposure: {type, ticker, intensity, reason, evidence_ids, evidence}\n"
            "- hedge: {type, hedge_hint, reason, evidence_ids, evidence}\n"
            "\n"
            "evidence format:\n"
            '- evidence is a list of 0-3 objects: {"headline": string, "date": "YYYY-MM-DD"|null, "source": string|null, "url": string|null}\n'
            "- If you are not sure, you may return evidence: [] (empty list).\n"

            "\n"
            "Hard rules:\n"
            f"- ticker must be one of: {tickers_u}\n"
            f"- return at most {max_actions} actions\n"
            "- Keep reasons short and specific.\n"
            "- Each action MUST include evidence_ids (1-2 ids).\n"
            f"- Allowed evidence_ids (one per line):\n{allowed_ids_block}\n"
            "- If you cannot support an action with evidence_ids, DO NOT output that action.\n"
            "Output MUST start with '{' and end with '}'.\n"
            "Do NOT include any text outside JSON.\n"
            "evidence_ids must match allowed_evidence_ids EXACTLY (copy-paste).\n"
            "Never output empty evidence_ids.\n"
            "\n"
            "Schema:\n"
            "{\n"
            '  "actions": [\n'
            '    {"type":"reduce_exposure","ticker":"NVDA","intensity":"medium","reason":"...","evidence_ids":["NVDA_01"],"evidence":[...]},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
        )

        compact_items = []
        for it in (news_items or []):
            if not isinstance(it, dict):
                continue
            compact_items.append({
                "evidence_id": it.get("evidence_id"),
                "ticker": it.get("ticker"),
                "date": it.get("date"),
                "source": it.get("source") or it.get("provider"),
                "headline": it.get("headline"),
                "url": it.get("url"),
            })

        user = json.dumps(
            {
                "tickers": tickers_u,
                "snapshot": snapshot or {},
                "risk_json": risk_json or {},
                "news_items": compact_items,
            },
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()
        j, parse_mode = self._parse_json_best_effort(raw)

        if j is None:

            return {
                "ok": False,
                "actions": [],
                "issues": [f"actions_json_parse_failed(mode={parse_mode})"],
                "raw_text": raw,
                "parse_mode": parse_mode,
            }

        # ✅ Build allowed evidence set from snapshot_text + risk_json
        snap_text = ""
        if isinstance((snapshot or {}).get("snapshot_text"), str):
            snap_text = (snapshot or {}).get("snapshot_text") or ""

        allowed_set = set(allowed_ids)

        verified = self._verify_news_actions_json(j, tickers_u, allowed_evidence_ids=allowed_set)
        cleaned = (verified.get("cleaned") or {}).get("actions") or []

        # ✅ 2nd pass self-fix if output is unusable
        if (not cleaned) or (verified.get("issues")):
            fix_system = (
                "You are a strict JSON fixer.\n"
                "Return ONLY valid JSON. No markdown. No extra text.\n"
                "You MUST output at least 1 action if possible.\n"
                "Every action MUST include evidence_ids (1-2), and they MUST be chosen from allowed_evidence_ids.\n"
                "Copy evidence_ids EXACTLY. Do not invent IDs.\n"
                "Schema:\n"
                '{ "actions": [ {"type":"reduce_exposure","ticker":"TICKER","intensity":"low|medium|high","reason":"...","evidence_ids":["ID1"],"evidence":[]} ] }\n'
            )

            fix_user = json.dumps(
                {
                    "allowed_tickers": tickers_u,
                    "allowed_evidence_ids": allowed_ids,
                    "news_items": compact_items,
                    "previous_raw_output": raw,
                    "detected_issues": list(verified.get("issues") or []),
                    "max_actions": max_actions,
                },
                ensure_ascii=False,
            )

            raw2 = (self.chat(system=fix_system, user=fix_user) or "").strip()
            j2, parse_mode2 = self._parse_json_best_effort(raw2)
            if j2 is not None:
                verified2 = self._verify_news_actions_json(j2, tickers_u, allowed_evidence_ids=allowed_set)
                cleaned2 = (verified2.get("cleaned") or {}).get("actions") or []
                # if improved, use it
                if cleaned2:
                    return {
                        "ok": bool(verified2.get("ok")),
                        "actions": cleaned2,
                        "issues": list(verified2.get("issues") or []),
                        "raw_text": raw2,
                        "parse_mode": f"fix:{parse_mode2}",
                    }

        # default return
        return {
            "ok": bool(verified.get("ok")),
            "actions": cleaned,
            "issues": list(verified.get("issues") or []),
            "raw_text": raw,
            "parse_mode": parse_mode,
        }

        # =========================================================
    # ✅ NEW: Attach evidence_ids to already-generated actions (2nd pass)
    # =========================================================
    def attach_evidence_ids_to_actions(
        self,
        *,
        actions: List[Dict[str, Any]],
        news_items: List[Dict[str, Any]],
        tickers: List[str],
        max_ids_per_action: int = 2,
    ) -> Dict[str, Any]:
        """
        2nd LLM pass:
        - Takes existing actions (already cleaned/verified)
        - Attaches evidence_ids based on news_items that already contain evidence_id
        - NEVER drops actions; if not sure -> evidence_ids=[]
        """
        tickers_u = [str(t).upper().strip() for t in (tickers or []) if str(t).strip()]

        compact_items: List[Dict[str, Any]] = []
        for it in (news_items or []):
            compact_items.append(
                {
                    "evidence_id": it.get("evidence_id"),
                    "ticker": str(it.get("ticker") or "").upper().strip(),
                    "date": it.get("date"),
                    "source": it.get("source") or it.get("provider"),
                    "headline": it.get("headline"),
                    "url": it.get("url"),
                }
            )

        system = (
            "You are an evidence linker.\n"
            "Attach evidence_ids to each portfolio action using ONLY the provided news_items.\n"
            "Return ONLY valid JSON (no markdown, no extra text).\n"
            "\n"
            "Hard rules:\n"
            f"- Allowed tickers: {tickers_u}\n"
            "- You may use ONLY evidence_id values that appear in news_items.\n"
            f"- For each action, choose 1-{max_ids_per_action} evidence_ids if possible.\n"
            "- If you cannot find strong support, set evidence_ids: []\n"
            "- NEVER delete or drop actions.\n"
            "\n"
            "Schema:\n"
            '{ "actions": [ { ...original action fields..., "evidence_ids": ["TICKER_01"] } ] }\n'
        )

        user = json.dumps(
            {
                "tickers": tickers_u,
                "actions": actions or [],
                "news_items": compact_items,
            },
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()
        j, parse_mode = self._parse_json_best_effort(raw)

        if not isinstance(j, dict) or not isinstance(j.get("actions"), list):
            # fail-safe: keep original actions
            return {"ok": False, "actions": actions or [], "raw_text": raw, "parse_mode": parse_mode}

        # soft sanitize evidence_ids to be list[str]
        out_actions: List[Dict[str, Any]] = []
        allowed_ids = {str(x.get("evidence_id")).strip() for x in compact_items if x.get("evidence_id")}

        for a in j["actions"]:
            if not isinstance(a, dict):
                continue
            ev = a.get("evidence_ids")
            ev_ids: List[str] = []
            if isinstance(ev, list):
                ev_ids = [str(x).strip() for x in ev if str(x).strip() and str(x).strip() in allowed_ids]
                ev_ids = ev_ids[: max(0, int(max_ids_per_action))]

            # ensure field exists even if empty
            a2 = dict(a)
            a2["evidence_ids"] = ev_ids
            out_actions.append(a2)

        return {"ok": True, "actions": out_actions, "raw_text": raw, "parse_mode": parse_mode}

    def verify_news_actions(
        self,
        *,
        actions: List[Dict[str, Any]],
        snapshot: Optional[Dict[str, Any]],
        risk_json: Optional[Dict[str, Any]],
        universe: List[str],
    ) -> Dict[str, Any]:
        """
        LLM verifier rewrites / drops unsafe actions, then deterministic cleaner enforces schema.
        Returns:
        { ok, actions, issues, raw_text, parse_mode, notes }
        """
        tickers_u = [str(t).upper().strip() for t in (universe or []) if str(t).strip()]

        system = (
            "You are a strict verifier for NEWS-driven portfolio actions.\n"
            "Your job:\n"
            "- Remove actions that are not supported by the provided snapshot/risk_json\n"
            "- Remove duplicates and contradictions\n"
            "- Ensure tickers belong to the universe\n"
            "- Keep only allowed action types and valid fields\n"
            "Return ONLY valid JSON.\n"
            "\n"
            "Allowed action types: exclude_ticker, set_w_max, shift_objective, reduce_exposure, hedge\n"
            "Constraints:\n"
            "- set_w_max.value must be between 0.05 and 0.50\n"
            "- shift_objective.to must be 'minvar' or 'maxsharpe'\n"
            "- reduce_exposure.intensity must be low|medium|high\n"
            "\n"
            "Evidence rules (IMPORTANT):\n"
            "- Each action may include an 'evidence' field.\n"
            "- 'evidence' is a list of 0-3 objects.\n"
            '- Evidence object schema: {"headline": string, "date": "YYYY-MM-DD"|null, "source": string|null, "url": string|null}\n'
            "- Evidence must be grounded in snapshot/risk_json; if unsure, set evidence: [] (empty list).\n"
            "\n"
            "Action schemas:\n"
            '- exclude_ticker: {"type":"exclude_ticker","ticker":"TICKER","reason":"...","evidence":[...]} \n'
            '- set_w_max: {"type":"set_w_max","value":0.30,"reason":"...","evidence":[...]} \n'
            '- shift_objective: {"type":"shift_objective","to":"minvar|maxsharpe","reason":"...","evidence":[...]} \n'
            '- reduce_exposure: {"type":"reduce_exposure","ticker":"TICKER","intensity":"low|medium|high","reason":"...","evidence":[...]} \n'
            '- hedge: {"type":"hedge","hedge_hint":"...","reason":"...","evidence":[...]} \n'
            "\n"
            "Schema:\n"
            "{\n"
            '  "ok": true|false,\n'
            '  "notes": string,\n'
            '  "actions": [ {action}, ... ]\n'
            "}\n"
        )
        

        user = json.dumps(
            {
                "universe": tickers_u,
                "snapshot": snapshot or {},
                "risk_json": risk_json or {},
                "actions": actions or [],
            },
            ensure_ascii=False,
        )

        raw = (self.chat(system=system, user=user) or "").strip()
        j, parse_mode = self._parse_json_best_effort(raw)

        if j is None:
            det = self._verify_news_actions_json(
                {"actions": actions or []},
                tickers_u,
                allowed_evidence_ids=None,
            )
            return {
                "ok": bool(det.get("ok")),
                "actions": (det.get("cleaned") or {}).get("actions") or [],
                "issues": [f"verifier_json_parse_failed(mode={parse_mode})"] + list(det.get("issues") or []),
                "raw_text": raw,
                "parse_mode": parse_mode,
                "notes": "LLM verifier parse failed; used deterministic cleaner.",
            }

        # deterministic clean
        det = self._verify_news_actions_json(j, tickers_u, allowed_evidence_ids=None)
        cleaned = (det.get("cleaned") or {}).get("actions") or []

        ok_llm = bool(j.get("ok")) if isinstance(j.get("ok"), bool) else None
        notes = str(j.get("notes") or "").strip()

        issues = list(det.get("issues") or [])
        if ok_llm is False:
            issues.append("llm_verifier_marked_not_ok")

        return {
            "ok": (len(issues) == 0),
            "actions": cleaned,
            "issues": issues,
            "raw_text": raw,
            "parse_mode": parse_mode,
            "notes": notes or "Verifier completed.",
        }


    
    def _repair_to_json(self, *, raw_text: str, schema_hint: str) -> str:
        system = (
            "You are a JSON repair tool.\n"
            "Convert the given text into VALID JSON that matches the schema.\n"
            "Return ONLY JSON. No markdown. No extra text.\n"
            "The output MUST start with '{' and end with '}'.\n"
        )
        user = json.dumps({"schema": schema_hint, "text": raw_text}, ensure_ascii=False)
        return (self.chat(system=system, user=user) or "").strip()
 

 
