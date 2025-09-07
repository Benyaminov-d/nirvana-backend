from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
from enum import Enum
import json as _json
import os as _os
import re as _re
import time as _time

from pydantic import BaseModel  # type: ignore

# Short type alias to keep annotations within line length limits
DialogList = List[Dict[str, Any]]

# RAG integration with pgvector
try:
    from services.rag.pgvector_service import PgVectorRAG
    _rag_service: Optional[PgVectorRAG] = None
    RAG_AVAILABLE = True
except ImportError:
    PgVectorRAG = None  # type: ignore
    _rag_service = None
    RAG_AVAILABLE = False


class AssistantAction(str, Enum):
    MATCHES = "matches"
    CANDIDATES = "candidates"
    SUMMARY = "summary"
    ASK_TOL = "ask_tol"
    CLARIFY = "clarify"
    COMMENT = "comment"
    HELP = "help"
    WEATHER = "weather"


class AssistantIntent(BaseModel):
    action: AssistantAction
    query: Optional[str] = None
    symbol: Optional[str] = None
    loss_tolerance_pct: Optional[float] = None
    assistant_text: Optional[str] = None

    def coerce(self) -> "AssistantIntent":
        try:
            if self.loss_tolerance_pct is not None:
                v = float(self.loss_tolerance_pct)
                if v > 0:
                    v = -abs(v)
                self.loss_tolerance_pct = v
        except Exception:
            self.loss_tolerance_pct = None
        try:
            if self.query is not None:
                self.query = self.query.strip()
        except Exception:
            pass
        try:
            if self.symbol is not None:
                self.symbol = self.symbol.strip().upper()
        except Exception:
            pass
        try:
            if self.assistant_text is not None:
                self.assistant_text = self.assistant_text.strip()
        except Exception:
            pass
        return self


def detect_locale(text: str) -> str:
    s = (text or "").strip()
    try:
        if _re.search(r"[\u0400-\u04FF]", s):
            return "ru"
    except Exception:
        pass
    return "en"


def _extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```") and txt.endswith("```"):
        try:
            txt = txt.strip("`")
            if txt.startswith("json\n"):
                txt = txt[5:]
        except Exception:
            pass
    try:
        return _json.loads(txt)
    except Exception:
        pass
    try:
        m = _re.search(r"\{[\s\S]*\}", txt)
        if m:
            return _json.loads(m.group(0))
    except Exception:
        return None
    return None


def _flatten_message_text(msg: Any) -> str:
    """
    Extract plain text from Assistant API message object.
    Supports 'text' content blocks; ignores images/tools.
    """
    try:
        parts = []
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "text":
                val = getattr(getattr(block, "text", None), "value", None)
                if val:
                    parts.append(str(val))
        return "\n".join(parts).strip()
    except Exception:
        try:
            # fallback: raw .to_dict()
            d = msg.__dict__
            return str(d)
        except Exception:
            return ""


def _fetch_dialog_list(client, thread_id: str, limit: int = 20) -> DialogList:
    try:
        msgs = client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=limit,
        )
        # API usually returns in reverse order - sort by created_at
        data = list(getattr(msgs, "data", []) or [])
        data.sort(key=lambda m: getattr(m, "created_at", 0))
        out: List[Dict[str, Any]] = []
        for m in data:
            role = getattr(m, "role", "assistant")
            text = _flatten_message_text(m)
            out.append(
                {
                    "role": role,
                    "text": text,
                    "created_at": getattr(m, "created_at", None),
                }
            )
        return out
    except Exception:
        return []


def _get_rag_service() -> Optional[PgVectorRAG]:
    """Get or initialize the RAG service."""
    global _rag_service
    if not RAG_AVAILABLE:
        return None
    
    if _rag_service is None:
        try:
            _rag_service = PgVectorRAG()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"RAG service initialization failed: {e}")
            pass
    
    return _rag_service


def _enhance_with_rag_context(user_msg: str, sys_instructions: str) -> str:
    """Enhance system instructions with relevant RAG context if available."""
    if not RAG_AVAILABLE:
        return sys_instructions
    
    try:
        rag_service = _get_rag_service()
        if rag_service is None:
            return sys_instructions
        
        # Search for relevant context
        context_prompt, used_chunks = rag_service.get_context_for_query(
            query=user_msg,
            max_chunks=3,
            max_context_length=2000
        )
        
        # Always add basic compliance guidance for fintech
        base_compliance = """

IMPORTANT - COMPLIANCE GUIDANCE:
You are Satya, Nirvana's AI assistant for a financial services platform. You MUST:
1. Only provide information that aligns with Nirvana's official terms and policies
2. Never make investment recommendations or provide personalized financial advice
3. Always clarify that outputs are impersonal analytics, not advice
4. For policy questions, refer to official documentation or legal@nirvana.bm
5. Never contradict information in Nirvana's official documents"""
        
        if context_prompt:
            # We have relevant context - use it
            enhanced_instructions = f"""{sys_instructions}{base_compliance}

RELEVANT NIRVANA DOCUMENTATION CONTEXT:
{context_prompt}

CRITICAL: Use the above documentation context to provide accurate, compliant responses. Reference official information directly when available."""
        else:
            # No specific context, just add compliance guidance
            enhanced_instructions = f"""{sys_instructions}{base_compliance}

Note: Always ensure responses align with Nirvana's official position. For policy questions, direct users to official documentation."""
        
        return enhanced_instructions
    
    except Exception as e:
        # Log error but don't fail the entire request
        try:
            import logging
            logging.getLogger(__name__).warning(f"RAG context enhancement failed: {e}")
        except Exception:
            pass
        
        # Fallback with basic compliance
        return f"""{sys_instructions}

IMPORTANT: You are Satya for Nirvana financial platform. Ensure responses comply with financial services regulations and official terms."""


def _add_financial_focus_reminder(intent: AssistantIntent, user_msg: str) -> AssistantIntent:
    """Add financial focus reminder to off-topic responses."""
    
    # Define financial keywords
    financial_keywords = [
        'investment', 'portfolio', 'stock', 'bond', 'fund', 'etf', 'risk', 'return',
        'loss', 'tolerance', 'cvar', 'volatility', 'asset', 'allocation', 'diversification',
        'compass', 'score', 'symbol', 'ticker', 'market', 'finance', 'money', 'financial',
        'subscription', 'billing', 'terms', 'eula', 'policy', 'liability', 'refund',
        'cancel', 'nirvana', 'product', 'instrument'
    ]
    
    # Check if the response seems off-topic
    user_msg_lower = user_msg.lower()
    is_financial_query = any(keyword in user_msg_lower for keyword in financial_keywords)
    
    # Actions that are clearly financial
    financial_actions = ['matches', 'candidates', 'summary', 'ask_tol']
    is_financial_action = intent.action in financial_actions
    
    # If it's not financial query and not financial action, add reminder
    if not is_financial_query and not is_financial_action and intent.assistant_text:
        
        # Don't add reminder if it already contains financial focus language
        assistant_text_lower = intent.assistant_text.lower()
        if 'financial' not in assistant_text_lower and 'nirvana' not in assistant_text_lower:
            
            # Add the reminder
            reminder = " At the present time, I operate as a search engine for your queries in finance."
            intent.assistant_text = intent.assistant_text.rstrip('.') + '.' + reminder
    
    return intent


def _assistants_turn(
    user_msg: str,
    sys_instructions: str,
    thread_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], DialogList]:
    """
    Run one turn via Assistants API.
    Returns: (assistant_last_text, thread_id, dialog_list)
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
        assistant_id = _os.getenv("OPENAI_ASSISTANT_ID")
        if not assistant_id:
            return (None, None, [])

        # 1) ensure thread
        th_id = thread_id
        if not th_id:
            th = client.beta.threads.create()
            th_id = getattr(th, "id", None)
            if not th_id:
                return (None, None, [])

        # 2) add user message
        client.beta.threads.messages.create(
            thread_id=th_id,
            role="user",
            content=user_msg,
        )

        # 3) enhance sys_instructions with RAG context
        enhanced_instructions = _enhance_with_rag_context(user_msg, sys_instructions)
        
        # 4) create run with enhanced instructions
        run = client.beta.threads.runs.create(
            thread_id=th_id,
            assistant_id=assistant_id,
            instructions=enhanced_instructions,
        )

        # 4) poll
        run_id = getattr(run, "id", None)
        t0 = _time.time()
        while True:
            cur = client.beta.threads.runs.retrieve(
                thread_id=th_id,
                run_id=run_id,
            )
            status = getattr(cur, "status", "unknown")
            if status in ("completed", "failed", "cancelled", "expired"):
                break
            if (_time.time() - t0) > float(
                _os.getenv("OPENAI_RUN_TIMEOUT_SEC", "30")
            ):
                break
            _time.sleep(0.4)

        # 5) fetch dialog
        dialog = _fetch_dialog_list(client, th_id, limit=20)
        # last assistant text (if any)
        last_ass_text = ""
        for m in reversed(dialog):
            if m["role"] == "assistant":
                last_ass_text = m["text"] or ""
                break

        return (last_ass_text or None, th_id, dialog)
    except Exception:
        return (None, None, [])


def _chat_completions_turn(
    user_msg: str,
    sys_instructions: str,
) -> Optional[str]:
    """
    Fallback to chat.completions (stateless, no thread).
    Returns assistant text or None.
    """
    try:
        from openai import OpenAI  # type: ignore
        cl = OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
        model = _os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Enhance sys_instructions with RAG context
        enhanced_instructions = _enhance_with_rag_context(user_msg, sys_instructions)
        
        resp = cl.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": enhanced_instructions},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )
        return (
            resp.choices[0].message.content or None
        )
    except Exception:
        return None


def call_llm_intent(
    user_msg: str,
    locale: str,
    session_id: str = "default",
    thread_id: Optional[str] = None,
) -> Tuple[Optional[AssistantIntent], Optional[str], List[Dict[str, Any]]]:
    """
    Main entry point.
    Uses Assistants API if OPENAI_ASSISTANT_ID is set, otherwise falls back to
    chat.completions. Returns (intent_or_none, thread_id_or_none, dialog_list).
    """
    # Get system prompt from environment variable with fallback to default
    sys = _os.getenv("ASSISTANT_SYSTEM_PROMPT")
    
    if not sys:
        # Default fallback prompt
        sys_parts: List[str] = [
            "You are Satya — Nirvana's conversational guide and backend "
            "orchestrator for the demo.\n",
            "Persona: calm, precise, impartial, empathetic. You de-jargonise "
            "finance, avoid hype/fear, and gently reduce bias (loss aversion, "
            "framing, present bias). Be brief, trustworthy, and "
            "non-patronising.\n",
            "\n",
            "Output contract: Return STRICT JSON only — no markdown, no prose "
            "outside JSON. Emit only the keys defined below; omit any field you "
            "don't set. Numbers use plain decimals, no % signs.\n",
            "Schema: {action: 'matches'|'candidates'|'summary'|'ask_tol'|",
            "'clarify'|'comment'|'help'|'weather', query?: string, symbol?: ",
            "string, loss_tolerance_pct?: number, assistant_text?: string}.\n",
            "Session model: context-aware (Assistants API thread provides "
            "memory).\n",
            "Context hints: if the user asks for details without naming a "
            "product but prior turns imply one, prefer action='summary' with "
            "that symbol. If the user says 'same tolerance', reuse the last "
            "tolerance.\n",
            "\n",
            "Parsing & normalisation rules:\n",
            "- Tolerance extraction → loss_tolerance_pct is NEGATIVE (e.g., "
            "'10%' → -10; bare '10' → -10; '-12%' → -12).\n",
            "- Qualitative risk cues: 'capital preservation' → ask_tol with "
            "-5/-10/-15; 'moderate' → ask_tol with -10/-15; "
            "'aggressive' → ask_tol with -20/-30.\n",
            "- Product/entity resolution: 'show me details of X'|'now for X'|",
            "'what about X'|ticker (AAPL) → action='summary' with symbol='X'.\n",
            "- Domain hinting: 'financial product'|'fund search'|'bond etf' "
            "switch domain to finance.\n",
            "- Weather: when the user asks about weather, use action='weather' "
            "and put the location into 'query'.\n",
            "\n",
            "Action selection (one only):\n",
            "1) 'summary' — details for a specific product/ticker. Include "
            "symbol. For stock details, provide comprehensive market analysis "
            "with current trading context, recent performance insights, and "
            "relevant market developments. Structure response with overview, "
            "performance analysis, and market positioning. Be factual and "
            "analytical, avoid speculation. Do NOT include CVaR/loss-level "
            "numbers or %; the UI loads CVaR separately.\n",
            "2) 'matches' — when a tolerance is present or strongly implied. "
            "If both tolerance and a product appear, prefer 'summary'.\n",
            "3) 'candidates' — for keyword searches; put raw keywords into ",
            "'query'.\n",
            "4) 'ask_tol' — user wants matches but no tolerance provided; ",
            "suggest 3–4 options.\n",
            "5) 'clarify' — ambiguity (missing ticker/name, empty keyword "
            "request, multiple possible symbols, mixed topics).\n",
            "6) 'comment' — general non-finance Qs; reply in TWO paragraphs:\n",
            "   (1) a succinct, helpful answer to the user's question;\n",
            "   (2) a short pivot: you are a financial mutual-style search "
            "assistant and can show product details or suggest matches by "
            "loss tolerance. Vary phrasing. EN only.\n",
            "7) 'help' — unknown intent.\n",
            "\n",
            "Assistant text (EN only): For 'summary' actions, provide detailed "
            "market analysis and context (≤800 chars). For other actions, be "
            "succinct and concise (≤320 chars). Always neutral tone, no emojis "
            "or advice language. Mention Nirvana Standard on 'matches' briefly.\n",
            "Regulatory posture: no personalised advice/guarantees; if "
            "real-time data is implied, say you lack it and suggest a neutral "
            "next step, then pivot to finance.\n",
            "Quality gates: never invent symbols/data; never add extra JSON "
            "keys; omit null fields.\n",
        ]
        sys = "".join(sys_parts)

    has_assistant = bool(_os.getenv("OPENAI_ASSISTANT_ID"))
    if has_assistant:
        ass_text, th_id, dialog = _assistants_turn(
            user_msg,
            sys_instructions=sys,
            thread_id=thread_id,
        )
        if not ass_text:
            return (None, th_id, dialog)
        obj = _extract_json_obj(ass_text)
        if not isinstance(obj, dict):
            return (None, th_id, dialog)
        
        lt_raw = obj.get("loss_tolerance_pct")
        lt_val = None
        try:
            if lt_raw is not None:
                lt_val = float(lt_raw)
        except Exception:
            lt_val = None
        
        intent = AssistantIntent(
            action=AssistantAction(str(obj.get("action", "help"))),
            query=(
                obj.get("query") if isinstance(obj.get("query"), str) else None
            ),
            symbol=(
                obj.get("symbol")
                if isinstance(obj.get("symbol"), str)
                else None
            ),
            loss_tolerance_pct=lt_val,
            assistant_text=(
                obj.get("assistant_text")
                if isinstance(obj.get("assistant_text"), str)
                else None
            ),
        ).coerce()
        
        # Add financial focus reminder for off-topic responses
        intent = _add_financial_focus_reminder(intent, user_msg)
        
        return (
            intent,
            th_id,
            dialog,
        )

    # fallback to chat.completions
    ass_text = _chat_completions_turn(
        user_msg,
        sys_instructions=sys,
    )
    if not ass_text:
        return (None, None, [])
    obj = _extract_json_obj(ass_text)
    if not isinstance(obj, dict):
        return (None, None, [])
    
    lt_raw = obj.get("loss_tolerance_pct")
    lt_val = None
    try:
        if lt_raw is not None:
            lt_val = float(lt_raw)
    except Exception:
        lt_val = None
    
    intent = AssistantIntent(
        action=AssistantAction(str(obj.get("action", "help"))),
        query=(
            obj.get("query") if isinstance(obj.get("query"), str) else None
        ),
        symbol=(
            obj.get("symbol") if isinstance(obj.get("symbol"), str) else None
        ),
        loss_tolerance_pct=lt_val,
        assistant_text=(
            obj.get("assistant_text")
            if isinstance(obj.get("assistant_text"), str)
            else None
        ),
    ).coerce()
    
    # Add financial focus reminder for off-topic responses
    intent = _add_financial_focus_reminder(intent, user_msg)
    
    return (intent, None, [])


def fallback_intent(user_msg: str) -> AssistantIntent:
    m = (user_msg or "").strip()
    tol = None
    mm = _re.search(r"(-?\d+\.?\d*)\s*%", m) or _re.search(
        r"(\d+\.?\d*)\s*(percent|pct)", m, _re.I
    )
    if mm:
        try:
            tol = -abs(float(mm.group(1)))
        except Exception:
            tol = None
    if tol is None:
        try:
            if m and all(ch in "0123456789.- " for ch in m):
                v = float(m)
                tol = -abs(v)
        except Exception:
            tol = None
    if tol is not None:
        return AssistantIntent(
            action=AssistantAction.MATCHES,
            loss_tolerance_pct=tol,
        ).coerce()
    if m:
        return AssistantIntent(
            action=AssistantAction.CANDIDATES,
            query=m,
        ).coerce()
    return AssistantIntent(action=AssistantAction.HELP)


def get_thread_dialog(thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Public helper to retrieve dialog list from Assistants API for a thread.
    Returns a list of dicts with keys: role, text, created_at.
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
        return _fetch_dialog_list(client, thread_id, limit=limit)
    except Exception:
        return []
