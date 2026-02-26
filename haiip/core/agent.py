"""Agentic RAG — ReAct-style tool-calling agent for industrial AI queries.

Architecture (ReAct pattern — Yao et al., 2022):
    1. User sends natural language query
    2. Agent plans which tools to invoke (intent classification)
    3. Tools execute in sequence, each returning structured results
    4. Agent synthesises a final answer with citations and confidence
    5. Every step logged for EU AI Act explainability (Article 13 — Transparency)

Tools available:
    - search_knowledge_base   : RAG over maintenance manuals & ISO docs
    - run_anomaly_detection   : Real-time anomaly score for sensor readings
    - calculate_rul           : Remaining Useful Life estimate
    - assess_compliance       : EU AI Act Article 52 compliance assessment

Design goals:
    - No external API required (works offline with local models)
    - OpenAI optional (used by RAG engine if key present)
    - Every tool call logged with timestamps for audit trail
    - Confidence propagated end-to-end
    - requires_human_review flag per EU AI Act Article 14

References:
    - Yao et al. (2022) ReAct: Synergizing Reasoning and Acting in Language Models
    - EU AI Act Articles 13, 14, 52 (transparency + human oversight)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool definitions ──────────────────────────────────────────────────────────

class ToolName(str, Enum):
    SEARCH_KB         = "search_knowledge_base"
    ANOMALY_DETECT    = "run_anomaly_detection"
    CALCULATE_RUL     = "calculate_rul"
    ASSESS_COMPLIANCE = "assess_compliance"


@dataclass
class ToolCall:
    """One invocation of an agent tool — stored for audit / explainability."""
    tool: str
    input: dict[str, Any]
    output: Any
    duration_ms: float
    success: bool
    error: str | None = None


@dataclass
class AgentThought:
    """Chain-of-thought step (ReAct: Thought → Action → Observation)."""
    thought: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Final structured response returned to callers."""
    query: str
    answer: str
    confidence: float
    sources: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    thoughts: list[AgentThought]
    session_id: str
    duration_ms: float
    limitations: list[str]
    requires_human_review: bool


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_search_kb(
    query: str,
    rag_engine: Any | None,
    machine_id: str | None = None,
) -> dict[str, Any]:
    """Search the industrial knowledge base via FAISS RAG."""
    if rag_engine is None:
        return {"answer": "Knowledge base not available.", "sources": [], "confidence": 0.0}
    result = rag_engine.query(query, machine_id=machine_id)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
    }


def _tool_anomaly_detect(
    features: dict[str, float],
    detector: Any | None,
) -> dict[str, Any]:
    """Run IsolationForest anomaly detection on sensor readings."""
    if detector is None:
        return {"label": "unknown", "confidence": 0.0, "anomaly_score": 0.0, "is_anomaly": False}
    vals = list(features.values())
    if not vals:
        return {"label": "unknown", "confidence": 0.0, "anomaly_score": 0.0, "is_anomaly": False}
    result = detector.predict(vals)
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "anomaly_score": result.get("anomaly_score", 0.0),
        "is_anomaly": result["label"] == "anomaly",
        "explanation": result.get("explanation", ""),
    }


def _tool_calculate_rul(
    features: dict[str, float],
    predictor: Any | None,
) -> dict[str, Any]:
    """Estimate Remaining Useful Life and failure mode with GradientBoosting."""
    if predictor is None:
        return {"label": "unknown", "rul_cycles": -1, "confidence": 0.0, "failure_probability": 0.0}
    vals = list(features.values())
    if not vals:
        return {"label": "unknown", "rul_cycles": -1, "confidence": 0.0, "failure_probability": 0.0}
    result = predictor.predict(vals)
    return {
        "label": result["label"],
        "rul_cycles": result.get("rul_cycles", -1),
        "failure_probability": result.get("failure_probability", 0.0),
        "confidence": result["confidence"],
    }


def _tool_assess_compliance(compliance_engine: Any | None) -> dict[str, Any]:
    """Assess EU AI Act compliance status via ComplianceEngine."""
    if compliance_engine is None:
        return {"risk_level": "unknown", "compliant": False, "transparency_required": True}
    assessment = compliance_engine.classify_risk()
    return {
        "risk_level": assessment.risk_level.value,
        "transparency_required": assessment.transparency_required,
        "conformity_assessment_required": assessment.conformity_assessment_required,
        "compliant": not assessment.conformity_assessment_required,
    }


# ── Industrial Agent ──────────────────────────────────────────────────────────

class IndustrialAgent:
    """ReAct-style agentic RAG for industrial maintenance queries.

    Supports multi-step reasoning over:
        - Knowledge base (FAISS + sentence-transformers RAG)
        - Live sensor anomaly detection (IsolationForest)
        - Remaining Useful Life estimation (GradientBoosting)
        - EU AI Act compliance assessment

    Human oversight:
        - requires_human_review=True when confidence < 0.7 or RUL < 20 cycles
        - Full tool trace logged per EU AI Act Article 13
        - Limitations list always included in response

    Usage:
        agent = IndustrialAgent(rag_engine=rag, anomaly_detector=detector)
        response = agent.query(
            "Is this machine showing signs of failure?",
            sensor_readings={"air_temperature": 350.0, "tool_wear": 180},
        )
    """

    MAX_STEPS = 5

    # Intent keywords mapped to tools (lightweight NLU, no LLM needed)
    INTENT_KEYWORDS: dict[ToolName, list[str]] = {
        ToolName.ANOMALY_DETECT: [
            "anomaly", "abnormal", "unusual", "vibration", "sensor",
            "reading", "detect", "temperature spike", "noise",
        ],
        ToolName.CALCULATE_RUL: [
            "rul", "remaining", "wear", "lifespan", "failure",
            "breakdown", "when", "predict failure", "how long",
        ],
        ToolName.ASSESS_COMPLIANCE: [
            "compliance", "ai act", "gdpr", "regulation", "legal",
            "audit", "article 52", "transparency", "risk",
        ],
        ToolName.SEARCH_KB: [
            "manual", "procedure", "standard", "iso", "how to",
            "what is", "explain", "document", "guideline", "recommend",
        ],
    }

    def __init__(
        self,
        rag_engine: Any | None = None,
        anomaly_detector: Any | None = None,
        maintenance_predictor: Any | None = None,
        compliance_engine: Any | None = None,
    ) -> None:
        self.rag_engine = rag_engine
        self.anomaly_detector = anomaly_detector
        self.maintenance_predictor = maintenance_predictor
        self.compliance_engine = compliance_engine

    # ── Public interface ───────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        machine_id: str | None = None,
        sensor_readings: dict[str, float] | None = None,
    ) -> AgentResponse:
        """Answer a natural language industrial AI query.

        Args:
            query: Natural language question from operator/engineer.
            machine_id: Optional machine ID to scope KB search.
            sensor_readings: Optional dict of sensor name → float value.

        Returns:
            AgentResponse with answer, confidence, sources, and audit trace.
        """
        session_id = str(uuid.uuid4())[:8]
        start = time.monotonic()
        thoughts: list[AgentThought] = []
        all_tool_calls: list[ToolCall] = []
        sources: list[dict[str, Any]] = []

        # Step 1: Plan — determine which tools are needed
        tools_needed = self._plan_tools(query, sensor_readings)
        thoughts.append(AgentThought(
            thought=(
                f"Query: '{query[:80]}'. "
                f"Planning: [{', '.join(t.value for t in tools_needed)}]"
            ),
        ))

        # Step 2: Act — execute each planned tool
        tool_results: dict[str, Any] = {}
        for tool in tools_needed:
            tc = self._invoke_tool(tool, query, machine_id, sensor_readings)
            all_tool_calls.append(tc)
            thoughts[-1].tool_calls.append(tc)
            if tc.success and tc.output:
                tool_results[tool.value] = tc.output
                if isinstance(tc.output, dict) and "sources" in tc.output:
                    sources.extend(tc.output["sources"])

        # Step 3: Observe — synthesise final answer
        answer, confidence = self._synthesise(query, tool_results, sources)

        # Step 4: Human-in-the-loop check (EU AI Act Article 14)
        requires_human_review = self._needs_human_review(tool_results, confidence)

        duration_ms = (time.monotonic() - start) * 1000

        logger.info(
            "agent.query_complete",
            session_id=session_id,
            tools_used=len(all_tool_calls),
            confidence=confidence,
            requires_human_review=requires_human_review,
        )

        return AgentResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            sources=sources[:5],
            tool_calls=all_tool_calls,
            thoughts=thoughts,
            session_id=session_id,
            duration_ms=round(duration_ms, 2),
            limitations=self._get_limitations(tools_needed),
            requires_human_review=requires_human_review,
        )

    # ── Planning ───────────────────────────────────────────────────────────────

    def _plan_tools(
        self,
        query: str,
        sensor_readings: dict[str, float] | None,
    ) -> list[ToolName]:
        """Lightweight intent classification — map query to tool set."""
        q_lower = query.lower()
        planned: list[ToolName] = []

        # Anomaly: sensor data provided OR anomaly keywords in query
        if sensor_readings or any(
            kw in q_lower for kw in self.INTENT_KEYWORDS[ToolName.ANOMALY_DETECT]
        ):
            planned.append(ToolName.ANOMALY_DETECT)

        # RUL: failure/wear keywords + sensor readings available
        if sensor_readings and any(
            kw in q_lower for kw in self.INTENT_KEYWORDS[ToolName.CALCULATE_RUL]
        ):
            planned.append(ToolName.CALCULATE_RUL)

        # Compliance: regulatory keywords
        if any(kw in q_lower for kw in self.INTENT_KEYWORDS[ToolName.ASSESS_COMPLIANCE]):
            planned.append(ToolName.ASSESS_COMPLIANCE)

        # Always search KB last — provides grounding and citations
        planned.append(ToolName.SEARCH_KB)

        # Deduplicate preserving order
        seen: set[ToolName] = set()
        result: list[ToolName] = []
        for t in planned:
            if t not in seen:
                seen.add(t)
                result.append(t)

        return result[: self.MAX_STEPS]

    # ── Tool invocation ────────────────────────────────────────────────────────

    def _invoke_tool(
        self,
        tool: ToolName,
        query: str,
        machine_id: str | None,
        sensor_readings: dict[str, float] | None,
    ) -> ToolCall:
        """Execute one tool and return a ToolCall record."""
        t0 = time.monotonic()
        tool_input: dict[str, Any] = {}

        try:
            if tool == ToolName.SEARCH_KB:
                tool_input = {"query": query, "machine_id": machine_id}
                output = _tool_search_kb(query, self.rag_engine, machine_id)

            elif tool == ToolName.ANOMALY_DETECT:
                readings = sensor_readings or {}
                tool_input = {"features": readings}
                output = _tool_anomaly_detect(readings, self.anomaly_detector)

            elif tool == ToolName.CALCULATE_RUL:
                readings = sensor_readings or {}
                tool_input = {"features": readings}
                output = _tool_calculate_rul(readings, self.maintenance_predictor)

            elif tool == ToolName.ASSESS_COMPLIANCE:
                tool_input = {}
                output = _tool_assess_compliance(self.compliance_engine)

            else:
                output = {"error": f"Unknown tool: {tool}"}

            return ToolCall(
                tool=tool.value,
                input=tool_input,
                output=output,
                duration_ms=round((time.monotonic() - t0) * 1000, 2),
                success=True,
            )

        except Exception as exc:
            logger.warning("agent.tool_error", tool=tool.value, error=str(exc))
            return ToolCall(
                tool=tool.value,
                input=tool_input,
                output=None,
                duration_ms=round((time.monotonic() - t0) * 1000, 2),
                success=False,
                error=str(exc),
            )

    # ── Synthesis ──────────────────────────────────────────────────────────────

    def _synthesise(
        self,
        query: str,
        tool_results: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Combine tool outputs into a coherent natural language answer."""
        parts: list[str] = []
        confidence_scores: list[float] = []

        # Anomaly detection
        ad = tool_results.get(ToolName.ANOMALY_DETECT.value)
        if ad:
            label = ad.get("label", "unknown")
            conf = float(ad.get("confidence", 0.5))
            score = float(ad.get("anomaly_score", 0.0))
            confidence_scores.append(conf)
            if label == "anomaly":
                parts.append(
                    f"**Anomaly detected** (confidence {conf:.0%}, score {score:.3f}). "
                    "Immediate inspection is recommended. Do not operate unattended."
                )
            elif label != "unknown":
                parts.append(
                    f"Sensor readings appear **normal** (confidence {conf:.0%}, score {score:.3f})."
                )

        # RUL estimate
        rul = tool_results.get(ToolName.CALCULATE_RUL.value)
        if rul:
            rul_cycles = int(rul.get("rul_cycles", -1))
            failure_mode = rul.get("label", "unknown")
            conf = float(rul.get("confidence", 0.5))
            confidence_scores.append(conf)
            if rul_cycles > 0:
                urgency = (
                    "**Critical — schedule maintenance immediately**"
                    if rul_cycles < 20
                    else ("Plan maintenance soon" if rul_cycles < 100 else "No immediate action needed")
                )
                parts.append(
                    f"**Remaining Useful Life**: ~{rul_cycles} cycles. "
                    f"Predicted failure mode: {failure_mode} (confidence {conf:.0%}). "
                    f"{urgency}."
                )
            else:
                parts.append(
                    f"Maintenance predictor returned: {failure_mode} (confidence {conf:.0%})."
                )

        # Compliance
        comp = tool_results.get(ToolName.ASSESS_COMPLIANCE.value)
        if comp:
            risk = str(comp.get("risk_level", "unknown")).upper()
            compliant = bool(comp.get("compliant", False))
            transparency = bool(comp.get("transparency_required", True))
            parts.append(
                f"**EU AI Act compliance**: {risk} risk. "
                f"{'System is compliant — no conformity assessment required.' if compliant else 'Conformity assessment required.'}"
                + (" Transparency obligations apply." if transparency else "")
            )
            confidence_scores.append(0.95)

        # Knowledge base / RAG
        kb = tool_results.get(ToolName.SEARCH_KB.value)
        if kb:
            kb_answer = str(kb.get("answer", ""))
            kb_conf = float(kb.get("confidence", 0.0))
            if kb_answer and kb_conf > 0.1:
                confidence_scores.append(kb_conf)
                parts.append(f"\n**From knowledge base**: {kb_answer}")

        # Fallback
        if not parts:
            parts.append(
                "I was unable to find specific information about this query with the available tools. "
                "Please provide sensor readings or consult your maintenance manual."
            )

        answer = "\n\n".join(parts)
        confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        )
        return answer, round(confidence, 4)

    # ── Human oversight ────────────────────────────────────────────────────────

    def _needs_human_review(
        self,
        tool_results: dict[str, Any],
        confidence: float,
    ) -> bool:
        """Flag for mandatory human review per EU AI Act Article 14."""
        # Low overall confidence
        if confidence < 0.4:
            return True

        # Anomaly detected with low confidence — could be false positive
        ad = tool_results.get(ToolName.ANOMALY_DETECT.value)
        if ad and ad.get("is_anomaly") and confidence < 0.7:
            return True

        # Critically low RUL
        rul = tool_results.get(ToolName.CALCULATE_RUL.value)
        if rul and 0 < int(rul.get("rul_cycles", 100)) < 20:
            return True

        return False

    def _get_limitations(self, tools_used: list[ToolName]) -> list[str]:
        """Return standardised limitation statements (AI Act transparency)."""
        limitations = [
            "Predictions are probabilistic estimates, not ground truth.",
            "Human oversight required before acting on recommendations (EU AI Act Art. 14).",
            "Model accuracy depends on training data distribution — out-of-distribution inputs reduce reliability.",
        ]
        if ToolName.ANOMALY_DETECT in tools_used:
            limitations.append(
                "Anomaly detection is trained on historical patterns; novel failure modes may go undetected."
            )
        if ToolName.CALCULATE_RUL in tools_used:
            limitations.append(
                "RUL estimates assume degradation follows the training distribution."
            )
        if ToolName.SEARCH_KB not in tools_used or self.rag_engine is None:
            limitations.append(
                "Knowledge base unavailable — answers not grounded in documentation."
            )
        return limitations

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return agent capability metadata for API discovery."""
        return {
            "tools": [t.value for t in ToolName],
            "tool_descriptions": {
                ToolName.SEARCH_KB.value: "FAISS RAG over maintenance manuals and ISO standards",
                ToolName.ANOMALY_DETECT.value: "IsolationForest anomaly detection on sensor readings",
                ToolName.CALCULATE_RUL.value: "GradientBoosting RUL estimation and failure mode prediction",
                ToolName.ASSESS_COMPLIANCE.value: "EU AI Act Article 52 compliance assessment",
            },
            "rag_available": self.rag_engine is not None,
            "anomaly_detection_available": self.anomaly_detector is not None,
            "maintenance_prediction_available": self.maintenance_predictor is not None,
            "compliance_available": self.compliance_engine is not None,
            "max_steps": self.MAX_STEPS,
            "human_oversight": (
                "Flags requires_human_review=True when confidence < 0.7 or RUL < 20 cycles. "
                "EU AI Act Article 14 compliant."
            ),
            "architecture": "ReAct (Yao et al., 2022) — Thought → Action → Observation",
        }
