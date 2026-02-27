"""Agent route — natural language industrial AI assistant endpoint.

POST /api/v1/agent/query
    Submit a natural language query; get an agentic answer with tool traces.

GET /api/v1/agent/capabilities
    Discover which tools are available in this deployment.

POST /api/v1/agent/diagnose
    Convenience endpoint: machine_id + sensor readings → full diagnosis.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from haiip.api.deps import CurrentUser
from haiip.core.agent import IndustrialAgent

router = APIRouter()

# Singleton agent — components are None by default (graceful degradation).
# In production, inject real components via dependency injection or
# populate via startup event from haiip/api/main.py.
_agent = IndustrialAgent()


# ── Schemas ───────────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Natural language query")
    machine_id: str | None = Field(None, description="Optional machine ID to scope KB search")
    sensor_readings: dict[str, float] | None = Field(
        None, description="Optional sensor name → float value map"
    )


class ToolCallResponse(BaseModel):
    tool: str
    success: bool
    duration_ms: float
    error: str | None = None


class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: list[dict[str, Any]]
    tool_calls: list[ToolCallResponse]
    session_id: str
    duration_ms: float
    limitations: list[str]
    requires_human_review: bool


class DiagnoseRequest(BaseModel):
    machine_id: str = Field(..., min_length=1, max_length=128)
    sensor_readings: dict[str, float] = Field(..., description="Current sensor readings")
    query: str = Field(
        default="Is this machine operating normally? Should I schedule maintenance?",
        description="Natural language question (defaults to standard diagnostic query)",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/agent/query", response_model=AgentQueryResponse, tags=["agent"])
async def agent_query(
    body: AgentQueryRequest,
    current_user: CurrentUser,
) -> AgentQueryResponse:
    """Submit a natural language query to the industrial AI agent.

    The agent will:
    1. Classify intent and select appropriate tools
    2. Execute tools (anomaly detection, RUL, compliance, KB search)
    3. Synthesise a coherent answer with citations
    4. Flag requires_human_review when confidence is low or RUL is critical

    **EU AI Act Article 14**: All responses include human oversight flags.
    """
    result = _agent.query(
        query=body.query,
        machine_id=body.machine_id,
        sensor_readings=body.sensor_readings,
    )
    return AgentQueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        sources=result.sources,
        tool_calls=[
            ToolCallResponse(
                tool=tc.tool,
                success=tc.success,
                duration_ms=tc.duration_ms,
                error=tc.error,
            )
            for tc in result.tool_calls
        ],
        session_id=result.session_id,
        duration_ms=result.duration_ms,
        limitations=result.limitations,
        requires_human_review=result.requires_human_review,
    )


@router.post("/agent/diagnose", response_model=AgentQueryResponse, tags=["agent"])
async def agent_diagnose(
    body: DiagnoseRequest,
    current_user: CurrentUser,
) -> AgentQueryResponse:
    """Diagnose a specific machine from sensor readings.

    Convenience wrapper that always runs anomaly detection + RUL estimation
    in addition to knowledge base search.
    """
    result = _agent.query(
        query=body.query,
        machine_id=body.machine_id,
        sensor_readings=body.sensor_readings,
    )
    return AgentQueryResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        sources=result.sources,
        tool_calls=[
            ToolCallResponse(
                tool=tc.tool,
                success=tc.success,
                duration_ms=tc.duration_ms,
                error=tc.error,
            )
            for tc in result.tool_calls
        ],
        session_id=result.session_id,
        duration_ms=result.duration_ms,
        limitations=result.limitations,
        requires_human_review=result.requires_human_review,
    )


@router.get("/agent/capabilities", tags=["agent"])
async def agent_capabilities(current_user: CurrentUser) -> dict[str, Any]:
    """List available agent tools and their descriptions.

    Useful for UI discovery and for human operators to understand
    what the agent can and cannot do (EU AI Act Article 13 transparency).
    """
    return _agent.capabilities
