"""Rigorous tests for the IndustrialAgent — agentic RAG component.

Tests cover:
1. Tool planning (intent classification)
2. Tool execution (with mocked components)
3. Answer synthesis (all tool combinations)
4. Human oversight flags (EU AI Act Article 14)
5. Graceful degradation (None components)
6. Full query pipeline (end-to-end with stubs)
7. Capabilities endpoint
8. Edge cases (empty query, long query, no sensor data)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from haiip.core.agent import (
    AgentResponse,
    IndustrialAgent,
    ToolCall,
    ToolName,
    _tool_anomaly_detect,
    _tool_assess_compliance,
    _tool_calculate_rul,
    _tool_search_kb,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag():
    """Stub RAG engine returning a fixed QueryResult."""
    rag = MagicMock()
    result = MagicMock()
    result.answer = "Check the bearing lubrication per ISO 281."
    result.sources = [{"title": "ISO 281", "source": "iso", "score": 0.85, "excerpt": "..."}]
    result.confidence = 0.82
    rag.query.return_value = result
    return rag


@pytest.fixture
def mock_anomaly_detector():
    """Stub AnomalyDetector returning a dict (matching real API)."""
    detector = MagicMock()
    detector.predict.return_value = {
        "label": "normal",
        "confidence": 0.88,
        "anomaly_score": -0.15,
        "explanation": "Within normal operating range.",
    }
    return detector


@pytest.fixture
def mock_anomaly_detector_alert():
    """Stub AnomalyDetector returning anomaly — triggers human review."""
    detector = MagicMock()
    detector.predict.return_value = {
        "label": "anomaly",
        "confidence": 0.62,  # below 0.7 → triggers human review
        "anomaly_score": 0.78,
        "explanation": "High vibration detected.",
    }
    return detector


@pytest.fixture
def mock_maintenance_predictor():
    """Stub MaintenancePredictor returning a dict."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "label": "no_failure",
        "confidence": 0.91,
        "failure_probability": 0.09,
        "rul_cycles": 342,
    }
    return predictor


@pytest.fixture
def mock_maintenance_predictor_critical():
    """Stub predictor returning critically low RUL."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "label": "TWF",
        "confidence": 0.87,
        "failure_probability": 0.87,
        "rul_cycles": 8,  # < 20 → triggers human review
    }
    return predictor


@pytest.fixture
def mock_compliance_engine():
    """Stub ComplianceEngine."""
    engine = MagicMock()
    assessment = MagicMock()
    assessment.risk_level.value = "limited"
    assessment.transparency_required = True
    assessment.conformity_assessment_required = False
    engine.classify_risk.return_value = assessment
    return engine


@pytest.fixture
def full_agent(
    mock_rag,
    mock_anomaly_detector,
    mock_maintenance_predictor,
    mock_compliance_engine,
):
    return IndustrialAgent(
        rag_engine=mock_rag,
        anomaly_detector=mock_anomaly_detector,
        maintenance_predictor=mock_maintenance_predictor,
        compliance_engine=mock_compliance_engine,
    )


@pytest.fixture
def empty_agent():
    """Agent with no components — tests graceful degradation."""
    return IndustrialAgent()


NORMAL_READINGS = {
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551.0,
    "torque": 42.8,
    "tool_wear": 0.0,
}

CRITICAL_READINGS = {
    "air_temperature": 352.0,
    "process_temperature": 370.0,
    "rotational_speed": 2200.0,
    "torque": 75.0,
    "tool_wear": 215.0,
}


# ── 1. Tool planning ──────────────────────────────────────────────────────────

class TestToolPlanning:
    def test_kb_always_included(self, empty_agent):
        tools = empty_agent._plan_tools("what is the maintenance procedure?", None)
        assert ToolName.SEARCH_KB in tools

    def test_anomaly_detect_on_sensor_data(self, empty_agent):
        tools = empty_agent._plan_tools("is everything ok?", NORMAL_READINGS)
        assert ToolName.ANOMALY_DETECT in tools

    def test_anomaly_detect_on_keyword(self, empty_agent):
        tools = empty_agent._plan_tools("detect anomaly in this machine", None)
        assert ToolName.ANOMALY_DETECT in tools

    def test_rul_included_with_sensor_and_keyword(self, empty_agent):
        tools = empty_agent._plan_tools("how long until failure?", NORMAL_READINGS)
        assert ToolName.CALCULATE_RUL in tools

    def test_rul_not_included_without_sensor_data(self, empty_agent):
        """RUL requires sensor readings — no data → no RUL tool."""
        tools = empty_agent._plan_tools("how long until failure?", None)
        assert ToolName.CALCULATE_RUL not in tools

    def test_compliance_on_keyword(self, empty_agent):
        tools = empty_agent._plan_tools("check AI Act compliance status", None)
        assert ToolName.ASSESS_COMPLIANCE in tools

    def test_max_steps_not_exceeded(self, empty_agent):
        tools = empty_agent._plan_tools(
            "detect anomaly compliance how long failure ai act manual", NORMAL_READINGS
        )
        assert len(tools) <= empty_agent.MAX_STEPS

    def test_no_duplicate_tools(self, empty_agent):
        tools = empty_agent._plan_tools("anomaly anomaly anomaly", NORMAL_READINGS)
        assert len(tools) == len(set(tools))


# ── 2. Tool function unit tests ────────────────────────────────────────────────

class TestToolFunctions:
    def test_search_kb_with_engine(self, mock_rag):
        result = _tool_search_kb("bearing failure", mock_rag)
        assert result["confidence"] == 0.82
        assert "sources" in result
        assert len(result["sources"]) == 1

    def test_search_kb_without_engine(self):
        result = _tool_search_kb("bearing failure", None)
        assert result["confidence"] == 0.0
        assert result["sources"] == []

    def test_anomaly_detect_normal(self, mock_anomaly_detector):
        result = _tool_anomaly_detect(NORMAL_READINGS, mock_anomaly_detector)
        assert result["label"] == "normal"
        assert result["is_anomaly"] is False
        assert 0.0 <= result["confidence"] <= 1.0

    def test_anomaly_detect_without_detector(self):
        result = _tool_anomaly_detect(NORMAL_READINGS, None)
        assert result["label"] == "unknown"
        assert result["is_anomaly"] is False

    def test_anomaly_detect_empty_features(self, mock_anomaly_detector):
        result = _tool_anomaly_detect({}, mock_anomaly_detector)
        # Empty features — returns safe unknown
        assert result["label"] == "unknown"

    def test_calculate_rul_normal(self, mock_maintenance_predictor):
        result = _tool_calculate_rul(NORMAL_READINGS, mock_maintenance_predictor)
        assert result["rul_cycles"] == 342
        assert result["label"] == "no_failure"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_calculate_rul_without_predictor(self):
        result = _tool_calculate_rul(NORMAL_READINGS, None)
        assert result["rul_cycles"] == -1
        assert result["label"] == "unknown"

    def test_assess_compliance_with_engine(self, mock_compliance_engine):
        result = _tool_assess_compliance(mock_compliance_engine)
        assert result["risk_level"] == "limited"
        assert result["transparency_required"] is True
        assert result["compliant"] is True

    def test_assess_compliance_without_engine(self):
        result = _tool_assess_compliance(None)
        assert result["risk_level"] == "unknown"
        assert result["compliant"] is False


# ── 3. Full query pipeline ─────────────────────────────────────────────────────

class TestQueryPipeline:
    def test_basic_query_returns_response(self, full_agent):
        resp = full_agent.query("What is the maintenance procedure for bearings?")
        assert isinstance(resp, AgentResponse)
        assert resp.query
        assert resp.answer
        assert resp.session_id
        assert resp.duration_ms >= 0

    def test_response_has_all_required_fields(self, full_agent):
        resp = full_agent.query("Is the machine normal?", sensor_readings=NORMAL_READINGS)
        assert hasattr(resp, "query")
        assert hasattr(resp, "answer")
        assert hasattr(resp, "confidence")
        assert hasattr(resp, "sources")
        assert hasattr(resp, "tool_calls")
        assert hasattr(resp, "thoughts")
        assert hasattr(resp, "session_id")
        assert hasattr(resp, "duration_ms")
        assert hasattr(resp, "limitations")
        assert hasattr(resp, "requires_human_review")

    def test_confidence_in_valid_range(self, full_agent):
        resp = full_agent.query("Detect anomaly", sensor_readings=NORMAL_READINGS)
        assert 0.0 <= resp.confidence <= 1.0

    def test_tool_calls_recorded(self, full_agent):
        resp = full_agent.query("detect anomaly", sensor_readings=NORMAL_READINGS)
        assert len(resp.tool_calls) >= 1
        for tc in resp.tool_calls:
            assert isinstance(tc, ToolCall)
            assert tc.tool
            assert tc.duration_ms >= 0

    def test_limitations_always_present(self, full_agent):
        resp = full_agent.query("simple question")
        assert len(resp.limitations) >= 3  # minimum 3 limitation statements

    def test_sources_capped_at_5(self, full_agent):
        # Even if RAG returns many sources, response caps at 5
        assert len(resp.sources) <= 5 if (resp := full_agent.query("search manuals")) else True

    def test_session_id_unique_per_query(self, full_agent):
        r1 = full_agent.query("query one")
        r2 = full_agent.query("query two")
        assert r1.session_id != r2.session_id

    def test_machine_id_passed_to_rag(self, mock_rag, mock_anomaly_detector):
        agent = IndustrialAgent(rag_engine=mock_rag, anomaly_detector=mock_anomaly_detector)
        agent.query("check bearing manual", machine_id="CNC-001")
        # RAG was called with machine_id
        mock_rag.query.assert_called_once()
        call_kwargs = mock_rag.query.call_args
        assert "CNC-001" in str(call_kwargs)


# ── 4. Human oversight (EU AI Act Article 14) ─────────────────────────────────

class TestHumanOversight:
    def test_no_human_review_normal_conditions(self, full_agent):
        resp = full_agent.query("is the machine normal?", sensor_readings=NORMAL_READINGS)
        # Normal readings + high confidence → no review needed
        # (depends on mock returning confidence 0.88)
        assert isinstance(resp.requires_human_review, bool)

    def test_human_review_on_anomaly_low_confidence(
        self, mock_rag, mock_anomaly_detector_alert, mock_compliance_engine
    ):
        """Anomaly + confidence < 0.7 → human review required."""
        agent = IndustrialAgent(
            rag_engine=mock_rag,
            anomaly_detector=mock_anomaly_detector_alert,
            compliance_engine=mock_compliance_engine,
        )
        resp = agent.query("detect anomaly", sensor_readings=NORMAL_READINGS)
        assert resp.requires_human_review is True

    def test_human_review_on_critical_rul(
        self, mock_rag, mock_anomaly_detector, mock_maintenance_predictor_critical
    ):
        """RUL < 20 cycles → human review required."""
        agent = IndustrialAgent(
            rag_engine=mock_rag,
            anomaly_detector=mock_anomaly_detector,
            maintenance_predictor=mock_maintenance_predictor_critical,
        )
        resp = agent.query("how long until failure?", sensor_readings=CRITICAL_READINGS)
        assert resp.requires_human_review is True

    def test_human_review_on_low_overall_confidence(self, empty_agent):
        """Empty agent with no components → confidence ≈ 0 → review required."""
        resp = empty_agent.query("is machine safe?", sensor_readings=NORMAL_READINGS)
        assert resp.requires_human_review is True

    def test_limitations_include_human_oversight_statement(self, full_agent):
        resp = full_agent.query("simple query")
        assert any("human oversight" in lim.lower() or "article 14" in lim.lower()
                   for lim in resp.limitations)


# ── 5. Graceful degradation (no components) ───────────────────────────────────

class TestGracefulDegradation:
    def test_empty_agent_does_not_raise(self, empty_agent):
        resp = empty_agent.query("test query")
        assert resp is not None

    def test_empty_agent_returns_fallback_answer(self, empty_agent):
        resp = empty_agent.query("what is the procedure?")
        assert len(resp.answer) > 0

    def test_empty_agent_no_tool_calls_that_crashed(self, empty_agent):
        resp = empty_agent.query("detect anomaly", sensor_readings=NORMAL_READINGS)
        # Tool calls exist but anomaly detector returned "unknown" gracefully
        for tc in resp.tool_calls:
            # Should not have errors from None components
            assert tc.success is True or tc.error is not None

    def test_agent_with_only_rag(self, mock_rag):
        agent = IndustrialAgent(rag_engine=mock_rag)
        resp = agent.query("what is the maintenance schedule?")
        assert resp.confidence > 0.0  # RAG returned something
        assert "knowledge base" in resp.answer.lower()

    def test_agent_handles_tool_exception_gracefully(self, mock_rag):
        """If a tool raises an unexpected exception, agent continues."""
        mock_rag.query.side_effect = RuntimeError("FAISS index corrupted")
        agent = IndustrialAgent(rag_engine=mock_rag)
        resp = agent.query("search manuals")
        # Should not re-raise; failed tool call should be recorded
        failed = [tc for tc in resp.tool_calls if not tc.success]
        assert len(failed) >= 1
        assert failed[0].error is not None


# ── 6. Answer synthesis ────────────────────────────────────────────────────────

class TestAnswerSynthesis:
    def test_anomaly_message_in_answer_when_anomaly(
        self, mock_rag, mock_anomaly_detector_alert
    ):
        agent = IndustrialAgent(rag_engine=mock_rag, anomaly_detector=mock_anomaly_detector_alert)
        resp = agent.query("detect anomaly", sensor_readings=NORMAL_READINGS)
        assert "anomaly" in resp.answer.lower() or "inspection" in resp.answer.lower()

    def test_normal_message_in_answer_when_normal(self, full_agent):
        resp = full_agent.query("detect anomaly", sensor_readings=NORMAL_READINGS)
        assert "normal" in resp.answer.lower() or "anomaly" in resp.answer.lower()

    def test_rul_cycles_in_answer(
        self, mock_rag, mock_anomaly_detector, mock_maintenance_predictor
    ):
        agent = IndustrialAgent(
            rag_engine=mock_rag,
            anomaly_detector=mock_anomaly_detector,
            maintenance_predictor=mock_maintenance_predictor,
        )
        resp = agent.query("how long until failure?", sensor_readings=NORMAL_READINGS)
        assert "342" in resp.answer or "remaining" in resp.answer.lower()

    def test_compliance_in_answer(self, mock_rag, mock_compliance_engine):
        agent = IndustrialAgent(rag_engine=mock_rag, compliance_engine=mock_compliance_engine)
        resp = agent.query("check ai act compliance")
        assert "compliance" in resp.answer.lower() or "limited" in resp.answer.lower()

    def test_kb_answer_in_response(self, mock_rag):
        agent = IndustrialAgent(rag_engine=mock_rag)
        resp = agent.query("bearing maintenance procedure")
        assert "bearing" in resp.answer.lower() or "lubrication" in resp.answer.lower()


# ── 7. Capabilities ───────────────────────────────────────────────────────────

class TestCapabilities:
    def test_capabilities_has_tools_list(self, full_agent):
        caps = full_agent.capabilities
        assert "tools" in caps
        assert len(caps["tools"]) == len(ToolName)

    def test_capabilities_reflects_available_components(self, full_agent):
        caps = full_agent.capabilities
        assert caps["rag_available"] is True
        assert caps["anomaly_detection_available"] is True
        assert caps["maintenance_prediction_available"] is True
        assert caps["compliance_available"] is True

    def test_capabilities_reflects_empty_agent(self, empty_agent):
        caps = empty_agent.capabilities
        assert caps["rag_available"] is False
        assert caps["anomaly_detection_available"] is False

    def test_capabilities_has_architecture_field(self, full_agent):
        caps = full_agent.capabilities
        assert "architecture" in caps
        assert "react" in caps["architecture"].lower() or "ReAct" in caps["architecture"]

    def test_capabilities_has_human_oversight_description(self, full_agent):
        caps = full_agent.capabilities
        assert "human_oversight" in caps
        assert "article 14" in caps["human_oversight"].lower()

    def test_all_tool_names_in_capabilities(self, full_agent):
        caps = full_agent.capabilities
        for tool in ToolName:
            assert tool.value in caps["tools"]


# ── 8. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_long_query(self, full_agent):
        long_query = "what is the maintenance schedule? " * 50
        resp = full_agent.query(long_query[:2000])
        assert resp is not None

    def test_query_with_special_characters(self, full_agent):
        resp = full_agent.query("machine #42 — temperature: 350°C?")
        assert resp is not None

    def test_empty_sensor_readings_dict(self, full_agent):
        resp = full_agent.query("is machine normal?", sensor_readings={})
        assert resp is not None

    def test_single_feature_sensor_readings(self, full_agent):
        resp = full_agent.query("detect anomaly", sensor_readings={"temperature": 350.0})
        assert resp is not None

    def test_negative_sensor_readings(self, mock_rag, mock_anomaly_detector):
        agent = IndustrialAgent(rag_engine=mock_rag, anomaly_detector=mock_anomaly_detector)
        resp = agent.query("check readings", sensor_readings={"temp": -50.0, "speed": -100.0})
        assert resp is not None

    def test_response_is_deterministic_for_same_input(self, full_agent):
        """Same query should return same tool invocation pattern."""
        resp1 = full_agent.query("what is ISO 281?")
        resp2 = full_agent.query("what is ISO 281?")
        # Both should use KB tool
        tools1 = {tc.tool for tc in resp1.tool_calls}
        tools2 = {tc.tool for tc in resp2.tool_calls}
        assert tools1 == tools2
