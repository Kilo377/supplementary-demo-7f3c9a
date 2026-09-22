from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass

from core.feedback_types import EnvironmentFeedback
from contextual_world.world_old.graph.world_graph import WorldGraph
from contextual_world.world_old.graph.world_graph_transition import WorldGraphTransitionReport
from contextual_world.world_old.graph.world_state_diff import build_agent_centered_world_state_diff

from contextual_world.modules.feedback_summary import run_world_feedback_summary
from contextual_world.modules.feedback_summary.variables import build_world_feedback_summary_variables
from contextual_world.modules.interaction_focus import run_world_interaction_focus
from contextual_world.modules.interaction_focus.variables import build_world_interaction_focus_variables
from contextual_world.modules.node_support import run_world_node_support
from contextual_world.modules.node_support.variables import build_world_node_support_variables
from contextual_world.modules.state_transition import run_world_state_transition
from contextual_world.modules.state_transition.variables import build_world_state_transition_variables
from contextual_world.modules.transition_check import check_world_state_transition_result
from contextual_world.transition_apply import apply_state_transition_delta, materialize_node_support
from contextual_world.types import (
    ContextualWorldActionResult,
    ContextualWorldModuleError,
    ContextualWorldTraceStep,
)


class ContextualWorldUnsupportedAction(RuntimeError):
    pass


class ContextualWorldPipelineFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failed_module: str,
        trace: list[ContextualWorldTraceStep] | None = None,
        support_result: dict | None = None,
        focus_result: dict | None = None,
        transition_result: dict | None = None,
        transition_check: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_module = failed_module
        self.trace = list(trace or [])
        self.support_result = dict(support_result or {})
        self.focus_result = dict(focus_result or {})
        self.transition_result = dict(transition_result or {})
        self.transition_check = dict(transition_check or {})

    def diagnostics(self) -> dict:
        return {
            "failed_module": self.failed_module,
            "error": str(self),
            "support_result": self.support_result,
            "focus_result": self.focus_result,
            "world_state_transition": self.transition_result,
            "transition_check": self.transition_check,
            "contextual_world_trace": [step.to_dict() for step in self.trace],
        }


class ContextualWorldTransitionRejected(ContextualWorldPipelineFailure):
    pass


@dataclass
class ContextualWorldPipeline:
    graph: WorldGraph
    agent_node_id: str = "agent_01"
    provider_name: str = "ollama"
    model: str | None = None
    use_llm_feedback_summary: bool = True
    raise_unsupported: bool = True
    raise_check_rejected: bool = True

    def process_action(
        self,
        action_proposal: str,
        *,
        support_result: dict | None = None,
        support_trace: ContextualWorldTraceStep | None = None,
    ) -> ContextualWorldActionResult:
        trace: list[ContextualWorldTraceStep] = []
        before_graph = copy.deepcopy(self.graph)
        active_module = "world_node_support"
        active_variables: dict = {}
        support_dict: dict = {}
        focus_result = None
        transition_result = None
        check_result = None

        try:
            if support_result is None:
                support_variables = build_world_node_support_variables(
                    graph=self.graph,
                    agent_node_id=self.agent_node_id,
                    action_proposal=action_proposal,
                )
                active_variables = support_variables.to_dict()
                parsed_support, support_trace = run_world_node_support(
                    support_variables,
                    provider_name=self.provider_name,
                    model=self.model,
                )
                trace.append(support_trace)
                support_dict = parsed_support.to_dict()
                support_status = parsed_support.support_status
                support_reason = parsed_support.support_reason
            else:
                support_dict = dict(support_result or {})
                support_status = str(support_dict.get("support_status", "") or "supported")
                support_reason = str(support_dict.get("support_reason", "") or "")
                trace.append(
                    support_trace
                    or ContextualWorldTraceStep(
                        module_name="world_node_support",
                        parsed_output=support_dict,
                        phase="provided_result",
                    )
                )
            if support_status == "unsupported" and self.raise_unsupported:
                raise ContextualWorldUnsupportedAction(support_reason)

            active_module = "world_node_support_materialize"
            active_variables = {"support_result": support_dict}
            support_report = materialize_node_support(
                graph=self.graph,
                action_proposal=action_proposal,
                support_result=support_dict,
            )

            active_module = "world_interaction_focus"
            focus_variables = build_world_interaction_focus_variables(
                graph=self.graph,
                agent_node_id=self.agent_node_id,
                action_proposal=action_proposal,
                support_result=support_dict,
            )
            active_variables = focus_variables.to_dict()
            focus_result, focus_trace = run_world_interaction_focus(
                focus_variables,
                provider_name=self.provider_name,
                model=self.model,
            )
            trace.append(focus_trace)

            active_module = "world_state_transition"
            transition_variables = build_world_state_transition_variables(
                graph=self.graph,
                agent_node_id=self.agent_node_id,
                action_proposal=action_proposal,
                focus_result=focus_result,
                support_result=support_dict,
            )
            active_variables = transition_variables.to_dict()
            transition_result, transition_trace = run_world_state_transition(
                transition_variables,
                provider_name=self.provider_name,
                model=self.model,
            )
            trace.append(transition_trace)

            active_module = "world_transition_check"
            allowed_node_ids = set(transition_variables.node_reference.keys())
            removable_fact_keys = {
                (
                    str(edge.get("from_node_id", "") or ""),
                    str(edge.get("relation", "") or ""),
                    str(edge.get("to_node_id", "") or ""),
                )
                for edge in transition_variables.focused_fact_edges
            }
            active_variables = {
                "allowed_node_ids": sorted(allowed_node_ids),
                "removable_fact_keys": [list(item) for item in sorted(removable_fact_keys)],
            }
            check_result = check_world_state_transition_result(
                transition_result=transition_result,
                allowed_node_ids=allowed_node_ids,
                removable_fact_keys=removable_fact_keys,
                node_reference=transition_variables.node_reference,
            )
            trace.append(ContextualWorldTraceStep(
                module_name="world_transition_check",
                variables=active_variables,
                parsed_output=check_result.to_dict(),
            ))
            if check_result.check_status == "rejected" and self.raise_check_rejected:
                raise ContextualWorldTransitionRejected(
                    "; ".join(check_result.issues),
                    failed_module="world_transition_check",
                    trace=trace,
                    support_result=support_dict,
                    focus_result=focus_result.to_dict(),
                    transition_result=transition_result.to_dict(),
                    transition_check=check_result.to_dict(),
                )
        except (ContextualWorldUnsupportedAction, ContextualWorldPipelineFailure):
            raise
        except Exception as error:
            trace.append(_failure_trace_step(error, active_module, active_variables))
            raise ContextualWorldPipelineFailure(
                str(error),
                failed_module=active_module,
                trace=trace,
                support_result=support_dict,
                focus_result=focus_result.to_dict() if focus_result is not None else {},
                transition_result=transition_result.to_dict() if transition_result is not None else {},
                transition_check=check_result.to_dict() if check_result is not None else {},
            ) from error

        try:
            active_module = "world_transition_apply"
            active_variables = {
                "world_state_transition": transition_result.to_dict(),
                "transition_check": check_result.to_dict(),
            }
            transition_report = apply_state_transition_delta(
                graph=self.graph,
                transition_result=transition_result.to_dict(),
                report=support_report,
            )
            transition_report.transition_check = check_result.to_dict()
            world_state_diff = build_agent_centered_world_state_diff(
                before_graph=before_graph,
                after_graph=self.graph,
                transition_report=transition_report.to_dict(),
                agent_node_id=self.agent_node_id,
            ).to_dict()
        except Exception as error:
            trace.append(_failure_trace_step(error, active_module, active_variables))
            raise ContextualWorldPipelineFailure(
                str(error),
                failed_module=active_module,
                trace=trace,
                support_result=support_dict,
                focus_result=focus_result.to_dict(),
                transition_result=transition_result.to_dict(),
                transition_check=check_result.to_dict(),
            ) from error

        execution_result = transition_result.execution_result
        feedback_variables = build_world_feedback_summary_variables(
            agent_name=transition_variables.agent_name,
            action_proposal=action_proposal,
            execution_result=execution_result,
            world_state_diff=world_state_diff,
        )
        try:
            active_module = "world_feedback_summary"
            active_variables = feedback_variables.to_dict()
            summary_result, summary_trace = run_world_feedback_summary(
                feedback_variables,
                provider_name=self.provider_name,
                model=self.model,
                use_llm=self.use_llm_feedback_summary,
            )
            trace.append(summary_trace)
        except Exception as error:
            trace.append(_failure_trace_step(error, active_module, active_variables))
            raise ContextualWorldPipelineFailure(
                str(error),
                failed_module=active_module,
                trace=trace,
                support_result=support_dict,
                focus_result=focus_result.to_dict(),
                transition_result=transition_result.to_dict(),
                transition_check=check_result.to_dict(),
            ) from error
        feedback = EnvironmentFeedback(
            route="contextual_world",
            perception_summary=summary_result.perception_summary,
            graph_transition_report=transition_report.to_dict(),
            world_state_diff=world_state_diff,
        )
        return ContextualWorldActionResult(
            action_proposal=action_proposal,
            accepted=transition_result.accepted,
            actual_event=str(execution_result.get("actual_event", "") or ""),
            estimated_duration=str(execution_result.get("estimated_duration", "") or ""),
            support_result=support_dict,
            focus_result=focus_result.to_dict(),
            transition_result=transition_result.to_dict(),
            transition_check=check_result.to_dict(),
            transition_report=transition_report,
            world_state_diff=world_state_diff,
            feedback=feedback,
            trace=trace,
        )


def _failure_trace_step(
    error: Exception,
    module_name: str,
    variables: dict,
) -> ContextualWorldTraceStep:
    if isinstance(error, ContextualWorldModuleError):
        return error.trace_step
    return ContextualWorldTraceStep(
        module_name=module_name,
        variables=variables,
        error=traceback.format_exc(),
        phase="internal",
        error_type=type(error).__name__,
    )


def merge_transition_reports(first: WorldGraphTransitionReport, second: WorldGraphTransitionReport) -> WorldGraphTransitionReport:
    return WorldGraphTransitionReport(
        created_temporary_nodes=[*first.created_temporary_nodes, *second.created_temporary_nodes],
        updated_temporary_nodes=[*first.updated_temporary_nodes, *second.updated_temporary_nodes],
        removed_fact_edges=[*first.removed_fact_edges, *second.removed_fact_edges],
        added_fact_edges=[*first.added_fact_edges, *second.added_fact_edges],
        updated_agent_node=second.updated_agent_node or first.updated_agent_node,
        updated_element_nodes=[*first.updated_element_nodes, *second.updated_element_nodes],
        world_action_event=second.world_action_event or first.world_action_event,
        transition_check=second.transition_check or first.transition_check,
        warnings=[*first.warnings, *second.warnings],
    )
