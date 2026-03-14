"""Session Recording Watchdog — records browser actions with agent-native state.

Captures the DOM text (exactly what the agent reads via llm_representation()),
a screenshot, and stable element identifiers at each browser action.  The
result is a SessionRecord that can be fed back to an agent as structured replay
context via to_agent_context() or by injecting individual step hints.

Usage:
    watchdog = SessionRecordingWatchdog(
        event_bus=session.event_bus,
        browser_session=session,
        output_path=Path("recording.json"),
        task_description="Fill out the contact form",
    )
    watchdog.attach_to_session()

    # Access the live record at any time:
    record = watchdog.record

    # After the session ends, the record is saved automatically to output_path.
    # Or save manually:
    watchdog.record.save("my_recording.json")
"""

import time
from pathlib import Path
from typing import Any, ClassVar

from bubus import BaseEvent
from pydantic import Field
from pydantic import PrivateAttr

from browser_use.browser.events import (
	BrowserStopEvent,
	ClickCoordinateEvent,
	ClickElementEvent,
	GoBackEvent,
	NavigateToUrlEvent,
	ScrollEvent,
	SelectDropdownOptionEvent,
	SendKeysEvent,
	SwitchTabEvent,
	TypeTextEvent,
)
from browser_use.browser.views import RecordedStep, SessionRecord
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.dom.views import DEFAULT_INCLUDE_ATTRIBUTES, EnhancedDOMTreeNode


class SessionRecordingWatchdog(BaseWatchdog):
	"""Records browser actions with the agent-native browser state at each step.

	Reads browser_session._cached_browser_state_summary before each action event
	is processed — this is the same DOM snapshot the agent used when it decided
	to take the action, so the recording faithfully represents what the agent saw.

	The watchdog is intentionally passive: it never modifies the browser or
	dispatches new events.  It only observes and records.
	"""

	LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [
		ClickElementEvent,
		ClickCoordinateEvent,
		TypeTextEvent,
		NavigateToUrlEvent,
		ScrollEvent,
		SendKeysEvent,
		GoBackEvent,
		SwitchTabEvent,
		SelectDropdownOptionEvent,
		BrowserStopEvent,
	]
	EMITS: ClassVar[list[type[BaseEvent]]] = []

	output_path: Path | None = None
	task_description: str | None = None
	include_attributes: list[str] | None = Field(default_factory=lambda: list(DEFAULT_INCLUDE_ATTRIBUTES))

	_record: SessionRecord = PrivateAttr()
	_step_counter: int = PrivateAttr(default=0)

	def model_post_init(self, __context: Any) -> None:
		self._record = SessionRecord(task_description=self.task_description)
		self._step_counter = 0

	# -------------------------------------------------------------------------
	# Internal helpers
	# -------------------------------------------------------------------------

	def _snapshot_pre_action_state(self) -> tuple[str, str, str, str | None]:
		"""Read the cached pre-action browser state without triggering a new CDP round-trip.

		The agent always calls get_browser_state_summary() in _prepare_context()
		before dispatching action events.  The DOMWatchdog stores that result in
		browser_session._cached_browser_state_summary.  Reading it here gives us
		the exact DOM state the agent saw when deciding to take the current action.

		Returns:
		    (url, title, dom_text, screenshot_b64)
		"""
		cached = self.browser_session._cached_browser_state_summary
		if cached is None:
			return '', '', '', None

		url = cached.url
		title = cached.title
		screenshot = cached.screenshot

		try:
			dom_text = cached.dom_state.llm_representation(include_attributes=self.include_attributes)
		except Exception as e:
			self.logger.debug(f'[SessionRecordingWatchdog] dom llm_representation failed: {e}')
			dom_text = ''

		return url, title, dom_text, screenshot

	def _node_info(self, node: EnhancedDOMTreeNode | None) -> dict[str, str | None]:
		"""Extract stable element identifiers from an EnhancedDOMTreeNode."""
		if node is None:
			return {'element_tag': None, 'element_ax_name': None, 'element_xpath': None, 'element_stable_hash': None}
		return {
			'element_tag': node.node_name,
			'element_ax_name': node.ax_node.name if node.ax_node else None,
			'element_xpath': node.xpath,
			'element_stable_hash': node.stable_hash,
		}

	def _append_step(
		self,
		action_type: str,
		action_params: dict[str, Any],
		node: EnhancedDOMTreeNode | None = None,
	) -> None:
		"""Build a RecordedStep from current cached state and append it to the record."""
		self._step_counter += 1
		url, title, dom_text, screenshot = self._snapshot_pre_action_state()
		step = RecordedStep(
			step_number=self._step_counter,
			timestamp=time.time(),
			url=url,
			title=title,
			dom_text=dom_text,
			screenshot_b64=screenshot,
			action_type=action_type,
			action_params=action_params,
			**self._node_info(node),
		)
		self._record.steps.append(step)
		self.logger.debug(
			f'[SessionRecordingWatchdog] step {self._step_counter}: {action_type} on {url!r}'
		)

	# -------------------------------------------------------------------------
	# Event handlers
	# -------------------------------------------------------------------------

	async def on_ClickElementEvent(self, event: ClickElementEvent) -> None:
		self._append_step('click_element', {'button': event.button}, event.node)

	async def on_ClickCoordinateEvent(self, event: ClickCoordinateEvent) -> None:
		self._append_step('click_coordinate', {'x': event.coordinate_x, 'y': event.coordinate_y})

	async def on_TypeTextEvent(self, event: TypeTextEvent) -> None:
		# Never record sensitive text — replace with a placeholder so the
		# recording can still convey "a password was typed here".
		text = '<redacted>' if event.is_sensitive else event.text
		self._append_step('input_text', {'text': text, 'clear': event.clear}, event.node)

	async def on_NavigateToUrlEvent(self, event: NavigateToUrlEvent) -> None:
		self._append_step('navigate', {'url': event.url, 'new_tab': event.new_tab})

	async def on_ScrollEvent(self, event: ScrollEvent) -> None:
		self._append_step('scroll', {'direction': event.direction, 'amount': event.amount}, event.node)

	async def on_SendKeysEvent(self, event: SendKeysEvent) -> None:
		self._append_step('send_keys', {'keys': event.keys})

	async def on_GoBackEvent(self, event: GoBackEvent) -> None:
		self._append_step('go_back', {})

	async def on_SwitchTabEvent(self, event: SwitchTabEvent) -> None:
		target = str(event.target_id) if event.target_id else None
		self._append_step('switch_tab', {'target_id': target})

	async def on_SelectDropdownOptionEvent(self, event: SelectDropdownOptionEvent) -> None:
		self._append_step('select_dropdown_option', {'text': event.text}, event.node)

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		"""Auto-save recording to output_path when the browser session ends."""
		if self.output_path and self._record.steps:
			self._record.save(self.output_path)
			self.logger.info(
				f'[SessionRecordingWatchdog] saved {len(self._record.steps)} steps → {self.output_path}'
			)

	# -------------------------------------------------------------------------
	# Public API
	# -------------------------------------------------------------------------

	@property
	def record(self) -> SessionRecord:
		"""The live SessionRecord.  Access at any time; steps accumulate as actions fire."""
		return self._record
