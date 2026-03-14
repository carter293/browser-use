import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bubus import BaseEvent
from cdp_use.cdp.target import TargetID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer
from uuid_extensions import uuid7str

from browser_use.dom.views import DOMInteractedElement, SerializedDOMState

# Known placeholder image data for about:blank pages - a 4x4 white PNG
PLACEHOLDER_4PX_SCREENSHOT = (
	'iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGP8//8/AwwwMSAB3BwAlm4DBfIlvvkAAAAASUVORK5CYII='
)


# Pydantic
class TabInfo(BaseModel):
	"""Represents information about a browser tab"""

	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		populate_by_name=True,
	)

	# Original fields
	url: str
	title: str
	target_id: TargetID = Field(serialization_alias='tab_id', validation_alias=AliasChoices('tab_id', 'target_id'))
	parent_target_id: TargetID | None = Field(
		default=None, serialization_alias='parent_tab_id', validation_alias=AliasChoices('parent_tab_id', 'parent_target_id')
	)  # parent page that contains this popup or cross-origin iframe

	@field_serializer('target_id')
	def serialize_target_id(self, target_id: TargetID, _info: Any) -> str:
		return target_id[-4:]

	@field_serializer('parent_target_id')
	def serialize_parent_target_id(self, parent_target_id: TargetID | None, _info: Any) -> str | None:
		return parent_target_id[-4:] if parent_target_id else None


class PageInfo(BaseModel):
	"""Comprehensive page size and scroll information"""

	# Current viewport dimensions
	viewport_width: int
	viewport_height: int

	# Total page dimensions
	page_width: int
	page_height: int

	# Current scroll position
	scroll_x: int
	scroll_y: int

	# Calculated scroll information
	pixels_above: int
	pixels_below: int
	pixels_left: int
	pixels_right: int

	# Page statistics are now computed dynamically instead of stored


@dataclass
class NetworkRequest:
	"""Information about a pending network request"""

	url: str
	method: str = 'GET'
	loading_duration_ms: float = 0.0  # How long this request has been loading (ms since request started, max 10s)
	resource_type: str | None = None  # e.g., 'Document', 'Stylesheet', 'Image', 'Script', 'XHR', 'Fetch'


@dataclass
class PaginationButton:
	"""Information about a pagination button detected on the page"""

	button_type: str  # 'next', 'prev', 'first', 'last', 'page_number'
	backend_node_id: int  # Backend node ID for clicking
	text: str  # Button text/label
	selector: str  # XPath or other selector to locate the element
	is_disabled: bool = False  # Whether the button appears disabled


@dataclass
class BrowserStateSummary:
	"""The summary of the browser's current state designed for an LLM to process"""

	# provided by SerializedDOMState:
	dom_state: SerializedDOMState

	url: str
	title: str
	tabs: list[TabInfo]
	screenshot: str | None = field(default=None, repr=False)
	page_info: PageInfo | None = None  # Enhanced page information

	# Keep legacy fields for backward compatibility
	pixels_above: int = 0
	pixels_below: int = 0
	browser_errors: list[str] = field(default_factory=list)
	is_pdf_viewer: bool = False  # Whether the current page is a PDF viewer
	recent_events: str | None = None  # Text summary of recent browser events
	pending_network_requests: list[NetworkRequest] = field(default_factory=list)  # Currently loading network requests
	pagination_buttons: list[PaginationButton] = field(default_factory=list)  # Detected pagination buttons
	closed_popup_messages: list[str] = field(default_factory=list)  # Messages from auto-closed JavaScript dialogs


@dataclass
class BrowserStateHistory:
	"""The summary of the browser's state at a past point in time to usse in LLM message history"""

	url: str
	title: str
	tabs: list[TabInfo]
	interacted_element: list[DOMInteractedElement | None] | list[None]
	screenshot_path: str | None = None

	def get_screenshot(self) -> str | None:
		"""Load screenshot from disk and return as base64 string"""
		if not self.screenshot_path:
			return None

		import base64
		from pathlib import Path

		path_obj = Path(self.screenshot_path)
		if not path_obj.exists():
			return None

		try:
			with open(path_obj, 'rb') as f:
				screenshot_data = f.read()
			return base64.b64encode(screenshot_data).decode('utf-8')
		except Exception:
			return None

	def to_dict(self) -> dict[str, Any]:
		data = {}
		data['tabs'] = [tab.model_dump() for tab in self.tabs]
		data['screenshot_path'] = self.screenshot_path
		data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['url'] = self.url
		data['title'] = self.title
		return data


class BrowserError(Exception):
	"""Browser error with structured memory for LLM context management.

	This exception class provides separate memory contexts for browser actions:
	- short_term_memory: Immediate context shown once to the LLM for the next action
	- long_term_memory: Persistent error information stored across steps
	"""

	message: str
	short_term_memory: str | None = None
	long_term_memory: str | None = None
	details: dict[str, Any] | None = None
	while_handling_event: BaseEvent[Any] | None = None

	def __init__(
		self,
		message: str,
		short_term_memory: str | None = None,
		long_term_memory: str | None = None,
		details: dict[str, Any] | None = None,
		event: BaseEvent[Any] | None = None,
	):
		"""Initialize a BrowserError with structured memory contexts.

		Args:
			message: Technical error message for logging and debugging
			short_term_memory: Context shown once to LLM (e.g., available actions, options)
			long_term_memory: Persistent error info stored in agent memory
			details: Additional metadata for debugging
			event: The browser event that triggered this error
		"""
		self.message = message
		self.short_term_memory = short_term_memory
		self.long_term_memory = long_term_memory
		self.details = details
		self.while_handling_event = event
		super().__init__(message)

	def __str__(self) -> str:
		if self.details:
			return f'{self.message} ({self.details}) during: {self.while_handling_event}'
		elif self.while_handling_event:
			return f'{self.message} (while handling: {self.while_handling_event})'
		else:
			return self.message


class URLNotAllowedError(BrowserError):
	"""Error raised when a URL is not allowed"""


# ============================================================================
# Session Recording Models
# ============================================================================


class RecordedStep(BaseModel):
	"""One recorded agent action with the browser state the agent saw before acting.

	Captures exactly what the agent would see at each step: the DOM text representation
	(same format as the LLM receives), a screenshot, and stable element identifiers
	so the recording can be fed back as context during replay.
	"""

	model_config = ConfigDict(extra='forbid')

	step_number: int
	timestamp: float

	# Agent-native state: what the agent saw before this action was taken
	url: str
	title: str
	dom_text: str  # output of dom_state.llm_representation() — same text the agent reads
	screenshot_b64: str | None = None  # base64 PNG, same image the agent sees

	# Action that was taken
	action_type: str  # e.g. 'click_element', 'input_text', 'navigate'
	action_params: dict[str, Any] = Field(default_factory=dict)

	# Stable element identifiers for element-based actions (for future hybrid replay)
	element_tag: str | None = None
	element_ax_name: str | None = None  # accessible name (aria-label, button text, etc.)
	element_xpath: str | None = None
	element_stable_hash: str | None = None

	# Optional voice overlay transcript aligned to this step
	voice_transcript: str | None = None


class SessionRecord(BaseModel):
	"""Complete recording of a browser automation session.

	Stores per-step agent-native state (DOM text + screenshot) alongside the
	action taken, so the recording can be replayed by feeding it as structured
	context to an agent.
	"""

	model_config = ConfigDict(extra='forbid')

	session_id: str = Field(default_factory=uuid7str)
	created_at: float = Field(default_factory=time.time)
	task_description: str | None = None
	steps: list[RecordedStep] = Field(default_factory=list)

	def save(self, path: Path | str) -> None:
		"""Serialize the recording to a JSON file."""
		dest = Path(path)
		dest.parent.mkdir(parents=True, exist_ok=True)
		dest.write_text(self.model_dump_json(indent=2), encoding='utf-8')

	@classmethod
	def load(cls, path: Path | str) -> 'SessionRecord':
		"""Load a recording from a JSON file."""
		return cls.model_validate_json(Path(path).read_text(encoding='utf-8'))

	def to_agent_context(self) -> str:
		"""Serialize the recording as a compact, human-readable string for agent replay.

		The output is suitable for injection into an agent's extend_system_message
		or as a per-step hint, giving the agent structured context about what was
		done in the original recording at each equivalent step.
		"""
		lines: list[str] = []
		if self.task_description:
			lines.append(f'Recorded task: {self.task_description}')
			lines.append('')
		lines.append(f'Recording has {len(self.steps)} steps:')
		lines.append('')
		for step in self.steps:
			line = f'Step {step.step_number}: [{step.action_type}]'
			if step.url:
				line += f' on {step.url}'
			lines.append(line)
			if step.element_ax_name:
				lines.append(f'  element: {step.element_ax_name} <{step.element_tag}>')
			if step.action_params:
				params_str = json.dumps(step.action_params, ensure_ascii=False)
				lines.append(f'  params: {params_str}')
			if step.voice_transcript:
				lines.append(f'  voice note: {step.voice_transcript}')
		return '\n'.join(lines)
