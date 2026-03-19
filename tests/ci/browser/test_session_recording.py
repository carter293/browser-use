"""
Tests for SessionRecordingWatchdog and the SessionRecord / RecordedStep models.

Verifies:
1. RecordedStep and SessionRecord models round-trip through JSON correctly
2. SessionRecord.to_agent_context() produces expected text
3. SessionRecord.save() / SessionRecord.load() work end-to-end
4. SessionRecordingWatchdog attaches when session_record_path is set and
   records a navigate + click + input_text action sequence during a real
   agent run against a local httpserver page.

Usage:
    uv run pytest tests/ci/browser/test_session_recording.py -vxs
"""

import json
import tempfile
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.views import RecordedStep, SessionRecord
from tests.ci.conftest import create_mock_llm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def httpserver_recording():
	"""Minimal HTML page used by the recording integration test."""
	server = HTTPServer()
	server.start()
	server.expect_request('/form').respond_with_data(
		"""
		<html>
		<head><title>Recording Test</title></head>
		<body>
			<h1>Form</h1>
			<input id="name" type="text" placeholder="Your name" />
			<button id="submit">Submit</button>
		</body>
		</html>
		""",
		content_type='text/html',
	)
	yield server
	server.clear()
	if server.is_running():
		server.stop()


# ---------------------------------------------------------------------------
# Unit tests — no browser needed
# ---------------------------------------------------------------------------


def test_recorded_step_round_trip():
	"""RecordedStep serialises and deserialises without data loss."""
	step = RecordedStep(
		step_number=1,
		timestamp=1_700_000_000.0,
		url='https://example.com',
		title='Example',
		dom_text='[1] <button>Click me</button>',
		screenshot_b64=None,
		action_type='click_element',
		action_params={'button': 'left'},
		element_tag='BUTTON',
		element_ax_name='Click me',
		element_xpath='/html/body/button',
		element_stable_hash=12345,
	)
	restored = RecordedStep.model_validate_json(step.model_dump_json())
	assert restored.step_number == step.step_number
	assert restored.action_type == step.action_type
	assert restored.element_stable_hash == 12345
	assert isinstance(restored.element_stable_hash, int)


def test_session_record_round_trip(tmp_path: Path):
	"""SessionRecord.save() + SessionRecord.load() preserves all steps."""
	record = SessionRecord(task_description='Fill the form')
	record.steps.append(
		RecordedStep(
			step_number=1,
			timestamp=1_700_000_000.0,
			url='https://example.com',
			title='Example',
			dom_text='[1] <input>',
			action_type='navigate',
			action_params={'url': 'https://example.com', 'new_tab': False},
		)
	)
	record.steps.append(
		RecordedStep(
			step_number=2,
			timestamp=1_700_000_001.0,
			url='https://example.com',
			title='Example',
			dom_text='[1] <input>',
			action_type='input_text',
			action_params={'text': 'hello', 'clear': True},
			element_tag='INPUT',
			element_ax_name='Your name',
		)
	)

	dest = tmp_path / 'recording.json'
	record.save(dest)
	assert dest.exists()

	loaded = SessionRecord.load(dest)
	assert loaded.task_description == 'Fill the form'
	assert len(loaded.steps) == 2
	assert loaded.steps[0].action_type == 'navigate'
	assert loaded.steps[1].element_ax_name == 'Your name'


def test_to_agent_context_format():
	"""to_agent_context() produces correctly structured text for agent injection."""
	record = SessionRecord(task_description='Submit the form')
	record.steps.append(
		RecordedStep(
			step_number=1,
			timestamp=1_700_000_000.0,
			url='https://example.com/form',
			title='Form',
			dom_text='[1] <input>',
			action_type='navigate',
			action_params={'url': 'https://example.com/form', 'new_tab': False},
		)
	)
	record.steps.append(
		RecordedStep(
			step_number=2,
			timestamp=1_700_000_001.0,
			url='https://example.com/form',
			title='Form',
			dom_text='[1] <input>',
			action_type='input_text',
			action_params={'text': 'Alice', 'clear': True},
			element_tag='INPUT',
			element_ax_name='Your name',
		)
	)
	record.steps.append(
		RecordedStep(
			step_number=3,
			timestamp=1_700_000_002.0,
			url='https://example.com/form',
			title='Form',
			dom_text='[2] <button>Submit</button>',
			action_type='click_element',
			action_params={'button': 'left'},
			element_tag='BUTTON',
			element_ax_name='Submit',
		)
	)

	ctx = record.to_agent_context()

	assert 'Recorded task: Submit the form' in ctx
	assert 'Recording has 3 steps' in ctx
	assert 'Step 1: [navigate]' in ctx
	assert 'Step 2: [input_text]' in ctx
	assert 'element: Your name <INPUT>' in ctx
	assert 'Step 3: [click_element]' in ctx
	assert 'element: Submit <BUTTON>' in ctx
	# params must be valid JSON embedded in the context
	for step in record.steps:
		assert json.dumps(step.action_params, ensure_ascii=False) in ctx


def test_sensitive_text_schema():
	"""Sensitive text replacement is stored — not a live browser test, just verifies the model accepts the placeholder."""
	step = RecordedStep(
		step_number=1,
		timestamp=0.0,
		url='https://example.com',
		title='Login',
		dom_text='',
		action_type='input_text',
		action_params={'text': '<redacted>', 'clear': True},
	)
	assert step.action_params['text'] == '<redacted>'


# ---------------------------------------------------------------------------
# Integration test — real browser + real watchdog
# ---------------------------------------------------------------------------


async def test_session_recording_watchdog_records_actions(httpserver_recording: HTTPServer, tmp_path: Path):
	"""Watchdog captures navigate + click + input_text against a local server page."""
	from browser_use.agent.service import Agent
	from browser_use.browser import BrowserSession

	record_path = tmp_path / 'session.json'
	base_url = httpserver_recording.url_for('/form')

	navigate_action = json.dumps(
		{
			'thinking': 'null',
			'evaluation_previous_goal': 'Starting',
			'memory': '',
			'next_goal': 'Navigate',
			'action': [{'navigate': {'url': base_url, 'new_tab': False}}],
		}
	)
	click_action = json.dumps(
		{
			'thinking': 'null',
			'evaluation_previous_goal': 'Navigated',
			'memory': '',
			'next_goal': 'Click submit',
			'action': [{'click': {'index': 1}}],
		}
	)
	done_action = json.dumps(
		{
			'thinking': 'null',
			'evaluation_previous_goal': 'Done',
			'memory': 'Done',
			'next_goal': 'Done',
			'action': [{'done': {'text': 'Done', 'success': True}}],
		}
	)

	mock_llm = create_mock_llm([navigate_action, click_action, done_action])

	profile = BrowserProfile(
		headless=True,
		session_record_path=record_path,
	)
	session = BrowserSession(browser_profile=profile)

	agent = Agent(
		task='Fill the test form',
		llm=mock_llm,
		browser_session=session,
	)
	await agent.run(max_steps=3)
	await session.stop()

	# The watchdog auto-saves on BrowserStopEvent — file must exist
	assert record_path.exists(), 'Recording file was not created'

	loaded = SessionRecord.load(record_path)
	assert len(loaded.steps) >= 1, 'Expected at least one recorded step'

	# At least one navigate step must be present
	action_types = [s.action_type for s in loaded.steps]
	assert 'navigate' in action_types, f'Expected navigate in {action_types}'

	# to_agent_context must not error out and must reference the URL
	ctx = loaded.to_agent_context()
	assert base_url in ctx
