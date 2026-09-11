'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const core = require('../experiments/section_editor_state.js');

const DURATION = 120;

function firstLayer(state) {
  return state.layers[0];
}

test('creation makes one blank, selected Song parts partition', () => {
  const state = core.create(DURATION);
  const layer = firstLayer(state);
  const segment = layer.segments[0];
  assert.equal(layer.name, 'Song parts');
  assert.deepEqual(
    { start_s: segment.start_s, end_s: segment.end_s, label: segment.label, motif: segment.motif, notes: segment.notes, certainty: segment.certainty },
    { start_s: 0, end_s: DURATION, label: '', motif: '', notes: '', certainty: 'unspecified' }
  );
  assert.equal(state.active_layer_id, layer.id);
  assert.equal(state.selected_segment_id, segment.id);
  assert.deepEqual(core.validate(JSON.parse(JSON.stringify(state)), DURATION), state);
});

test('split preserves left annotations, creates a blank right segment, and rejects duplicate boundaries', () => {
  let state = core.create(DURATION);
  const layer = firstLayer(state);
  const originalId = layer.segments[0].id;
  state = core.updateSegment(state, layer.id, originalId, {
    label: 'Verse', motif: 'riff', notes: 'intro', certainty: 'clear'
  }, DURATION);
  const split = core.split(state, layer.id, 48, DURATION);
  const [left, right] = firstLayer(split).segments;
  assert.equal(left.id, originalId);
  assert.equal(left.end_s, 48);
  assert.deepEqual(
    { label: left.label, motif: left.motif, notes: left.notes, certainty: left.certainty },
    { label: 'Verse', motif: 'riff', notes: 'intro', certainty: 'clear' }
  );
  assert.deepEqual(
    { start_s: right.start_s, end_s: right.end_s, label: right.label, motif: right.motif, notes: right.notes, certainty: right.certainty },
    { start_s: 48, end_s: DURATION, label: '', motif: '', notes: '', certainty: 'unspecified' }
  );
  assert.equal(split.selected_segment_id, right.id);
  assert.throws(() => core.split(split, layer.id, 48, DURATION), /strictly inside/);
});

test('moveBoundary changes only its adjacent spans and cannot cross neighbors', () => {
  let state = core.create(DURATION);
  const layer = firstLayer(state);
  state = core.split(state, layer.id, 40, DURATION);
  const rightId = firstLayer(state).segments[1].id;
  const moved = core.moveBoundary(state, layer.id, rightId, 55, DURATION);
  assert.deepEqual(firstLayer(moved).segments.map(s => [s.start_s, s.end_s]), [[0, 55], [55, 120]]);
  assert.throws(() => core.moveBoundary(moved, layer.id, rightId, 0, DURATION), /cannot cross/);
  assert.throws(() => core.moveBoundary(moved, layer.id, rightId, 120, DURATION), /cannot cross/);
});

test('removeBoundary conservatively retains both sides annotations and downgrades disagreement', () => {
  let state = core.create(DURATION);
  const layer = firstLayer(state);
  const leftId = layer.segments[0].id;
  state = core.updateSegment(state, layer.id, leftId, {
    label: 'Verse', motif: 'riff', notes: 'left note', certainty: 'clear'
  }, DURATION);
  state = core.split(state, layer.id, 60, DURATION);
  const rightId = firstLayer(state).segments[1].id;
  state = core.updateSegment(state, layer.id, rightId, {
    label: 'Chorus', motif: 'hook', notes: 'right note', certainty: 'uncertain'
  }, DURATION);
  const merged = core.removeBoundary(state, layer.id, rightId, DURATION);
  const segment = firstLayer(merged).segments[0];
  assert.equal(segment.id, leftId);
  assert.deepEqual(
    { label: segment.label, motif: segment.motif, notes: segment.notes, certainty: segment.certainty },
    { label: 'Verse / Chorus', motif: 'riff / hook', notes: 'left note\n\nright note', certainty: 'uncertain' }
  );
  assert.equal(merged.selected_segment_id, leftId);
});

test('layers are independent and may cross one another boundaries', () => {
  let state = core.create(DURATION);
  const songLayer = firstLayer(state);
  state = core.split(state, songLayer.id, 40, DURATION);
  state = core.addLayer(state, 'Energy', DURATION);
  const energyLayer = state.layers[1];
  state = core.split(state, energyLayer.id, 65, DURATION);
  assert.deepEqual(state.layers[0].segments.map(s => s.end_s), [40, 120]);
  assert.deepEqual(state.layers[1].segments.map(s => s.end_s), [65, 120]);
  assert.doesNotThrow(() => core.validate(state, DURATION));
});

test('all mutations are immutable and patches are field-whitelisted', () => {
  const initial = core.create(DURATION);
  const snapshot = JSON.stringify(initial);
  const layer = firstLayer(initial);
  const changed = core.updateSegment(initial, layer.id, layer.segments[0].id, { label: 'Intro' }, DURATION);
  assert.notEqual(changed, initial);
  assert.equal(JSON.stringify(initial), snapshot);
  assert.equal(changed.layers[0].segments[0].label, 'Intro');
  assert.throws(
    () => core.updateSegment(initial, layer.id, layer.segments[0].id, JSON.parse('{"__proto__":{"polluted":true}}'), DURATION),
    /not allowed/
  );
  assert.equal({}.polluted, undefined);
});

test('fallback IDs skip imported IDs when crypto.randomUUID is unavailable', () => {
  const source = fs.readFileSync(path.join(__dirname, '../experiments/section_editor_state.js'), 'utf8');
  const sandbox = { crypto: {} };
  vm.runInNewContext(source, sandbox, { filename: 'section_editor_state.js' });
  const fallbackCore = sandbox.SectionEditorCore;
  sandbox.importedStateJson = JSON.stringify({
    layers: [{
      id: 'section-editor-1',
      name: 'Imported',
      segments: [{
        id: 'section-editor-2', start_s: 0, end_s: 120,
        label: '', motif: '', notes: '', certainty: 'unspecified'
      }]
    }],
    active_layer_id: 'section-editor-1',
    selected_segment_id: 'section-editor-2',
    global_notes: ''
  });
  const imported = vm.runInNewContext('JSON.parse(importedStateJson)', sandbox);
  const split = fallbackCore.split(imported, 'section-editor-1', 50);
  const added = fallbackCore.addLayer(split, 'Imported second layer', 120);
  const allIds = Array.from(added.layers.flatMap(layer => [layer.id, ...layer.segments.map(segment => segment.id)]));
  assert.equal(new Set(allIds).size, allIds.length);
  assert.deepEqual(allIds, ['section-editor-1', 'section-editor-2', 'section-editor-3', 'section-editor-4', 'section-editor-5']);
  assert.doesNotThrow(() => fallbackCore.validate(added, 120));
});

test('validation rejects malformed, noncontiguous, nonfinite, duplicate, and invalid selection states', () => {
  const valid = core.create(DURATION);
  const cases = [
    state => { state.layers[0].segments[0].end_s = '120'; },
    state => { state.layers[0].segments[0].end_s = Infinity; },
    state => { state.layers[0].segments = [
      { ...state.layers[0].segments[0], end_s: 50 },
      { ...state.layers[0].segments[0], id: 'other', start_s: 51, end_s: 120 }
    ]; },
    state => { state.layers[0].segments[0].certainty = 'maybe'; },
    state => { state.layers.push({ id: state.layers[0].id, name: 'Duplicate', segments: [{ ...state.layers[0].segments[0], id: 'different' }] }); },
    state => { state.selected_segment_id = 'missing'; },
    state => { state.layers = []; }
  ];
  for (const mutate of cases) {
    const malformed = JSON.parse(JSON.stringify(valid));
    mutate(malformed);
    assert.throws(() => core.validate(malformed, DURATION), Error);
  }
});
