(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.SectionEditorCore = api;
  }
}(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : null), function () {
  'use strict';

  var CERTAINTIES = { unspecified: true, clear: true, uncertain: true };
  var PATCH_FIELDS = { label: true, motif: true, notes: true, certainty: true };
  var fallbackIdCounter = 0;

  function hasOwn(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
  }

  function isRecord(value) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
    var prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
  }

  function fail(message) {
    throw new Error('Invalid section annotation state: ' + message);
  }

  function assertDuration(duration) {
    if (typeof duration !== 'number' || !Number.isFinite(duration) || duration <= 0) {
      throw new Error('duration_s must be a finite number greater than zero');
    }
  }

  function assertString(value, field) {
    if (typeof value !== 'string') fail(field + ' must be a string');
  }

  function assertId(value, field, seenIds) {
    assertString(value, field);
    if (value.length === 0) fail(field + ' must not be empty');
    if (seenIds[value]) fail('duplicate id: ' + value);
    seenIds[value] = true;
  }

  function documentIdSet(state) {
    var ids = Object.create(null);
    state.layers.forEach(function (layer) {
      ids[layer.id] = true;
      layer.segments.forEach(function (segment) {
        ids[segment.id] = true;
      });
    });
    return ids;
  }

  function newId(usedIds) {
    var id;
    var cryptoObject = typeof globalThis !== 'undefined' ? globalThis.crypto : null;
    do {
      if (cryptoObject && typeof cryptoObject.randomUUID === 'function') {
        id = cryptoObject.randomUUID();
      } else {
        fallbackIdCounter += 1;
        id = 'section-editor-' + fallbackIdCounter;
      }
    } while (usedIds && hasOwn(usedIds, id));
    if (usedIds) usedIds[id] = true;
    return id;
  }

  function blankSegment(id, start, end) {
    return {
      id: id,
      start_s: start,
      end_s: end,
      label: '',
      motif: '',
      notes: '',
      certainty: 'unspecified'
    };
  }

  function validate(state, duration) {
    assertDuration(duration);
    if (!isRecord(state)) fail('state must be an object');
    if (!hasOwn(state, 'layers') || !Array.isArray(state.layers) || state.layers.length === 0) {
      fail('layers must be a non-empty array');
    }
    if (!hasOwn(state, 'active_layer_id')) fail('active_layer_id is required');
    if (!hasOwn(state, 'selected_segment_id')) fail('selected_segment_id is required');
    if (!hasOwn(state, 'global_notes')) fail('global_notes is required');
    assertString(state.active_layer_id, 'active_layer_id');
    assertString(state.selected_segment_id, 'selected_segment_id');
    assertString(state.global_notes, 'global_notes');

    var seenIds = Object.create(null);
    var normalizedLayers = [];
    var activeLayer = null;
    var selectedSegment = null;

    state.layers.forEach(function (layer, layerIndex) {
      if (!isRecord(layer)) fail('layers[' + layerIndex + '] must be an object');
      if (!hasOwn(layer, 'id') || !hasOwn(layer, 'name') || !hasOwn(layer, 'segments')) {
        fail('layers[' + layerIndex + '] is missing a required field');
      }
      assertId(layer.id, 'layers[' + layerIndex + '].id', seenIds);
      assertString(layer.name, 'layers[' + layerIndex + '].name');
      if (!Array.isArray(layer.segments) || layer.segments.length === 0) {
        fail('layers[' + layerIndex + '].segments must be a non-empty array');
      }

      var normalizedSegments = [];
      var expectedStart = 0;
      layer.segments.forEach(function (segment, segmentIndex) {
        var path = 'layers[' + layerIndex + '].segments[' + segmentIndex + ']';
        if (!isRecord(segment)) fail(path + ' must be an object');
        ['id', 'start_s', 'end_s', 'label', 'motif', 'notes', 'certainty'].forEach(function (field) {
          if (!hasOwn(segment, field)) fail(path + '.' + field + ' is required');
        });
        assertId(segment.id, path + '.id', seenIds);
        if (typeof segment.start_s !== 'number' || !Number.isFinite(segment.start_s) ||
            typeof segment.end_s !== 'number' || !Number.isFinite(segment.end_s)) {
          fail(path + ' bounds must be finite numbers');
        }
        if (segment.start_s !== expectedStart) fail(path + ' is unsorted, gapped, or overlapping');
        if (segment.start_s < 0 || segment.end_s > duration || segment.end_s <= segment.start_s) {
          fail(path + ' has invalid bounds');
        }
        assertString(segment.label, path + '.label');
        assertString(segment.motif, path + '.motif');
        assertString(segment.notes, path + '.notes');
        if (typeof segment.certainty !== 'string' || !hasOwn(CERTAINTIES, segment.certainty)) {
          fail(path + '.certainty is invalid');
        }
        expectedStart = segment.end_s;
        normalizedSegments.push({
          id: segment.id,
          start_s: segment.start_s,
          end_s: segment.end_s,
          label: segment.label,
          motif: segment.motif,
          notes: segment.notes,
          certainty: segment.certainty
        });
      });
      if (expectedStart !== duration) fail('layers[' + layerIndex + '] does not end at duration_s');
      var normalizedLayer = { id: layer.id, name: layer.name, segments: normalizedSegments };
      normalizedLayers.push(normalizedLayer);
      if (layer.id === state.active_layer_id) activeLayer = normalizedLayer;
    });

    if (!activeLayer) fail('active_layer_id does not identify a layer');
    activeLayer.segments.forEach(function (segment) {
      if (segment.id === state.selected_segment_id) selectedSegment = segment;
    });
    if (!selectedSegment) fail('selected_segment_id must belong to the active layer');

    return {
      layers: normalizedLayers,
      active_layer_id: state.active_layer_id,
      selected_segment_id: state.selected_segment_id,
      global_notes: state.global_notes
    };
  }

  function create(duration) {
    assertDuration(duration);
    var usedIds = Object.create(null);
    var layerId = newId(usedIds);
    var segmentId = newId(usedIds);
    return {
      layers: [{ id: layerId, name: 'Song parts', segments: [blankSegment(segmentId, 0, duration)] }],
      active_layer_id: layerId,
      selected_segment_id: segmentId,
      global_notes: ''
    };
  }

  function findLayer(state, layerId) {
    var layer = state.layers.find(function (item) { return item.id === layerId; });
    if (!layer) throw new Error('Unknown layer id: ' + layerId);
    return layer;
  }

  function durationFromState(state) {
    if (!isRecord(state) || !Array.isArray(state.layers) || state.layers.length === 0 ||
        !isRecord(state.layers[0]) || !Array.isArray(state.layers[0].segments) ||
        state.layers[0].segments.length === 0) {
      return NaN;
    }
    return state.layers[0].segments[state.layers[0].segments.length - 1].end_s;
  }

  function cloneForEdit(state) {
    return validate(state, durationFromState(state));
  }

  function split(state, layerId, time) {
    var next = cloneForEdit(state);
    if (typeof time !== 'number' || !Number.isFinite(time)) throw new Error('Split time must be finite');
    var layer = findLayer(next, layerId);
    var index = layer.segments.findIndex(function (segment) {
      return time > segment.start_s && time < segment.end_s;
    });
    if (index === -1) throw new Error('Split time must be strictly inside a segment');
    var left = layer.segments[index];
    var right = blankSegment(newId(documentIdSet(next)), time, left.end_s);
    left.end_s = time;
    layer.segments.splice(index + 1, 0, right);
    next.active_layer_id = layer.id;
    next.selected_segment_id = right.id;
    return next;
  }

  function moveBoundary(state, layerId, rightSegmentId, time) {
    var next = cloneForEdit(state);
    if (typeof time !== 'number' || !Number.isFinite(time)) throw new Error('Boundary time must be finite');
    var layer = findLayer(next, layerId);
    var rightIndex = layer.segments.findIndex(function (segment) { return segment.id === rightSegmentId; });
    if (rightIndex < 1) throw new Error('A boundary requires a right segment with a left neighbor');
    var left = layer.segments[rightIndex - 1];
    var right = layer.segments[rightIndex];
    if (time <= left.start_s || time >= right.end_s) {
      throw new Error('Boundary cannot cross its neighboring boundaries');
    }
    left.end_s = time;
    right.start_s = time;
    return next;
  }

  function joinUnique(left, right, delimiter) {
    if (left === '') return right;
    if (right === '' || right === left) return left;
    return left + delimiter + right;
  }

  // A mismatch never upgrades confidence: unspecified wins; clear/uncertain becomes uncertain.
  function mergeCertainty(left, right) {
    if (left === right) return left;
    if (left === 'unspecified' || right === 'unspecified') return 'unspecified';
    return 'uncertain';
  }

  function removeBoundary(state, layerId, rightSegmentId) {
    var next = cloneForEdit(state);
    var layer = findLayer(next, layerId);
    var rightIndex = layer.segments.findIndex(function (segment) { return segment.id === rightSegmentId; });
    if (rightIndex < 1) throw new Error('A boundary requires a right segment with a left neighbor');
    var left = layer.segments[rightIndex - 1];
    var right = layer.segments[rightIndex];
    left.end_s = right.end_s;
    left.label = joinUnique(left.label, right.label, ' / ');
    left.motif = joinUnique(left.motif, right.motif, ' / ');
    left.notes = joinUnique(left.notes, right.notes, '\n\n');
    left.certainty = mergeCertainty(left.certainty, right.certainty);
    layer.segments.splice(rightIndex, 1);
    next.active_layer_id = layer.id;
    next.selected_segment_id = left.id;
    return next;
  }

  function addLayer(state, name, duration) {
    var next = validate(state, duration);
    if (typeof name !== 'string') throw new Error('Layer name must be a string');
    var usedIds = documentIdSet(next);
    var layerId = newId(usedIds);
    var segmentId = newId(usedIds);
    next.layers.push({ id: layerId, name: name, segments: [blankSegment(segmentId, 0, duration)] });
    next.active_layer_id = layerId;
    next.selected_segment_id = segmentId;
    return next;
  }

  function renameLayer(state, layerId, name) {
    var next = cloneForEdit(state);
    if (typeof name !== 'string') throw new Error('Layer name must be a string');
    findLayer(next, layerId).name = name;
    return next;
  }

  function updateSegment(state, layerId, segmentId, patch) {
    var next = cloneForEdit(state);
    if (!isRecord(patch)) throw new Error('Segment patch must be a plain object');
    Object.keys(patch).forEach(function (key) {
      if (!hasOwn(PATCH_FIELDS, key)) throw new Error('Segment patch field is not allowed: ' + key);
    });
    var layer = findLayer(next, layerId);
    var segment = layer.segments.find(function (item) { return item.id === segmentId; });
    if (!segment) throw new Error('Unknown segment id: ' + segmentId);
    ['label', 'motif', 'notes'].forEach(function (key) {
      if (hasOwn(patch, key)) {
        if (typeof patch[key] !== 'string') throw new Error('Segment patch ' + key + ' must be a string');
        segment[key] = patch[key];
      }
    });
    if (hasOwn(patch, 'certainty')) {
      if (typeof patch.certainty !== 'string' || !hasOwn(CERTAINTIES, patch.certainty)) {
        throw new Error('Segment patch certainty is invalid');
      }
      segment.certainty = patch.certainty;
    }
    return next;
  }

  function select(state, layerId, segmentId) {
    var next = cloneForEdit(state);
    var layer = findLayer(next, layerId);
    if (!layer.segments.some(function (segment) { return segment.id === segmentId; })) {
      throw new Error('Unknown segment id: ' + segmentId);
    }
    next.active_layer_id = layer.id;
    next.selected_segment_id = segmentId;
    return next;
  }

  return {
    create: create,
    validate: validate,
    split: split,
    moveBoundary: moveBoundary,
    removeBoundary: removeBoundary,
    addLayer: addLayer,
    renameLayer: renameLayer,
    updateSegment: updateSegment,
    select: select
  };
}));
