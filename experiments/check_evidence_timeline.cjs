/*
 * Actual-browser smoke check. Usage:
 * SONGVIZ_PLAYWRIGHT_MODULE=/path/to/playwright node experiments/check_evidence_timeline.cjs URL
 *
 * Page contract (kept deliberately small): #source-audio is the one native
 * full-song <audio>; #evidence-timeline is the clickable plot; #metric,
 * #stem, and #scale are the real selectors; four [data-excerpt-id] buttons
 * are the excerpt shortcuts; and #audio-status reports load errors.  The
 * page exposes window.__songvizEvidenceTimeline with data, seek(time),
 * playRange(start,end), selectedContext(), selectedPair(), and retryAudio().
 */
const assert = require('node:assert/strict');
const { chromium } = require(process.env.SONGVIZ_PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const url = process.argv[2];
  assert(url, 'Passe a URL da página evidence-timeline.');
  const browser = await chromium.launch({
    headless: true,
    args: ['--autoplay-policy=no-user-gesture-required'],
  });
  try {
    const context = await browser.newContext({ viewport: { width: 1100, height: 800 } });
    const page = await context.newPage();
    const failures = [];
    page.on('pageerror', error => failures.push('pageerror: ' + error.message));
    page.on('console', message => {
      if (message.type() === 'error') failures.push('console: ' + message.text());
    });
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => {
      const api = window.__songvizEvidenceTimeline;
      return api && api.data && ['seek', 'playRange', 'selectedContext', 'selectedPair', 'retryAudio']
        .every(name => typeof api[name] === 'function');
    }, undefined, { timeout: 15000 });

    const apiSummary = await page.evaluate(() => {
      const api = window.__songvizEvidenceTimeline;
      const data = api.data;
      return {
        audioPath: data.audio_path || data.audio?.path || data.source_audio_path || '',
        excerpts: Array.isArray(data.excerpts) ? data.excerpts.length : 0,
        stems: Array.isArray(data.stem_names) ? data.stem_names.length : 0,
      };
    });
    assert(apiSummary.excerpts >= 4, 'A página deve expor os quatro excertos existentes.');
    assert(apiSummary.stems >= 2, 'Dados de stems ausentes da página.');
    assert.equal(await page.locator('#stem').inputValue(), 'vocals', 'A abertura deve explicar a curva com voz como stem inicial.');
    assert.match(await page.locator('#graph-reading').innerText(), /linha ciano.*linha dourada/i,
      'A página não explica as duas linhas do gráfico inicial.');
    assert.match(await page.locator('#timeline-caption').innerText(), /Eixo horizontal.*Eixo vertical/i,
      'Os eixos/unidades do gráfico inicial não estão explicados.');

    const audio = page.locator('#source-audio');
    await audio.waitFor();
    await page.waitForFunction(() => {
      const a = document.querySelector('#source-audio');
      return a && a.readyState >= 1 && Number.isFinite(a.duration) && a.duration > 0;
    }, undefined, { timeout: 15000 });
    const nativeAudio = await audio.evaluate(a => ({ controls: a.controls, currentSrc: a.currentSrc, duration: a.duration }));
    assert(nativeAudio.controls && nativeAudio.currentSrc, 'O áudio original não usa controles nativos/fonte real.');
    if (apiSummary.audioPath) {
      const declared = new URL(apiSummary.audioPath, url);
      assert(new URL(nativeAudio.currentSrc).pathname === declared.pathname,
        'O elemento de áudio não aponta para a fonte declarada pelo pacote.');
    }
    await audio.evaluate(a => { a.muted = true; });

    const timeline = page.locator('#evidence-timeline');
    await timeline.waitFor();
    const duration = await audio.evaluate(a => a.duration);
    const openingGuide = await page.locator('#graph-reading').innerText();
    assert.match(openingGuide, /dados começam na primeira âncora com suporte/i,
      'A explicação deve distinguir a âncora inicial da posição atual.');
    async function clickPlotTime(time, width) {
      await page.setViewportSize({ width, height: 812 });
      const box = await timeline.boundingBox();
      assert(box && box.width > 80 && box.height > 20, 'Linha do tempo não possui área clicável.');
      // The source plot occupies SVG x=40…870 in a 900-unit viewBox; click a
      // real plotted coordinate, not the SVG edge.
      const x = box.width * (40 + 830 * time / duration) / 900;
      await timeline.click({ position: { x, y: box.height * 0.5 } });
      await page.waitForFunction(() => !document.querySelector('#source-audio').seeking, undefined, { timeout: 3000 });
      const actual = await audio.evaluate(a => a.currentTime);
      assert(Math.abs(actual - time) <= 0.45,
        'Clique no tempo '+time+' buscou '+actual+' em '+width+'px, não a coordenada do gráfico.');
    }
    // P1: verify tick/end transforms in both responsive layouts.
    for (const time of [0, 60, duration]) await clickPlotTime(time, 1100);
    for (const time of [60, duration]) await clickPlotTime(time, 320);
    await page.setViewportSize({ width: 1100, height: 800 });

    // P5: the initial supported anchor is immutable when playback/cursor moves.
    await page.evaluate(() => window.__songvizEvidenceTimeline.seek(100));
    await page.waitForFunction(() => Math.abs(document.querySelector('#source-audio').currentTime - 100) <= .35);
    assert.equal(await page.locator('#graph-reading').innerText(), openingGuide,
      'A explicação da âncora inicial mudou com o cursor atual.');

    // P2: playback and native seeking must keep the range slider on the shared cursor.
    await page.evaluate(() => window.__songvizEvidenceTimeline.seek(40));
    await audio.evaluate(a => a.play());
    await page.waitForFunction(() => document.querySelector('#source-audio').currentTime > 40.7, undefined, { timeout: 4000 });
    let cursorState = await page.evaluate(() => ({
      audio: document.querySelector('#source-audio').currentTime,
      slider: Number(document.querySelector('#cursor').value),
      displayed: document.querySelector('#time').textContent,
    }));
    // Native timeupdate is coarsely scheduled, so permit one update interval,
    // while rejecting the former whole-range stale slider.
    assert(Math.abs(cursorState.audio - cursorState.slider) <= .30, 'Slider ficou atrasado durante reprodução.');
    const displayedMatch = cursorState.displayed.match(/^(\d+):(\d+(?:\.\d+)?)/);
    assert(displayedMatch, 'Tempo exibido não contém um cursor legível.');
    const displayedSeconds = Number(displayedMatch[1]) * 60 + Number(displayedMatch[2]);
    assert(Math.abs(cursorState.audio - displayedSeconds) <= .30, 'Tempo exibido não acompanhou o áudio em reprodução.');
    await audio.evaluate(a => { a.pause(); a.currentTime = 100; });
    await page.waitForFunction(() => Math.abs(Number(document.querySelector('#cursor').value) - 100) <= .35, undefined, { timeout: 3000 });

    // P3: a new external seek replaces an old bounded audition regardless of origin.
    async function assertRangeDoesNotHijack(seekKind) {
      await page.evaluate(() => window.__songvizEvidenceTimeline.playRange(16, 26));
      await page.waitForFunction(() => !document.querySelector('#source-audio').paused, undefined, { timeout: 3000 });
      if (seekKind === 'plot') await clickPlotTime(100, 1100);
      if (seekKind === 'slider') await page.locator('#cursor').evaluate(el => {
        el.value = '100'; el.dispatchEvent(new Event('input', { bubbles: true }));
      });
      if (seekKind === 'native') await audio.evaluate(a => { a.currentTime = 100; });
      await page.waitForFunction(() => document.querySelector('#source-audio').currentTime > 100.35, undefined, { timeout: 4000 });
      const state = await audio.evaluate(a => ({ time: a.currentTime, paused: a.paused }));
      assert(state.time > 100.35 && state.time < 105 && !state.paused,
        'Seek '+seekKind+' foi sobrescrito pelo limite antigo de 26s: '+JSON.stringify(state));
      const stateLabel = await page.locator('#audio-status').innerText();
      assert.doesNotMatch(stateLabel, /Tocando 0:16.*0:26/, 'O estado textual reteve a audição anterior após seek '+seekKind+'.');
      await audio.evaluate(a => a.pause());
    }
    for (const seekKind of ['plot', 'slider', 'native']) await assertRangeDoesNotHijack(seekKind);
    await page.evaluate(() => window.__songvizEvidenceTimeline.playRange(57, 58));
    await page.waitForFunction(() => document.querySelector('#source-audio').paused && Math.abs(document.querySelector('#source-audio').currentTime - 58) <= .01,
      undefined, { timeout: 4000 });

    // P4: a history value is valid only inside its saved posterior source span.
    const historyCase = await page.evaluate(() => {
      const scale = window.__songvizEvidenceTimeline.data.recurrence.scales[0];
      const row = scale.context.find(x => Number.isFinite(x.historical_pattern_novelty));
      return { inside: (scale.spans[row.span].start_s + scale.spans[row.span].end_s) / 2, tail: window.__songvizEvidenceTimeline.data.duration_s };
    });
    await page.evaluate(time => window.__songvizEvidenceTimeline.seek(time), historyCase.inside);
    await page.waitForFunction(() => /Janela-fonte posterior/.test(document.querySelector('#history').textContent));
    const insideHistory = await page.locator('#history').innerText();
    assert.match(insideHistory, /cursor .* está dentro dessa janela/i, 'História não declarou sua relação com o cursor.');
    assert.match(insideHistory, /independente|mesma janela/i, 'História não declarou sua relação com o par selecionado.');
    await page.evaluate(time => window.__songvizEvidenceTimeline.seek(time), historyCase.tail);
    await page.waitForFunction(() => /História: indisponível no cursor/.test(document.querySelector('#history').textContent));
    const tailHistory = await page.locator('#history').innerText();
    assert.doesNotMatch(tailHistory, /0\.28247/, 'Valor histórico da última janela foi apresentado no final sem suporte.');
    assert.match(tailHistory, /Nenhum valor de registro vizinho foi transportado/i,
      'A borda sem suporte precisa declarar que não transportou o registro vizinho.');

    const shortcuts = page.locator('[data-excerpt-id]');
    assert.equal(await shortcuts.count(), 4, 'Esperados exatamente quatro atalhos de excerto.');
    for (let index = 0; index < 4; index += 1) {
      await shortcuts.nth(index).click();
      await page.waitForFunction(expected => {
        const selected = window.__songvizEvidenceTimeline.selectedContext();
        return selected && String(selected.excerpt_id || selected.id) === expected;
      }, await shortcuts.nth(index).getAttribute('data-excerpt-id'), { timeout: 3000 });
    }

    // Snapshot the source object, not presentation text: each select must select
    // a different real record rather than merely repainting a label.
    const contextBefore = await page.evaluate(() => JSON.stringify(window.__songvizEvidenceTimeline.selectedContext()));
    for (const selector of ['#metric', '#stem', '#scale']) {
      const control = page.locator(selector);
      assert(await control.count(), 'Controle ausente: ' + selector);
      const count = await control.locator('option').count();
      assert(count >= 2, selector + ' não oferece uma alternativa real.');
      const current = await control.inputValue();
      const values = await control.locator('option').evaluateAll(options => options.map(option => option.value));
      await control.selectOption(values.find(value => value !== current));
    }
    const contextAfter = await page.evaluate(() => JSON.stringify(window.__songvizEvidenceTimeline.selectedContext()));
    assert.notEqual(contextBefore, contextAfter, 'Seletores de métrica/stem/escala não mudaram o contexto factual.');

    // The end of a supported curve must be unavailable, never copied from the
    // previous value. The builder publishes an explicit probe for this edge.
    const unsupported = await page.evaluate(() => {
      const api = window.__songvizEvidenceTimeline;
      const data = api.data;
      const probes = data.unavailable_probes || data.unsupported_probes || [];
      return probes.find(probe => Number.isFinite(probe.time_s)) || null;
    });
    assert(unsupported, 'Pacote não declarou uma borda de suporte indisponível para verificação.');
    await page.evaluate(probe => window.__songvizEvidenceTimeline.seek(probe.time_s), unsupported);
    await page.waitForFunction(() => {
      const selected = window.__songvizEvidenceTimeline.selectedContext();
      return selected && (selected.available === false || selected.value === null || selected.metric_value === null);
    }, undefined, { timeout: 3000 });
    const unavailableText = await page.locator('#selected-context').innerText();
    assert.match(unavailableText, /indispon[ií]vel|desconhecid/i, 'Borda sem suporte carregou um valor anterior.');

    const pair = await page.evaluate(() => window.__songvizEvidenceTimeline.selectedPair());
    assert(pair, 'Nenhum par de recorrência foi selecionado.');
    const intervals = {
      prior: pair.prior || pair.reference || pair.earlier,
      target: pair.target || pair.current || pair.later,
    };
    for (const [name, interval] of Object.entries(intervals)) {
      assert(interval && Number.isFinite(interval.start_s) && Number.isFinite(interval.end_s) && interval.end_s > interval.start_s,
        'Intervalo ' + name + ' inválido no par de recorrência.');
      const result = await page.evaluate(({ start, end }) => window.__songvizEvidenceTimeline.playRange(start, end),
        { start: interval.start_s, end: interval.end_s });
      assert.notEqual(result, false, 'playRange recusou o intervalo ' + name + '.');
      await page.waitForFunction(({ start }) => Math.abs(document.querySelector('#source-audio').currentTime - start) <= 0.35,
        { start: interval.start_s }, { timeout: 3000 });
    }

    await audio.evaluate(a => a.dispatchEvent(new Event('error')));
    await page.waitForFunction(() => /erro|falh/i.test(document.querySelector('#audio-status')?.textContent || ''),
      undefined, { timeout: 3000 });
    await page.evaluate(() => window.__songvizEvidenceTimeline.retryAudio());
    await page.waitForFunction(() => document.querySelector('#source-audio').readyState >= 1,
      undefined, { timeout: 15000 });

    for (const width of [320, 375, 768]) {
      await page.setViewportSize({ width, height: 812 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true,
        'Overflow horizontal a ' + width + 'px.');
    }
    assert.deepEqual(failures, [], failures.join('\n'));
    console.log('PASS: áudio nativo/fonte, seek, quatro excertos, seletores factuais, borda indisponível, par de recorrência, erro/retry, mobile e sem erros.');
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
