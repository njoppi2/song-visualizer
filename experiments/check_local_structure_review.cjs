/* Local smoke test. Usage:
 * SONGVIZ_PLAYWRIGHT_MODULE=/path/to/playwright node experiments/check_local_structure_review.cjs URL
 */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.SONGVIZ_PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const url = process.argv[2];
  assert(url, 'Passe a URL da página de revisão local.');
  const browser = await chromium.launch({headless:true, args:['--autoplay-policy=no-user-gesture-required']});
  try {
    const context = await browser.newContext({viewport:{width:1100,height:800}});
    const page = await context.newPage();
    const failures = [];
    page.on('pageerror', error => failures.push('pageerror: ' + error.message));
    page.on('console', message => { if (message.type() === 'error') failures.push('console: ' + message.text()); });
    await page.goto(url, {waitUntil:'domcontentloaded'});
    await page.locator('#song-title').waitFor();
    assert.notEqual((await page.locator('#song-title').innerText()).trim(), '', 'Título não foi renderizado.');
    await page.waitForFunction(() => { const audio = document.querySelector('#source-audio'); return audio && audio.readyState >= 1 && Number.isFinite(audio.duration) && audio.duration > 0; }, undefined, {timeout:15000});
    assert.equal(await page.locator('#source-audio').evaluate(node => node.controls && !!node.currentSrc), true, 'Controles nativos ou fonte de áudio ausentes.');
    const auditions = page.locator('button[data-start]');
    if (await auditions.count()) {
      const start = await auditions.first().getAttribute('data-start');
      assert(Number(start) > 0, 'A primeira audição precisa iniciar em tempo positivo.');
      await page.locator('#source-audio').evaluate(node => { node.muted = true; });
      await auditions.first().click();
      await page.waitForFunction(() => { const audio = document.querySelector('#source-audio'); return !audio.paused && audio.currentTime > 0; }, undefined, {timeout:5000});
      await page.waitForFunction(() => window.__songvizLocalReview && typeof window.__songvizLocalReview.playRange === 'function');
      await page.evaluate(() => window.__songvizLocalReview.playRange(.05, .12));
      await page.waitForFunction(() => { const audio = document.querySelector('#source-audio'); return audio.paused && audio.currentTime <= .13; }, undefined, {timeout:5000});
    }
    assert.equal(await page.locator('#summary').innerText().then(text => text.trim().length > 0), true, 'Resumo não foi renderizado.');
    assert.equal(await page.locator('#structure-timeline').count(), 1, 'Linha do tempo ausente.');
    await page.setViewportSize({width:375,height:812});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, 'Há overflow horizontal a 375px.');
    assert.deepEqual(failures, [], failures.join('\n'));
    console.log('PASS: título, metadados nativos, audição original, parada limitada, labels e layout móvel.');
    await context.close();
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
