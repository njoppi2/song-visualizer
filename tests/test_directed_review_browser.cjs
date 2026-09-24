/* Browser smoke for the authored 119–132s vocal-emphasis comparison.
 *
 * Usage: SONGVIZ_PLAYWRIGHT_MODULE=/path/to/playwright node tests/test_directed_review_browser.cjs [package-url]
 * The package is intentionally not built by this test; the caller supplies a
 * running static server (or uses the default URL below).
 */
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require(process.env.SONGVIZ_PLAYWRIGHT_MODULE || 'playwright');

const baseUrl = process.argv[2] || 'http://127.0.0.1:8770/directed-vocal-emphasis-02/';
const media = '#passage';

async function metadata(page, timeout = 15000) {
  await page.waitForFunction((selector) => {
    const v = document.querySelector(selector);
    return v && v.readyState >= 1 && Number.isFinite(v.duration) && v.duration > 0;
  }, media, { timeout });
}
async function settled(page) {
  await page.waitForFunction((selector) => {
    const v = document.querySelector(selector);
    return v && !v.seeking;
  }, media);
}
async function state(page, pattern, timeout = 15000) {
  await page.waitForFunction(({ source, flags }) => new RegExp(source, flags).test(document.querySelector('#state')?.textContent || ''), { source: pattern.source, flags: pattern.flags }, { timeout });
}
async function source(page) {
  return page.locator(media).evaluate((v) => v.currentSrc || v.src);
}
async function time(page) {
  return page.locator(media).evaluate((v) => v.currentTime);
}
async function controlsReady(page) {
  await page.waitForFunction(() => {
    const steady = document.querySelector('[data-view="steady"]');
    const reduced = document.querySelector('[data-view="reduced"]');
    return steady && reduced && !steady.disabled && !reduced.disabled;
  });
}

(async () => {
  const browser = await chromium.launch({ headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage({ viewport: { width: 1100, height: 850 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  try {
    // Initial failure must leave a usable retry path and not enable feedback.
    let failInitial = true;
    await page.route('**/directed.mp4', (route) => failInitial ? route.abort() : route.continue());
    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await state(page, /could not load|failed|retry/i);
    assert.equal(await page.locator('#feedback').evaluate((element) => element.disabled), true);
    await controlsReady(page);
    failInitial = false;
    await page.unroute('**/directed.mp4');
    await page.locator('[data-view="reduced"]').click();
    await metadata(page);
    await controlsReady(page);

    assert.equal(await page.locator('[data-view]:visible').count(), 2);
    assert.equal(await page.locator(media).evaluate((v) => v.tagName), 'VIDEO');
    assert.equal(await page.locator(media).evaluate((v) => v.duration), 13);
    assert.match(await source(page), /directed\.mp4$/);
    assert.equal(await page.locator('#feedback').isDisabled(), false);

    // Native playback, pause, and a playing switch preserve clip position.
    await page.locator(media).evaluate((v) => { v.currentTime = 4.25; return v.play(); });
    await page.waitForFunction(() => !document.querySelector('#passage').paused);
    const playingPosition = await time(page);
    await page.locator('[data-view="steady"]').click();
    await metadata(page); await settled(page);
    await page.waitForFunction(() => !document.querySelector('#passage').paused);
    assert.match(await source(page), /steady\.mp4$/);
    assert(Math.abs((await time(page)) - playingPosition) < 0.25, 'playing switch changed position');

    await page.locator(media).evaluate((v) => { v.pause(); v.currentTime = 6.5; });
    await settled(page);
    await page.locator('[data-view="reduced"]').click();
    await metadata(page); await settled(page);
    assert.equal(await page.locator(media).evaluate((v) => v.paused), true);
    assert(Math.abs((await time(page)) - 6.5) < 0.1, 'paused switch changed position');
    assert.match(await source(page), /directed\.mp4$/);

    // Seeking and the authored moment controls map song time to clip offset.
    await page.locator(media).evaluate((v) => { v.currentTime = 0; });
    await page.locator('[data-song-time="123"]').click();
    await page.waitForFunction(() => !document.querySelector('#passage').paused);
    assert(Math.abs((await time(page)) - 4) < 0.15, '123s did not seek to clip offset 4s');
    await page.locator('[data-song-time="127"]').click();
    assert(Math.abs((await time(page)) - 8) < 0.15, '127s did not seek to clip offset 8s');
    await page.locator('#restart').click();
    await page.waitForFunction(() => document.querySelector('#passage').currentTime < 0.2);

    // A stalled metadata request times out, restores the prior view, and
    // re-enables controls. Keep the route held only for this bounded check.
    await page.locator(media).evaluate((v) => { v.pause(); v.currentTime = 3.2; });
    await page.route('**/steady.mp4', (route) => {});
    await page.locator('[data-view="steady"]').click();
    await state(page, /could not load|timed out|retry|restored/i, 20000);
    assert.match(await source(page), /directed\.mp4$/);
    assert.equal(await page.locator('#feedback').isDisabled(), false);
    await page.unroute('**/steady.mp4');

    // An explicit aborted switch also restores the prior clip and position.
    await page.route('**/steady.mp4', (route) => route.abort());
    const restoreStarted = Date.now();
    await page.locator('[data-view="steady"]').click();
    await state(page, /could not load|restored|retry/i);
    assert(Date.now() - restoreStarted < 3000, 'abort recovery exceeded bounded restore time');
    assert.match(await source(page), /directed\.mp4$/);
    assert(Math.abs((await time(page)) - 3.2) < 0.25, 'abort recovery changed position');
    await page.unroute('**/steady.mp4');

    // A media error after metadata must surface retry feedback and disable
    // feedback/export while recovery is in progress.
    await page.locator('[data-view="reduced"]').click();
    await metadata(page); await settled(page); await controlsReady(page);
    await page.locator(media).evaluate((v) => v.dispatchEvent(new Event('error')));
    await state(page, /could not load|retry|restored/i);
    assert.equal(await page.locator('#feedback').evaluate((element) => element.disabled), true);
    await page.locator('[data-view="steady"]').click();
    await metadata(page); await controlsReady(page);
    await page.locator('[data-view="reduced"]').click();
    await metadata(page); await controlsReady(page);

    // Marking/export preserves exact passage identity, view, preference, and
    // package manifest hash. Stub download without writing outside /tmp.
    await page.locator(media).evaluate((v) => { v.pause(); v.currentTime = 4; });
    await page.locator('#mark').click();
    assert.equal(await page.locator('#moment').inputValue(), '123.00');
    await page.locator('#preference').selectOption('reduced');
    await page.locator('#observation').fill('The reduced emphasis keeps the voice present near the verse ending.');
    await page.evaluate(() => {
      window.__testExport = null;
      const objectUrl = URL.createObjectURL;
      URL.createObjectURL = (blob) => { blob.text().then((text) => { window.__testExport = text; }); return objectUrl(blob); };
      HTMLAnchorElement.prototype.click = function () {};
    });
    await page.locator('#export').click();
    await page.waitForFunction(() => window.__testExport !== null);
    const payload = JSON.parse(await page.evaluate(() => window.__testExport));
    assert.match(payload.manifest_sha256, /^[a-f0-9]{64}$/);
    const manifestResponse = await page.request.get(new URL('manifest.json', baseUrl).toString());
    assert.equal(manifestResponse.ok(), true);
    const manifestHash = crypto.createHash('sha256').update(await manifestResponse.body()).digest('hex');
    assert.equal(payload.manifest_sha256, manifestHash);
    assert.equal(payload.observations[0].passage_id, 'verse-ending');
    assert.equal(payload.observations[0].song_time_s, 123);
    assert.equal(payload.observations[0].variant_viewed, 'reduced');
    assert.equal(payload.observations[0].preference, 'reduced');

    await page.screenshot({ path: '/tmp/songviz-directed-vocal-emphasis-desktop.png', fullPage: true });
    for (const width of [320, 375, 768]) {
      await page.setViewportSize({ width, height: 812 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `overflow ${width}`);
      const summary = page.locator('summary');
      if (!(await summary.evaluate((node) => node.parentElement.open))) await summary.click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `expanded details overflow ${width}`);
    }
    await page.setViewportSize({ width: 375, height: 812 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'expanded details overflow');
    await page.screenshot({ path: '/tmp/songviz-directed-vocal-emphasis-mobile.png', fullPage: true });
    assert.deepEqual(errors, []);
    console.log('PASS: authored 119–132s native video, steady/reduced playback and seeking, failure/retry/error recovery, exact feedback export, manifest hash, 320/375/768px overflow, screenshots, no JS errors.');
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exit(1); });
