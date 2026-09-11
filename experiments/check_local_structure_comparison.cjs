/* Usage: SONGVIZ_PLAYWRIGHT_MODULE=/path/to/playwright node experiments/check_local_structure_comparison.cjs URL */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.SONGVIZ_PLAYWRIGHT_MODULE||'playwright');
(async () => {
  const url = process.argv[2];
  assert(url, 'Pass the comparison review URL.');
  const browser = await chromium.launch({headless: true, args: ['--autoplay-policy=no-user-gesture-required']});
  try {
    const context = await browser.newContext({viewport: {width: 1100, height: 800}});
    const page = await context.newPage(), failures = [];
    page.on('pageerror', error => failures.push('pageerror: ' + error.message));
    page.on('console', message => { if (message.type() === 'error') failures.push('console: ' + message.text()); });
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.locator('#song-title').waitFor();
    await page.waitForFunction(() => { const a = document.querySelector('#source-audio'); return a && a.readyState >= 1 && Number.isFinite(a.duration) && a.duration > 0; }, {}, {timeout: 15000});
    assert(await page.locator('#human-table tr').count() > 1, 'human spans not readable');
    assert(await page.locator('#candidate-table tr').count() > 1, 'candidate rows not readable');
    assert(await page.locator('#count-table tr').count() >= 5, 'four proposal-burden rows not readable');
    assert((await page.locator('#candidate-table').innerText()).includes('—'), 'null evidence was rendered as a false numeric zero');
    assert(await page.locator('#source-audio').evaluate(a => a.controls && !!a.currentSrc), 'native audio unavailable');
    await page.locator('#source-audio').evaluate(a => { a.muted = true; });
    await page.locator('button[data-start]').first().click();
    await page.waitForFunction(() => !document.querySelector('#source-audio').paused, {}, {timeout: 5000});
    await page.evaluate(() => window.__songvizLocalComparison.audition(0, .08));
    await page.waitForFunction(() => { const a = document.querySelector('#source-audio'); return a.paused && a.currentTime <= .081; }, {}, {timeout: 5000});
    const plot = page.locator('#comparison-timeline'), box = await plot.boundingBox();
    assert(box, 'timeline has no layout box');
    await plot.click({position: {x: box.width / 2, y: box.height / 2}});
    await page.waitForFunction(() => document.querySelector('#source-audio').currentTime > .25, {}, {timeout: 3000});
    assert((await page.locator('#source-audio').evaluate(a => a.currentTime)) > .25, 'plot click did not seek audio');
    await page.locator('#retry').click();
    await page.waitForFunction(() => document.querySelector('#source-audio').readyState >= 1, {}, {timeout: 5000});
    await page.locator('#source-audio').evaluate(a => { a.currentTime = 10; });
    await plot.focus();
    await page.keyboard.press('ArrowRight');
    const afterKey = await page.locator('#source-audio').evaluate(a => a.currentTime);
    assert(afterKey >= 14.9 && afterKey <= 15.1, 'retry duplicated or lost the five-second keyboard seek');
    await page.setViewportSize({width: 375, height: 812});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, 'mobile horizontal overflow');
    assert.deepEqual(failures, [], failures.join('\n'));
    console.log('PASS: native playback, bounded audition, plot and keyboard seek after retry, unknowns, burden rows, mobile layout, no page errors.');
    await context.close();
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
